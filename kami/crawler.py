import copy
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup


DEFAULT_SOURCE_URL = {
    "fire": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E7%81%AB",
    "water": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E6%B0%B4",
    "wind": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%A2%A8",
    "thunder": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%9B%B7",
    "light": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E5%85%89",
    "dark": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%97%87",
}


def _atomic_write_jsonl(records: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


class KamihimeCrawler:
    def __init__(self, source_urls: list[str], headless: bool = True, wait_s: int = 15):
        del headless
        self.source_urls = source_urls
        self.wait_s = wait_s
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0
        self.client = httpx.Client(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/137.0 Safari/537.36"
                ),
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            },
            follow_redirects=True,
            timeout=wait_s,
        )

    def close(self) -> None:
        self.client.close()

    def _get_soup(self, url: str) -> BeautifulSoup:
        max_attempts = max(1, int(os.getenv("KAMI_HTTP_RETRIES", "5")))
        request_interval = max(
            0.0,
            float(os.getenv("KAMI_REQUEST_INTERVAL", "0.4")),
        )

        for attempt in range(max_attempts):
            with self._request_lock:
                elapsed = time.monotonic() - self._last_request_at
                if elapsed < request_interval:
                    time.sleep(request_interval - elapsed)
                response = self.client.get(url)
                self._last_request_at = time.monotonic()

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_attempts - 1:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2 ** (attempt + 1)
                time.sleep(min(delay, 30))
                continue

            response.raise_for_status()
            if "Sorry, you have been blocked" in response.text:
                raise RuntimeError(f"Cloudflare blocked the HTTP crawler at {url}")
            return BeautifulSoup(response.text, "html.parser")

        raise RuntimeError(f"Failed to fetch {url}")

    def extract_character_links(self) -> list[dict[str, str]]:
        characters: list[dict[str, str]] = []
        for source_url in self.source_urls:
            soup = self._get_soup(source_url)
            candidate_tables = soup.select("table")
            character_table = next(
                (
                    table
                    for table in candidate_tables
                    if "キャラクター" in table.get_text()
                    and (
                        "実装日" in table.get_text()
                        or "入手方法" in table.get_text()
                    )
                ),
                None,
            )
            if character_table is None:
                raise RuntimeError(
                    f"Character list table was not found at {source_url}"
                )

            for row in character_table.select("tbody tr"):
                cells = row.find_all("td", recursive=False)
                if len(cells) < 9:
                    continue

                image = cells[0].find("img")
                image_url = ""
                if image and image.get("src"):
                    image_url = urljoin(source_url, image.get("src"))

                release_date = cells[7].get_text(" ", strip=True)
                acquisition_method = cells[8].get_text(" ", strip=True)

                image_link = cells[0].find("a", href=True)
                primary_url = (
                    urljoin(source_url, image_link.get("href"))
                    if image_link
                    else ""
                )
                character_links = cells[1].find_all("a", href=True)
                primary_link = next(
                    (
                        link
                        for link in character_links
                        if urljoin(source_url, link.get("href")) == primary_url
                    ),
                    character_links[0] if character_links else None,
                )
                if primary_link is None:
                    continue

                href = urljoin(source_url, primary_link.get("href"))
                name = primary_link.get_text(" ", strip=True)
                if href and name:
                    characters.append(
                        {
                            "name": name,
                            "link": href,
                            "list_image": image_url,
                            "release_date": release_date,
                            "acquisition_method": acquisition_method,
                        }
                    )

        return characters

    @staticmethod
    def parse_character(html: str, character_name: str, page_url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        result = {
            "info": {"name": character_name, "source_url": page_url},
            "skill": [],
            "flavor": "",
        }
        current_skill_type = None

        def parse_skill_row(skill_type: str, cells) -> dict | None:
            expected_columns = 5 if skill_type == "アビリティ" else 3
            values = [cell.get_text(" ", strip=True) for cell in cells]

            # The first row in an icon group has an extra icon cell. Upgrade
            # rows share that icon via rowspan and therefore omit the cell.
            if len(values) == expected_columns + 1:
                values = values[1:]
            if len(values) != expected_columns:
                return None

            if skill_type == "アビリティ":
                name, requirement, interval, duration, effect = values
                return {
                    skill_type: name,
                    "習得条件": requirement or "-",
                    "使用間隔": interval,
                    "効果時間": duration,
                    "効果": effect,
                }

            name, requirement, effect = values
            return {
                skill_type: name,
                "習得条件": requirement or "-",
                "効果": effect,
            }

        for row in soup.find_all("tr"):
            ths = row.find_all("th", recursive=False)
            tds = row.find_all("td", recursive=False)

            if len(tds) == 1 and tds[0].get("colspan") == "6":
                result["flavor"] = tds[0].get_text(separator="\n", strip=True)
                continue

            if ths:
                header = ths[0].get_text(strip=True)
                if header == "基本情報" and tds:
                    image = tds[0].find("img")
                    if image and image.get("src"):
                        result["info"]["img"] = urljoin(
                            page_url, image.get("src")
                        )
                    continue

                skill_type = next(
                    (
                        value
                        for value in ("バースト", "アビリティ", "アシスト")
                        if value in header
                    ),
                    None,
                )
                if skill_type:
                    current_skill_type = skill_type
                    continue

                if len(ths) == 1 and len(tds) == 1:
                    value = tds[0].get_text(" ", strip=True)
                    result["info"][header] = value
                    continue

            if current_skill_type and not ths:
                skill = parse_skill_row(current_skill_type, tds)
                if skill is not None:
                    result["skill"].append(skill)

        return result

    def crawl_character(self, character: dict[str, str]) -> dict:
        delay_min = float(os.getenv("KAMI_CRAWL_DELAY_MIN", "0.1"))
        delay_max = float(os.getenv("KAMI_CRAWL_DELAY_MAX", "0.3"))
        if delay_max > 0:
            time.sleep(random.uniform(delay_min, max(delay_min, delay_max)))
        soup = self._get_soup(character["link"])
        containers = soup.select("div.h-scrollable")
        character_table = next(
            (
                container
                for container in containers
                if "基本情報" in container.get_text()
            ),
            None,
        )
        if character_table is None:
            raise RuntimeError(
                f"Character data table was not found at {character['link']}"
            )
        return self.parse_character(
            str(character_table),
            character["name"],
            character["link"],
        )

    @staticmethod
    def apply_list_metadata(record: dict, character: dict[str, str]) -> dict:
        updated = copy.deepcopy(record)
        info = updated.setdefault("info", {})
        info["name"] = character["name"]
        info["source_url"] = character["link"]
        info["list_image"] = character["list_image"]
        info["release_date"] = character["release_date"]
        info["acquisition_method"] = character["acquisition_method"]
        return updated

    def crawl(self, links: list[dict[str, str]] | None = None) -> list[dict]:
        links = links or self.extract_character_links()
        unique_characters = {
            character["link"]: character
            for character in links
        }
        workers = max(1, int(os.getenv("KAMI_CRAWL_WORKERS", "4")))
        record_cache: dict[str, dict] = {}

        if workers == 1:
            for url, character in unique_characters.items():
                record_cache[url] = self.crawl_character(character)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.crawl_character, character): url
                    for url, character in unique_characters.items()
                }
                for future in as_completed(futures):
                    url = futures[future]
                    record_cache[url] = future.result()

        records = []
        for character in links:
            records.append(
                self.apply_list_metadata(
                    record_cache[character["link"]],
                    character,
                )
            )
        return records


def element_data_path(data_dir: Path, element: str) -> Path:
    return data_dir / f"kamihime_{element}_raw.jsonl"


def crawl_element_to_jsonl(
    element: str,
    data_dir: Path,
    source_url: str | None = None,
) -> int:
    element = element.strip().lower()
    if element not in DEFAULT_SOURCE_URL:
        valid = ", ".join(DEFAULT_SOURCE_URL)
        raise ValueError(f"Unknown element '{element}'. Valid elements: {valid}")

    env_name = f"KAMI_SOURCE_URL_{element.upper()}"
    url = source_url or os.getenv(env_name) or DEFAULT_SOURCE_URL[element]
    crawler = KamihimeCrawler(
        source_urls=[url],
        headless=os.getenv("KAMI_HEADLESS", "1") != "0",
    )
    try:
        records = crawler.crawl()
    finally:
        crawler.close()

    if not records:
        raise RuntimeError(
            f"The {element} crawler returned no characters; old data was kept"
        )

    for record in records:
        info = record.get("info")
        if isinstance(info, dict):
            info["element"] = element

    _atomic_write_jsonl(records, element_data_path(data_dir, element))
    return len(records)


def update_element_latest(
    element: str,
    data_dir: Path,
    source_url: str | None = None,
) -> dict[str, int]:
    element = element.strip().lower()
    if element not in DEFAULT_SOURCE_URL:
        valid = ", ".join(DEFAULT_SOURCE_URL)
        raise ValueError(f"Unknown element '{element}'. Valid elements: {valid}")

    destination = element_data_path(data_dir, element)
    existing_records = _read_jsonl(destination)
    env_name = f"KAMI_SOURCE_URL_{element.upper()}"
    url = source_url or os.getenv(env_name) or DEFAULT_SOURCE_URL[element]
    crawler = KamihimeCrawler(
        source_urls=[url],
        headless=os.getenv("KAMI_HEADLESS", "1") != "0",
    )

    try:
        entries = crawler.extract_character_links()
        existing_by_identity: dict[tuple[str, str], dict] = {}
        existing_by_url: dict[str, dict] = {}
        for record in existing_records:
            info = record.get("info") if isinstance(record.get("info"), dict) else {}
            source = str(info.get("source_url") or "")
            name = str(info.get("name") or "")
            existing_by_identity.setdefault((source, name), record)
            if source and source not in existing_by_url:
                existing_by_url[source] = record

        records: list[dict] = []
        new_entries = 0
        crawled_details = 0
        new_detail_cache: dict[str, dict] = {}

        for entry in entries:
            identity = (entry["link"], entry["name"])
            if identity in existing_by_identity:
                base_record = existing_by_identity[identity]
            else:
                base_record = existing_by_url.get(entry["link"])
                if base_record is None:
                    new_entries += 1
                    if entry["link"] not in new_detail_cache:
                        new_detail_cache[entry["link"]] = crawler.crawl_character(entry)
                        crawled_details += 1
                    base_record = new_detail_cache[entry["link"]]

            record = crawler.apply_list_metadata(base_record, entry)
            record.setdefault("info", {})["element"] = element
            records.append(record)
    finally:
        crawler.close()

    if not records:
        raise RuntimeError(
            f"The {element} latest update returned no characters; old data was kept"
        )

    _atomic_write_jsonl(records, destination)
    return {
        "entries": len(records),
        "new_entries": new_entries,
        "crawled_details": crawled_details,
        "removed_entries": max(0, len(existing_records) - len(records)),
    }


def crawl_all_elements(data_dir: Path) -> dict[str, int]:
    configured = os.getenv("KAMI_ELEMENTS", ",".join(DEFAULT_SOURCE_URL))
    elements = [value.strip().lower() for value in configured.split(",") if value.strip()]
    if not elements:
        raise ValueError("KAMI_ELEMENTS must contain at least one element")

    counts: dict[str, int] = {}
    for element in elements:
        counts[element] = crawl_element_to_jsonl(element, data_dir)
    return counts


def update_all_elements_latest(data_dir: Path) -> dict[str, dict[str, int]]:
    configured = os.getenv("KAMI_ELEMENTS", ",".join(DEFAULT_SOURCE_URL))
    elements = [value.strip().lower() for value in configured.split(",") if value.strip()]
    if not elements:
        raise ValueError("KAMI_ELEMENTS must contain at least one element")

    return {
        element: update_element_latest(element, data_dir)
        for element in elements
    }


def crawl_to_jsonl(destination: Path) -> int:
    """Backward-compatible single-file crawl for callers outside the web pipeline."""
    data_dir = destination.parent
    counts = crawl_all_elements(data_dir)
    records: list[dict] = []
    for element in counts:
        path = element_data_path(data_dir, element)
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    _atomic_write_jsonl(records, destination)
    return len(records)
