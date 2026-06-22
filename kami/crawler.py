import copy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from .paths import element_raw_path


DEFAULT_SOURCE_URL = {
    "fire": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E7%81%AB",
    "water": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E6%B0%B4",
    "wind": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%A2%A8",
    "thunder": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%9B%B7",
    "light": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E5%85%89",
    "dark": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%97%87",
}

ProgressCallback = Callable[[dict[str, Any]], None]


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
    base = _env_float("KAMI_HTTP_BACKOFF_BASE", 4.0)
    maximum = _env_float("KAMI_HTTP_BACKOFF_MAX", 180.0)
    jitter_ratio = _env_float("KAMI_HTTP_BACKOFF_JITTER", 0.35)
    cooldown_429 = _env_float("KAMI_HTTP_429_COOLDOWN", 45.0)

    exponential = base * (2 ** attempt)
    if response.status_code == 429:
        exponential = max(exponential, cooldown_429)
    delay = retry_after if retry_after is not None else exponential
    delay = min(delay, maximum)
    if jitter_ratio:
        delay += random.uniform(0, delay * jitter_ratio)
    return delay


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
        max_attempts = _env_int("KAMI_HTTP_RETRIES", 8, minimum=1)
        request_interval = _env_float("KAMI_REQUEST_INTERVAL", 1.2)

        for attempt in range(max_attempts):
            with self._request_lock:
                elapsed = time.monotonic() - self._last_request_at
                if elapsed < request_interval:
                    time.sleep(request_interval - elapsed)
                response = self.client.get(url)
                self._last_request_at = time.monotonic()

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"HTTP {response.status_code} after {max_attempts} "
                        f"attempts while fetching {url}"
                    ) from None
                delay = _retry_delay_seconds(response, attempt)
                time.sleep(delay)
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
        current_skill_icon = ""

        def parse_skill_row(skill_type: str, cells) -> dict | None:
            nonlocal current_skill_icon
            expected_columns = 5 if skill_type == "アビリティ" else 3
            values = [cell.get_text(" ", strip=True) for cell in cells]

            # The first row in an icon group has an extra icon cell. Upgrade
            # rows share that icon via rowspan and therefore omit the cell.
            if len(values) == expected_columns + 1:
                image = cells[0].find("img")
                current_skill_icon = (
                    urljoin(page_url, image.get("src"))
                    if image and image.get("src")
                    else ""
                )
                values = values[1:]
            if len(values) != expected_columns:
                return None

            if skill_type == "アビリティ":
                name, requirement, interval, duration, effect = values
                return {
                    "icon": current_skill_icon,
                    skill_type: name,
                    "習得条件": requirement or "-",
                    "使用間隔": interval,
                    "効果時間": duration,
                    "効果": effect,
                }

            name, requirement, effect = values
            return {
                "icon": current_skill_icon,
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
                    current_skill_icon = ""
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
        delay_min = _env_float("KAMI_CRAWL_DELAY_MIN", 0.8)
        delay_max = _env_float("KAMI_CRAWL_DELAY_MAX", 1.6)
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

    def crawl(
        self,
        links: list[dict[str, str]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict]:
        links = links or self.extract_character_links()
        unique_characters = {
            character["link"]: character
            for character in links
        }
        workers = _env_int("KAMI_CRAWL_WORKERS", 1, minimum=1)
        record_cache: dict[str, dict] = {}
        total = len(unique_characters)
        completed = 0

        def report(character: dict[str, str] | None = None) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "processed": completed,
                    "total": total,
                    "character": character.get("name", "") if character else "",
                    "url": character.get("link", "") if character else "",
                }
            )

        report()

        if workers == 1:
            for url, character in unique_characters.items():
                record_cache[url] = self.crawl_character(character)
                completed += 1
                report(character)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.crawl_character, character): url
                    for url, character in unique_characters.items()
                }
                characters_by_url = {
                    character["link"]: character
                    for character in unique_characters.values()
                }
                for future in as_completed(futures):
                    url = futures[future]
                    record_cache[url] = future.result()
                    completed += 1
                    report(characters_by_url.get(url))

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
    return element_raw_path(data_dir, element)


def crawl_element_to_jsonl(
    element: str,
    data_dir: Path,
    source_url: str | None = None,
    progress_callback: ProgressCallback | None = None,
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
        links = crawler.extract_character_links()

        def report(progress: dict[str, Any]) -> None:
            if progress_callback:
                progress_callback({"element": element, **progress})

        records = crawler.crawl(links, report)
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
    progress_callback: ProgressCallback | None = None,
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
        new_detail_links = {
            entry["link"]
            for entry in entries
            if (entry["link"], entry["name"]) not in existing_by_identity
            and entry["link"] not in existing_by_url
        }
        total_details = len(new_detail_links)

        def report(entry: dict[str, str] | None = None) -> None:
            if progress_callback:
                progress_callback(
                    {
                        "element": element,
                        "processed": crawled_details,
                        "total": total_details,
                        "character": entry.get("name", "") if entry else "",
                        "url": entry.get("link", "") if entry else "",
                    }
                )

        report()

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
                        report(entry)
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


def configured_elements() -> list[str]:
    configured = os.getenv("KAMI_ELEMENTS", ",".join(DEFAULT_SOURCE_URL))
    elements = [value.strip().lower() for value in configured.split(",") if value.strip()]
    if not elements:
        raise ValueError("KAMI_ELEMENTS must contain at least one element")
    return elements


def crawl_all_elements(
    data_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    elements = configured_elements()

    counts: dict[str, int] = {}
    for element in elements:
        counts[element] = crawl_element_to_jsonl(
            element,
            data_dir,
            progress_callback=progress_callback,
        )
    return counts


def update_all_elements_latest(
    data_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, int]]:
    elements = configured_elements()

    return {
        element: update_element_latest(
            element,
            data_dir,
            progress_callback=progress_callback,
        )
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
