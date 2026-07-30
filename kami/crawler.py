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
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from .paths import (
    OBJECT_ELEMENTS,
    normalize_object_type,
    object_raw_path,
)


SOURCE_URLS = {
    "kamihime": {
        "fire": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E7%81%AB",
        "water": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E6%B0%B4",
        "wind": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%A2%A8",
        "thunder": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%9B%B7",
        "light": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E5%85%89",
        "dark": "https://wikiwiki.jp/kamiprodb/%E7%A5%9E%E5%A7%AB/SSR/%E9%97%87",
    },
    "weapon": {
        "fire": "https://wikiwiki.jp/kamiprodb/%E6%AD%A6%E5%99%A8/SSR/%E7%81%AB",
        "water": "https://wikiwiki.jp/kamiprodb/%E6%AD%A6%E5%99%A8/SSR/%E6%B0%B4",
        "wind": "https://wikiwiki.jp/kamiprodb/%E6%AD%A6%E5%99%A8/SSR/%E9%A2%A8",
        "thunder": "https://wikiwiki.jp/kamiprodb/%E6%AD%A6%E5%99%A8/SSR/%E9%9B%B7",
        "light": "https://wikiwiki.jp/kamiprodb/%E6%AD%A6%E5%99%A8/SSR/%E5%85%89",
        "dark": "https://wikiwiki.jp/kamiprodb/%E6%AD%A6%E5%99%A8/SSR/%E9%97%87",
        "phantom": "https://wikiwiki.jp/kamiprodb/%E6%AD%A6%E5%99%A8/SSR/%E5%B9%BB",
    },
    "eidolon": {
        "fire": "https://wikiwiki.jp/kamiprodb/%E5%B9%BB%E7%8D%A3/SSR/%E7%81%AB",
        "water": "https://wikiwiki.jp/kamiprodb/%E5%B9%BB%E7%8D%A3/SSR/%E6%B0%B4",
        "wind": "https://wikiwiki.jp/kamiprodb/%E5%B9%BB%E7%8D%A3/SSR/%E9%A2%A8",
        "thunder": "https://wikiwiki.jp/kamiprodb/%E5%B9%BB%E7%8D%A3/SSR/%E9%9B%B7",
        "light": "https://wikiwiki.jp/kamiprodb/%E5%B9%BB%E7%8D%A3/SSR/%E5%85%89",
        "dark": "https://wikiwiki.jp/kamiprodb/%E5%B9%BB%E7%8D%A3/SSR/%E9%97%87",
        "phantom": "https://wikiwiki.jp/kamiprodb/%E5%B9%BB%E7%8D%A3/SSR/%E5%B9%BB",
    },
}
DEFAULT_SOURCE_URL = SOURCE_URLS["kamihime"]

ProgressCallback = Callable[[dict[str, Any]], None]
UNLOCK_LINK_FIELDS = {
    "解放武器": "unlock_weapon_url",
    "解放神姫": "unlock_kamihime_url",
}


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


def _capture_unlock_link(
    info: dict,
    header: str,
    value_cell,
    page_url: str,
) -> None:
    field = UNLOCK_LINK_FIELDS.get(header)
    if field is None:
        return
    link = value_cell.find("a", href=True)
    if link is not None:
        info[field] = urljoin(page_url, link.get("href"))


def _catalog_detail_url(source_url: str, href: str | None) -> str:
    if not href:
        return ""
    candidate = urljoin(source_url, href)
    source = urlsplit(source_url)
    parsed = urlsplit(candidate)
    decoded_path = unquote(parsed.path).casefold()
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.netloc.casefold() != source.netloc.casefold()
        or "::cmd" in decoded_path
    ):
        return ""
    return candidate


class KamihimeCrawler:
    object_type = "kamihime"

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
                primary_url = _catalog_detail_url(
                    source_url,
                    image_link.get("href") if image_link else None,
                )
                character_links = [
                    link
                    for link in cells[1].find_all("a", href=True)
                    if _catalog_detail_url(
                        source_url,
                        link.get("href"),
                    )
                ]
                primary_link = next(
                    (
                        link
                        for link in character_links
                        if _catalog_detail_url(
                            source_url,
                            link.get("href"),
                        )
                        == primary_url
                    ),
                    character_links[0] if character_links else None,
                )
                if primary_link is None:
                    continue

                href = _catalog_detail_url(
                    source_url,
                    primary_link.get("href"),
                )
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
                    _capture_unlock_link(
                        result["info"],
                        header,
                        tds[0],
                        page_url,
                    )
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
    def _apply_common_list_metadata(
        record: dict,
        character: dict[str, str],
        object_type: str,
    ) -> dict:
        updated = copy.deepcopy(record)
        info = updated.setdefault("info", {})
        info["name"] = character["name"]
        info["source_url"] = character["link"]
        info["list_image"] = character["list_image"]
        info["release_date"] = character["release_date"]
        info["acquisition_method"] = character["acquisition_method"]
        info["object_type"] = object_type
        return updated

    def apply_list_metadata(
        self,
        record: dict,
        character: dict[str, str],
    ) -> dict:
        return self._apply_common_list_metadata(
            record,
            character,
            self.object_type,
        )

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


class CatalogCrawler(KamihimeCrawler):
    """Shared list/detail behavior for non-Kamihime wiki object catalogs."""

    list_name_header = ""
    list_date_index = 0
    list_acquisition_index = 0

    def extract_character_links(self) -> list[dict[str, str]]:
        objects: list[dict[str, str]] = []
        for source_url in self.source_urls:
            soup = self._get_soup(source_url)
            catalog_table = next(
                (
                    table
                    for table in soup.select("table")
                    if table.find("tr") is not None
                    if self.list_name_header
                    in table.find("tr").get_text(" ", strip=True)
                ),
                None,
            )
            if catalog_table is None:
                raise RuntimeError(
                    f"{self.object_type.title()} list table was not found "
                    f"at {source_url}"
                )

            for row in catalog_table.select("tbody tr"):
                cells = row.find_all("td", recursive=False)
                if len(cells) <= max(
                    1,
                    self.list_date_index,
                    self.list_acquisition_index,
                ):
                    continue
                name_cell = cells[1]
                name_link = next(
                    (
                        link
                        for link in name_cell.find_all("a", href=True)
                        if _catalog_detail_url(
                            source_url,
                            link.get("href"),
                        )
                    ),
                    None,
                )
                if name_link is None:
                    continue
                name = name_link.get_text(" ", strip=True)
                href = _catalog_detail_url(
                    source_url,
                    name_link.get("href"),
                )
                image = cells[0].find("img")
                image_url = (
                    urljoin(source_url, image.get("src"))
                    if image and image.get("src")
                    else ""
                )
                if name and href:
                    objects.append(
                        {
                            "name": name,
                            "link": href,
                            "list_image": image_url,
                            "release_date": cells[
                                self.list_date_index
                            ].get_text(" ", strip=True),
                            "acquisition_method": cells[
                                self.list_acquisition_index
                            ].get_text(" ", strip=True),
                            **self._list_specific_metadata(cells),
                        }
                    )
        return objects

    def _list_specific_metadata(self, cells) -> dict[str, str]:
        return {}

    @staticmethod
    def _detail_table(html: str):
        soup = BeautifulSoup(html, "html.parser")
        table = next(
            (
                candidate
                for candidate in soup.select("table")
                if "基本性能" in candidate.get_text()
            ),
            None,
        )
        if table is None:
            raise RuntimeError("Object detail table did not contain 基本性能")
        return table

    def crawl_character(self, character: dict[str, str]) -> dict:
        delay_min = _env_float("KAMI_CRAWL_DELAY_MIN", 0.8)
        delay_max = _env_float("KAMI_CRAWL_DELAY_MAX", 1.6)
        if delay_max > 0:
            time.sleep(random.uniform(delay_min, max(delay_min, delay_max)))
        soup = self._get_soup(character["link"])
        container = next(
            (
                item
                for item in soup.select("div.h-scrollable")
                if "基本性能" in item.get_text()
            ),
            None,
        )
        if container is None:
            raise RuntimeError(
                f"{self.object_type.title()} data table was not found at "
                f"{character['link']}"
            )
        return self.parse_character(
            str(container),
            character["name"],
            character["link"],
        )


class WeaponCrawler(CatalogCrawler):
    object_type = "weapon"
    list_name_header = "武器名"
    list_date_index = 6
    list_acquisition_index = 7

    def _list_specific_metadata(self, cells) -> dict[str, str]:
        return {
            "weapon_type": cells[3].get_text(" ", strip=True),
            "max_level": cells[4].get_text(" ", strip=True),
            "summary": cells[5].get_text("\n", strip=True),
        }

    def apply_list_metadata(
        self,
        record: dict,
        character: dict[str, str],
    ) -> dict:
        updated = super().apply_list_metadata(record, character)
        info = updated["info"]
        info["weapon_type"] = character.get("weapon_type", "")
        info["max_level"] = character.get("max_level", "")
        info["list_summary"] = character.get("summary", "")
        return updated

    @staticmethod
    def parse_character(html: str, character_name: str, page_url: str) -> dict:
        table = CatalogCrawler._detail_table(html)
        result = {
            "info": {
                "name": character_name,
                "source_url": page_url,
                "object_type": "weapon",
            },
            "stats": [],
            "bursts": [],
            "weapon_skills": [],
            "flavor": "",
        }
        phase = "basic"
        stat_headers: list[str] = []
        current_skill_name = ""
        current_skill_effect = ""

        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            ths = row.find_all("th", recursive=False)
            tds = row.find_all("td", recursive=False)
            first = cells[0].get_text(" ", strip=True)

            if first == "基本性能":
                image = next(
                    (cell.find("img") for cell in tds if cell.find("img")),
                    None,
                )
                if image and image.get("src"):
                    result["info"]["img"] = urljoin(
                        page_url,
                        image.get("src"),
                    )
                continue

            if first == "能力値":
                stat_headers = [
                    cell.get_text(" ", strip=True)
                    for cell in cells[1:]
                ]
                phase = "stats"
                continue
            if first == "バースト":
                phase = "bursts"
                continue
            if first == "スキル":
                phase = "weapon_skills"
                continue

            if phase == "basic" and len(ths) == 1 and len(tds) == 1:
                result["info"][first] = tds[0].get_text(" ", strip=True)
                _capture_unlock_link(
                    result["info"],
                    first,
                    tds[0],
                    page_url,
                )
                continue

            if len(cells) == 1 and cells[0].name == "td":
                result["flavor"] = cells[0].get_text(
                    separator="\n",
                    strip=True,
                )
                continue

            if phase == "stats" and ths and tds:
                values = [cell.get_text(" ", strip=True) for cell in tds]
                stat = {"limit_break": first}
                stat.update(dict(zip(stat_headers, values)))
                result["stats"].append(stat)
                continue

            if phase == "bursts" and ths and tds:
                result["bursts"].append(
                    {
                        "limit_break": first,
                        "effect": tds[0].get_text("\n", strip=True),
                    }
                )
                continue

            if phase == "weapon_skills" and ths and tds:
                values = [
                    cell.get_text("\n", strip=True)
                    for cell in tds
                ]
                max_level = values[0]
                if len(values) >= 3:
                    current_skill_name = values[1]
                    current_skill_effect = values[2]
                result["weapon_skills"].append(
                    {
                        "limit_break": first,
                        "max_level": max_level,
                        "name": current_skill_name,
                        "effect": current_skill_effect,
                    }
                )

        return result


class EidolonCrawler(CatalogCrawler):
    object_type = "eidolon"
    list_name_header = "幻獣名"
    list_date_index = 5
    list_acquisition_index = 6

    def _list_specific_metadata(self, cells) -> dict[str, str]:
        return {
            "max_level": cells[3].get_text(" ", strip=True),
            "summary": cells[4].get_text("\n", strip=True),
        }

    def apply_list_metadata(
        self,
        record: dict,
        character: dict[str, str],
    ) -> dict:
        updated = super().apply_list_metadata(record, character)
        info = updated["info"]
        info["max_level"] = character.get("max_level", "")
        info["list_summary"] = character.get("summary", "")
        return updated

    @staticmethod
    def parse_character(html: str, character_name: str, page_url: str) -> dict:
        table = CatalogCrawler._detail_table(html)
        result = {
            "info": {
                "name": character_name,
                "source_url": page_url,
                "object_type": "eidolon",
            },
            "stats": [],
            "eidolon_effects": [],
            "flavor": "",
        }
        phase = "basic"
        stat_headers: list[str] = []
        current_effect_name = ""
        effect_types = {"召喚効果", "メイン効果", "サブ効果"}

        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            ths = row.find_all("th", recursive=False)
            tds = row.find_all("td", recursive=False)
            first = cells[0].get_text(" ", strip=True)

            if first == "基本性能":
                image = next(
                    (cell.find("img") for cell in tds if cell.find("img")),
                    None,
                )
                if image and image.get("src"):
                    result["info"]["img"] = urljoin(
                        page_url,
                        image.get("src"),
                    )
                continue

            if first == "能力値":
                stat_headers = [
                    cell.get_text(" ", strip=True)
                    for cell in cells[1:]
                ]
                phase = "stats"
                continue
            if first in effect_types:
                phase = first
                current_effect_name = ""
                continue

            if phase == "basic" and len(ths) == 1 and len(tds) == 1:
                result["info"][first] = tds[0].get_text(" ", strip=True)
                continue

            if len(cells) == 1 and cells[0].name == "td":
                result["flavor"] = cells[0].get_text(
                    separator="\n",
                    strip=True,
                )
                continue

            if phase == "stats" and ths and tds:
                values = [cell.get_text(" ", strip=True) for cell in tds]
                stat = {"limit_break": first}
                stat.update(dict(zip(stat_headers, values)))
                result["stats"].append(stat)
                continue

            if phase == "召喚効果" and tds and len(tds) >= 5:
                current_effect_name = tds[0].get_text("\n", strip=True)
                result["eidolon_effects"].append(
                    {
                        "type": phase,
                        "name": current_effect_name,
                        "requirements": tds[1].get_text(" ", strip=True),
                        "effect": tds[2].get_text("\n", strip=True),
                        "interval": tds[3].get_text(" ", strip=True),
                        "duration": tds[4].get_text(" ", strip=True),
                    }
                )
                continue

            if phase in {"メイン効果", "サブ効果"} and tds:
                if len(tds) >= 3:
                    current_effect_name = tds[0].get_text("\n", strip=True)
                    requirements = tds[1].get_text(" ", strip=True)
                    effect = tds[2].get_text("\n", strip=True)
                elif len(tds) == 2:
                    requirements = tds[0].get_text(" ", strip=True)
                    effect = tds[1].get_text("\n", strip=True)
                else:
                    continue
                result["eidolon_effects"].append(
                    {
                        "type": phase,
                        "name": current_effect_name,
                        "requirements": requirements,
                        "effect": effect,
                    }
                )

        return result


CRAWLER_TYPES = {
    "kamihime": KamihimeCrawler,
    "eidolon": EidolonCrawler,
    "weapon": WeaponCrawler,
}


def crawler_class(object_type: str):
    return CRAWLER_TYPES[normalize_object_type(object_type)]


def object_data_path(
    data_dir: Path,
    object_type: str,
    element: str,
) -> Path:
    return object_raw_path(data_dir, object_type, element)


def element_data_path(data_dir: Path, element: str) -> Path:
    """Backward-compatible Kamihime raw-data path."""
    return object_data_path(data_dir, "kamihime", element)


def _source_url(
    object_type: str,
    element: str,
    source_url: str | None = None,
) -> str:
    selected = normalize_object_type(object_type)
    if element not in SOURCE_URLS[selected]:
        valid = ", ".join(SOURCE_URLS[selected])
        raise ValueError(
            f"Unknown {selected} element '{element}'. Valid elements: {valid}"
        )
    specific_env = (
        f"KAMI_SOURCE_URL_{selected.upper()}_{element.upper()}"
    )
    legacy_env = (
        f"KAMI_SOURCE_URL_{element.upper()}"
        if selected == "kamihime"
        else ""
    )
    return (
        source_url
        or os.getenv(specific_env)
        or (os.getenv(legacy_env) if legacy_env else None)
        or SOURCE_URLS[selected][element]
    )


def crawl_object_element_to_jsonl(
    object_type: str,
    element: str,
    data_dir: Path,
    source_url: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> int:
    selected = normalize_object_type(object_type)
    element = element.strip().lower()
    url = _source_url(selected, element, source_url)
    crawler = crawler_class(selected)(
        source_urls=[url],
        headless=os.getenv("KAMI_HEADLESS", "1") != "0",
    )
    try:
        links = crawler.extract_character_links()

        def report(progress: dict[str, Any]) -> None:
            if progress_callback:
                progress_callback(
                    {
                        "object_type": selected,
                        "element": element,
                        **progress,
                    }
                )

        records = crawler.crawl(links, report)
    finally:
        crawler.close()

    if not records:
        raise RuntimeError(
            f"The {selected}/{element} crawler returned no records; "
            "old data was kept"
        )

    for record in records:
        info = record.get("info")
        if isinstance(info, dict):
            info["element"] = element
            info["object_type"] = selected

    _atomic_write_jsonl(
        records,
        object_data_path(data_dir, selected, element),
    )
    return len(records)


def crawl_element_to_jsonl(
    element: str,
    data_dir: Path,
    source_url: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Backward-compatible Kamihime element crawl."""
    return crawl_object_element_to_jsonl(
        "kamihime",
        element,
        data_dir,
        source_url,
        progress_callback,
    )


def update_object_element_latest(
    object_type: str,
    element: str,
    data_dir: Path,
    source_url: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    selected = normalize_object_type(object_type)
    element = element.strip().lower()
    destination = object_data_path(data_dir, selected, element)
    existing_records = _read_jsonl(destination)
    url = _source_url(selected, element, source_url)
    crawler = crawler_class(selected)(
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
                        "object_type": selected,
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
            info = record.setdefault("info", {})
            info["element"] = element
            info["object_type"] = selected
            records.append(record)
    finally:
        crawler.close()

    if not records:
        raise RuntimeError(
            f"The {selected}/{element} latest update returned no records; "
            "old data was kept"
        )

    _atomic_write_jsonl(records, destination)
    return {
        "entries": len(records),
        "new_entries": new_entries,
        "crawled_details": crawled_details,
        "removed_entries": max(0, len(existing_records) - len(records)),
    }


def update_element_latest(
    element: str,
    data_dir: Path,
    source_url: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Backward-compatible Kamihime latest update."""
    return update_object_element_latest(
        "kamihime",
        element,
        data_dir,
        source_url,
        progress_callback,
    )


def configured_elements(object_type: str = "kamihime") -> list[str]:
    selected = normalize_object_type(object_type)
    env_name = (
        "KAMI_ELEMENTS"
        if selected == "kamihime"
        else f"KAMI_{selected.upper()}_ELEMENTS"
    )
    configured = os.getenv(
        env_name,
        ",".join(OBJECT_ELEMENTS[selected]),
    )
    elements = [value.strip().lower() for value in configured.split(",") if value.strip()]
    if not elements:
        raise ValueError(f"{env_name} must contain at least one element")
    invalid = [
        element
        for element in elements
        if element not in OBJECT_ELEMENTS[selected]
    ]
    if invalid:
        valid = ", ".join(OBJECT_ELEMENTS[selected])
        raise ValueError(
            f"Unknown {selected} elements: {', '.join(invalid)}. Valid: {valid}"
        )
    return elements


def crawl_all_object_elements(
    object_type: str,
    data_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    selected = normalize_object_type(object_type)
    elements = configured_elements(selected)

    counts: dict[str, int] = {}
    for element in elements:
        counts[element] = crawl_object_element_to_jsonl(
            selected,
            element,
            data_dir,
            progress_callback=progress_callback,
        )
    return counts


def crawl_all_elements(
    data_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Backward-compatible full Kamihime crawl."""
    return crawl_all_object_elements(
        "kamihime",
        data_dir,
        progress_callback,
    )


def update_all_object_elements_latest(
    object_type: str,
    data_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, int]]:
    selected = normalize_object_type(object_type)
    elements = configured_elements(selected)

    return {
        element: update_object_element_latest(
            selected,
            element,
            data_dir,
            progress_callback=progress_callback,
        )
        for element in elements
    }


def update_all_elements_latest(
    data_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, int]]:
    """Backward-compatible latest Kamihime update."""
    return update_all_object_elements_latest(
        "kamihime",
        data_dir,
        progress_callback,
    )


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
