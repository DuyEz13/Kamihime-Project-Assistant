import hashlib
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "kami" / "data"
RAW_DATA_PATH = BASE_DIR / "kami" / "kamihime_raw.jsonl"
ENGLISH_DATA_PATH = BASE_DIR / "kami" / "kamihime_en.jsonl"
LEGACY_DATA_PATH = BASE_DIR / "kami" / "all_kami_data.jsonl"


def _configured_data_path() -> Path | None:
    configured = os.getenv("KAMI_WIKI_DATA")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else BASE_DIR / path
    return None


def _data_paths() -> list[Path]:
    configured = _configured_data_path()
    if configured:
        return [configured]

    raw_element_paths = sorted(DATA_DIR.glob("kamihime_*_raw.jsonl"))
    if raw_element_paths:
        paths: list[Path] = []
        for raw_path in raw_element_paths:
            english_path = raw_path.with_name(
                raw_path.name.replace("_raw.jsonl", "_en.jsonl")
            )
            paths.append(
                english_path if english_path.exists() else raw_path
            )
        return paths
    if RAW_DATA_PATH.exists():
        return [RAW_DATA_PATH]
    if ENGLISH_DATA_PATH.exists():
        return [ENGLISH_DATA_PATH]
    return [LEGACY_DATA_PATH]


def _data_path() -> Path:
    """Return the first active data path for backward compatibility."""
    return _data_paths()[0]


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    if slug:
        return slug
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"character-{digest}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _name(record: dict[str, Any], index: int) -> str:
    info = record.get("info")
    if isinstance(info, dict):
        value = info.get("name") or info.get("Name")
        if value:
            return str(value)
    return str(record.get("name") or record.get("Name") or f"Character {index + 1}")


def _info_value(info: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = info.get(key)
        if value not in (None, ""):
            return str(value)
    return "-"


def _skill_value(skill: dict[str, Any], *patterns: str) -> str:
    for key, value in skill.items():
        normalized = re.sub(r"[^a-z0-9]+", " ", str(key).casefold()).strip()
        if any(pattern in normalized or pattern in str(key) for pattern in patterns):
            return str(value) if value not in (None, "") else "-"
    return "-"


def _skill_type_and_name(skill: dict[str, Any]) -> tuple[str, str]:
    type_patterns = {
        "Burst": ("burst", "バースト"),
        "Ability": ("ability", "アビリティ"),
        "Assist": ("assist", "アシスト"),
    }
    for label, patterns in type_patterns.items():
        value = _skill_value(skill, *patterns)
        if value != "-":
            return label, value
    return "Skill", "-"


def _skill_effect(skill: dict[str, Any]) -> str:
    for key, value in skill.items():
        normalized = re.sub(r"[^a-z0-9]+", " ", str(key).casefold()).strip()
        if normalized in {"effect", "effects"} or str(key) == "効果":
            return str(value) if value not in (None, "") else "-"
    return "-"


def _prepare_skill_sections(skills: list[Any]) -> list[dict[str, Any]]:
    sections = {
        "Burst": [],
        "Ability": [],
        "Assist": [],
        "Skill": [],
    }
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        skill_type, skill_name = _skill_type_and_name(skill)
        sections[skill_type].append(
            {
                "name": skill_name,
                "requirements": _skill_value(
                    skill,
                    "requirements for acquisition",
                    "acquisition requirements",
                    "acquisition requirement",
                    "習得条件",
                ),
                "interval": _skill_value(
                    skill,
                    "usage interval",
                    "use interval",
                    "cooldown",
                    "使用間隔",
                ),
                "duration": _skill_value(
                    skill,
                    "duration of effect",
                    "effect duration",
                    "効果時間",
                ),
                "effect": _skill_effect(skill),
            }
        )

    return [
        {"type": skill_type, "rows": rows}
        for skill_type, rows in sections.items()
        if rows
    ]


@lru_cache(maxsize=4)
def _load_cached(
    path_signatures: tuple[tuple[str, int], ...],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for path_string, _modified_ns in path_signatures:
        records.extend(_read_jsonl(Path(path_string)))
    used_slugs: dict[str, int] = {}
    characters: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        name = _name(record, index)
        base_slug = _slugify(name)
        used_slugs[base_slug] = used_slugs.get(base_slug, 0) + 1
        suffix = used_slugs[base_slug]
        slug = base_slug if suffix == 1 else f"{base_slug}-{suffix}"

        info = record.get("info") if isinstance(record.get("info"), dict) else {}
        skills = record.get("skills")
        if not isinstance(skills, list):
            skills = record.get("skill")
        if not isinstance(skills, list):
            skills = []

        characters.append(
            {
                "slug": slug,
                "name": name,
                "image": (
                    info.get("image")
                    or info.get("img")
                    or info.get("list_image")
                    or ""
                ),
                "list_image": (
                    info.get("list_image")
                    or info.get("image")
                    or info.get("img")
                    or ""
                ),
                "element": str(info.get("element") or "other").lower(),
                "release_date": _info_value(
                    info,
                    "release_date",
                    "実装日",
                    "Implementation Date",
                    "Release Date",
                ),
                "acquisition_method": _info_value(
                    info,
                    "acquisition_method",
                    "入手方法",
                    "Acquisition Method",
                    "How to Obtain",
                ),
                "info": info,
                "skills": skills,
                "skill_sections": _prepare_skill_sections(skills),
                "flavor": record.get("flavor") or "",
            }
        )
    return tuple(characters)


def load_characters() -> list[dict[str, Any]]:
    paths = _data_paths()
    signatures = tuple(
        (str(path), path.stat().st_mtime_ns if path.exists() else 0)
        for path in paths
    )
    return list(_load_cached(signatures))


def get_character(slug: str) -> dict[str, Any] | None:
    return next(
        (character for character in load_characters() if character["slug"] == slug),
        None,
    )
