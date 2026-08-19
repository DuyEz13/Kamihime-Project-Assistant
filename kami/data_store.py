import hashlib
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import (
    BASE_DIR,
    DATA_DIR,
    LEGACY_RAW_DATA_DIR,
    legacy_element_translation_path,
    normalize_object_type,
    object_translation_path,
    translation_provider_order,
)
from .series import SERIES_INFO_KEYS, enrich_info_series, enrich_record_series

RAW_DATA_PATH = BASE_DIR / "kami" / "kamihime_raw.jsonl"
ENGLISH_DATA_PATH = BASE_DIR / "kami" / "kamihime_en.jsonl"
LEGACY_DATA_PATH = BASE_DIR / "kami" / "all_kami_data.jsonl"
INTERNAL_INFO_KEYS = {
    "name",
    "Name",
    "img",
    "image",
    "list_image",
    "source_url",
    "release_date",
    "acquisition_method",
    "element",
    "object_type",
    "max_level",
    "weapon_type",
    "list_summary",
    "unlock_weapon_url",
    "unlock_kamihime_url",
    *SERIES_INFO_KEYS,
}


def _configured_data_path() -> Path | None:
    configured = os.getenv("KAMI_WIKI_DATA")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else BASE_DIR / path
    return None


def _object_data_paths(object_type: str = "kamihime") -> list[Path]:
    selected_object_type = normalize_object_type(object_type)
    configured = _configured_data_path()
    if configured and selected_object_type == "kamihime":
        return [configured]

    object_root = DATA_DIR / selected_object_type
    raw_element_paths = sorted(object_root.glob("*/raw.jsonl"))
    if raw_element_paths:
        paths: list[Path] = []
        for raw_path in raw_element_paths:
            element = raw_path.parent.name
            translated_paths = [
                object_translation_path(
                    DATA_DIR,
                    selected_object_type,
                    element,
                    provider,
                )
                for provider in translation_provider_order()
            ]
            translated_path = next(
                (path for path in translated_paths if path.exists()),
                None,
            )
            paths.append(translated_path or raw_path)
        return paths

    if selected_object_type != "kamihime":
        return []

    legacy_raw_paths = sorted(
        LEGACY_RAW_DATA_DIR.glob("kamihime_*_raw.jsonl")
    )
    if legacy_raw_paths:
        paths = []
        for raw_path in legacy_raw_paths:
            element = raw_path.name.removeprefix("kamihime_").removesuffix(
                "_raw.jsonl"
            )
            translated_paths = [
                legacy_element_translation_path(
                    DATA_DIR,
                    element,
                    provider,
                )
                for provider in translation_provider_order()
            ]
            translated_path = next(
                (path for path in translated_paths if path.exists()),
                None,
            )
            paths.append(translated_path or raw_path)
        return paths
    if RAW_DATA_PATH.exists():
        return [RAW_DATA_PATH]
    if ENGLISH_DATA_PATH.exists():
        return [ENGLISH_DATA_PATH]
    return [LEGACY_DATA_PATH]


def _object_raw_paths(object_type: str) -> list[Path]:
    return sorted((DATA_DIR / object_type).glob("*/raw.jsonl"))


def _data_paths() -> list[Path]:
    """Return active Kamihime paths for backward compatibility."""
    return _object_data_paths("kamihime")


def _data_path() -> Path:
    """Return the first active data path for backward compatibility."""
    return _data_paths()[0]


def load_object_records(
    object_type: str,
    element: str | None = None,
) -> list[dict[str, Any]]:
    """Load translated records for any supported wiki object type."""
    selected_object_type = normalize_object_type(object_type)
    selected_element = element.strip().lower() if element else None
    records: list[dict[str, Any]] = []
    for path in _object_data_paths(selected_object_type):
        if path.name.startswith("kamihime_"):
            path_element = path.name.removeprefix("kamihime_").split("_", 1)[0]
        elif path.parent.name == "translated":
            path_element = path.parent.parent.name
        else:
            path_element = path.parent.name
        if selected_element and path_element != selected_element:
            continue
        records.extend(_read_jsonl(path))
    return records


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


def _display_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if key not in INTERNAL_INFO_KEYS
    }


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


def _skill_type(skill: dict[str, Any]) -> str:
    type_patterns = {
        "Burst": ("burst", "バースト"),
        "Ability": ("ability", "アビリティ"),
        "Assist": ("assist", "アシスト"),
    }
    for key in skill:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(key).casefold()).strip()
        for label, patterns in type_patterns.items():
            if any(
                pattern in normalized or pattern in str(key) for pattern in patterns
            ):
                return label
    return "Skill"


def _skill_effect(skill: dict[str, Any]) -> str:
    for key, value in skill.items():
        normalized = re.sub(r"[^a-z0-9]+", " ", str(key).casefold()).strip()
        if normalized in {"effect", "effects"} or str(key) == "効果":
            return str(value) if value not in (None, "") else "-"
    return "-"


def _prepare_skill_sections(
    skills: list[Any],
    note_image: str = "",
) -> list[dict[str, Any]]:
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
        detected_type = _skill_type(skill)
        effect = _skill_effect(skill)
        if detected_type != "Skill" and skill_name == "-":
            if effect != "-":
                sections[detected_type].append(
                    {
                        "is_note": True,
                        "note": effect,
                        "image": note_image,
                    }
                )
            continue
        sections[skill_type].append(
            {
                "is_note": False,
                "icon": str(skill.get("icon") or skill.get("Icon") or ""),
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
                "effect": effect,
            }
        )

    return [
        {"type": skill_type, "rows": rows}
        for skill_type, rows in sections.items()
        if rows
    ]


def _is_zero_number(value: Any) -> bool:
    normalized = str(value or "").strip().replace(",", "")
    return bool(re.fullmatch(r"0+(?:\.0+)?", normalized))


def _weapon_stat_is_placeholder(row: dict[str, Any]) -> bool:
    hp = row.get("HP") if "HP" in row else row.get("hp")
    attack = row.get("Attack")
    if attack in (None, ""):
        attack = row.get("攻撃力")
    return (
        hp not in (None, "")
        and attack not in (None, "")
        and (_is_zero_number(hp) and _is_zero_number(attack))
    )


def _prepare_weapon_rows(
    stats: Any,
    bursts: Any,
    weapon_skills: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_stats = (
        [
            dict(row)
            for row in stats
            if isinstance(row, dict) and not _weapon_stat_is_placeholder(row)
        ]
        if isinstance(stats, list)
        else []
    )
    valid_bursts = (
        [
            dict(row)
            for row in bursts
            if isinstance(row, dict) and str(row.get("effect") or "").strip()
        ]
        if isinstance(bursts, list)
        else []
    )
    valid_skills = (
        [
            dict(row)
            for row in weapon_skills
            if isinstance(row, dict)
            and (
                str(row.get("name") or "").strip()
                or str(row.get("effect") or "").strip()
            )
        ]
        if isinstance(weapon_skills, list)
        else []
    )
    return valid_stats, valid_bursts, valid_skills


def _prepare_eidolon_effects(effects: list[Any]) -> list[dict[str, Any]]:
    rows = [
        dict(effect)
        for effect in effects
        if isinstance(effect, dict)
    ]
    for row in rows:
        row.update(
            {
                "show_type": False,
                "type_rowspan": 0,
                "show_name": False,
                "name_rowspan": 0,
            }
        )

    index = 0
    while index < len(rows):
        group_type = str(rows[index].get("type") or "")
        end = index + 1
        while (
            end < len(rows)
            and str(rows[end].get("type") or "") == group_type
        ):
            end += 1
        rows[index]["show_type"] = True
        rows[index]["type_rowspan"] = end - index
        index = end

    index = 0
    while index < len(rows):
        group_key = (
            str(rows[index].get("type") or ""),
            str(rows[index].get("name") or ""),
        )
        end = index + 1
        while end < len(rows):
            candidate_key = (
                str(rows[end].get("type") or ""),
                str(rows[end].get("name") or ""),
            )
            if candidate_key != group_key:
                break
            end += 1
        rows[index]["show_name"] = True
        rows[index]["name_rowspan"] = end - index
        index = end
    return rows


def _is_summon_effect(effect: dict[str, Any]) -> bool:
    effect_type = str(effect.get("type") or "")
    return (
        "summon" in effect_type.casefold()
        or "召喚" in effect_type
    )


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
                "object_type": "kamihime",
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
                "display_info": _display_info(info),
                "skills": skills,
                "skill_sections": _prepare_skill_sections(
                    skills,
                    note_image=str(
                        info.get("list_image")
                        or info.get("image")
                        or info.get("img")
                        or ""
                    ),
                ),
                "has_skill_icons": any(
                    isinstance(skill, dict)
                    and bool(skill.get("icon") or skill.get("Icon"))
                    for skill in skills
                ),
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


@lru_cache(maxsize=12)
def _load_catalog_cached(
    object_type: str,
    path_signatures: tuple[tuple[str, int], ...],
    raw_path_signatures: tuple[tuple[str, int], ...] = (),
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for path_string, _modified_ns in path_signatures:
        records.extend(_read_jsonl(Path(path_string)))
    raw_info_by_url: dict[str, dict[str, Any]] = {}
    for path_string, _modified_ns in raw_path_signatures:
        for raw_record in _read_jsonl(Path(path_string)):
            enrich_record_series(raw_record, object_type)
            raw_info = (
                raw_record.get("info")
                if isinstance(raw_record.get("info"), dict)
                else {}
            )
            source_url = str(raw_info.get("source_url") or "")
            original_name = str(
                raw_info.get("original_name") or raw_info.get("name") or ""
            )
            if source_url:
                raw_info_by_url[source_url] = {
                    key: value
                    for key, value in raw_info.items()
                    if key in SERIES_INFO_KEYS
                }
                if original_name:
                    raw_info_by_url[source_url]["original_name"] = original_name

    used_slugs: dict[str, int] = {}
    objects: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        name = _name(record, index)
        base_slug = _slugify(name)
        used_slugs[base_slug] = used_slugs.get(base_slug, 0) + 1
        suffix = used_slugs[base_slug]
        slug = base_slug if suffix == 1 else f"{base_slug}-{suffix}"
        info = (
            dict(record.get("info"))
            if isinstance(record.get("info"), dict)
            else {}
        )
        source_url = str(info.get("source_url") or "")
        raw_series_info = raw_info_by_url.get(source_url, {})
        for key, value in raw_series_info.items():
            if value not in (None, "", []):
                info[key] = value
        original_name = str(
            info.get("original_name")
            or info.get("name")
            or ""
        )
        enrich_info_series(info, object_type, original_name)
        stats = record.get("stats")
        bursts = record.get("bursts")
        weapon_skills = record.get("weapon_skills")
        if object_type == "weapon":
            stats, bursts, weapon_skills = _prepare_weapon_rows(
                stats,
                bursts,
                weapon_skills,
            )
        eidolon_effects = record.get("eidolon_effects")
        if isinstance(eidolon_effects, list):
            prepared_eidolon_effects = _prepare_eidolon_effects(
                eidolon_effects
            )
            eidolon_summon_effects = [
                row
                for row in prepared_eidolon_effects
                if _is_summon_effect(row)
            ]
            eidolon_passive_effects = _prepare_eidolon_effects(
                [
                    row
                    for row in eidolon_effects
                    if (
                        isinstance(row, dict)
                        and not _is_summon_effect(row)
                    )
                ]
            )
        else:
            prepared_eidolon_effects = []
            eidolon_summon_effects = []
            eidolon_passive_effects = []

        objects.append(
            {
                "slug": slug,
                "object_type": object_type,
                "name": name,
                "original_name": str(info.get("original_name") or ""),
                "series_key": str(info.get("series_key") or ""),
                "series_name": str(info.get("series_name") or ""),
                "series_aliases": list(info.get("series_aliases") or []),
                "series_expected_elements": list(
                    info.get("series_expected_elements") or []
                ),
                "series_lifecycle": str(
                    info.get("series_lifecycle") or "complete"
                ),
                "series_detection": str(info.get("series_detection") or ""),
                "image": str(
                    info.get("image")
                    or info.get("img")
                    or info.get("list_image")
                    or ""
                ),
                "list_image": str(
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
                "display_info": _display_info(info),
                "stats": stats if isinstance(stats, list) else [],
                "bursts": bursts if isinstance(bursts, list) else [],
                "weapon_skills": (
                    weapon_skills if isinstance(weapon_skills, list) else []
                ),
                "eidolon_effects": (
                    prepared_eidolon_effects
                ),
                "eidolon_summon_effects": eidolon_summon_effects,
                "eidolon_passive_effects": eidolon_passive_effects,
                "flavor": record.get("flavor") or "",
            }
        )
    series_groups: dict[str, list[dict[str, Any]]] = {}
    for item in objects:
        key = str(item.get("series_key") or "")
        if key:
            series_groups.setdefault(key, []).append(item)
    for members in series_groups.values():
        elements = list(
            dict.fromkeys(
                str(member.get("element") or "")
                for member in members
                if member.get("element")
            )
        )
        slugs = [str(member.get("slug") or "") for member in members]
        for member in members:
            member["series_catalog_elements"] = elements
            member["series_catalog_member_count"] = len(members)
            member["series_catalog_slugs"] = slugs
    return tuple(objects)


def load_catalog_items(
    object_type: str,
    element: str | None = None,
) -> list[dict[str, Any]]:
    selected_object_type = normalize_object_type(object_type)
    if selected_object_type == "kamihime":
        objects = load_characters()
    else:
        paths = _object_data_paths(selected_object_type)
        signatures = tuple(
            (str(path), path.stat().st_mtime_ns if path.exists() else 0)
            for path in paths
        )
        raw_signatures = tuple(
            (str(path), path.stat().st_mtime_ns if path.exists() else 0)
            for path in _object_raw_paths(selected_object_type)
        )
        objects = list(
            _load_catalog_cached(
                selected_object_type,
                signatures,
                raw_signatures,
            )
        )

    if element is None:
        return objects
    selected_element = element.strip().lower()
    return [
        item
        for item in objects
        if item.get("element") == selected_element
    ]


def get_catalog_item(
    object_type: str,
    slug: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in load_catalog_items(object_type)
            if item["slug"] == slug
        ),
        None,
    )
