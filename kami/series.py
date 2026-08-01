from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


REGISTRY_PATH = Path(__file__).with_name("series_registry.json")
SERIES_MANIFEST_NAME = "series_manifest.json"
CORE_ELEMENTS = ("fire", "water", "wind", "thunder", "light", "dark")
SERIES_INFO_KEYS = {
    "original_name",
    "series_key",
    "series_name",
    "series_aliases",
    "series_expected_elements",
    "series_lifecycle",
    "series_detection",
}


@lru_cache(maxsize=1)
def load_series_registry() -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    entries = payload.get("series") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return ()
    return tuple(entry for entry in entries if isinstance(entry, dict))


def _record_info(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    info = record.get("info")
    return info if isinstance(info, dict) else {}


def _effect_names(record: dict[str, Any] | None) -> list[str]:
    if not isinstance(record, dict):
        return []
    names: list[str] = []
    for section in ("eidolon_effects", "weapon_skills", "bursts", "skill"):
        rows = record.get(section)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("name", "Burst", "Ability", "Assist"):
                value = row.get(key)
                if value not in (None, ""):
                    names.append(str(value))
    return names


def _matches_values(value: str, entry: dict[str, Any], prefix: str) -> bool:
    selectors = {
        "names": entry.get(f"{prefix}_names") or [],
        "prefixes": entry.get(f"{prefix}_prefixes") or [],
        "suffixes": entry.get(f"{prefix}_suffixes") or [],
        "contains": entry.get(f"{prefix}_contains") or [],
        "regexes": entry.get(f"{prefix}_regexes") or [],
    }
    configured = any(selectors.values())
    if not configured:
        return True
    return bool(
        any(value == str(candidate) for candidate in selectors["names"])
        or any(value.startswith(str(candidate)) for candidate in selectors["prefixes"])
        or any(value.endswith(str(candidate)) for candidate in selectors["suffixes"])
        or any(str(candidate) in value for candidate in selectors["contains"])
        or any(re.search(str(pattern), value) for pattern in selectors["regexes"])
    )


def _matches_entry(
    original_name: str,
    entry: dict[str, Any],
    record: dict[str, Any] | None,
) -> bool:
    if not _matches_values(original_name, entry, "source"):
        return False

    info = _record_info(record)
    acquisition = str(
        info.get("acquisition_method")
        or info.get("Acquisition Method")
        or info.get("入手方法")
        or ""
    )
    acquisition_names = entry.get("acquisition_names") or []
    if acquisition_names and acquisition not in {
        str(value) for value in acquisition_names
    }:
        return False

    effect_selectors = any(
        entry.get(f"effect_{suffix}")
        for suffix in ("names", "prefixes", "suffixes", "contains", "regexes")
    )
    if effect_selectors and not any(
        _matches_values(name, entry, "effect")
        for name in _effect_names(record)
    ):
        return False
    return True


def series_metadata(
    object_type: str,
    original_name: str,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not original_name:
        return {}
    for entry in load_series_registry():
        if str(entry.get("object_type") or "") != object_type:
            continue
        if not _matches_entry(original_name, entry, record):
            continue
        aliases = [str(value) for value in entry.get("aliases") or [] if value]
        if original_name not in aliases:
            aliases.append(original_name)
        return {
            "series_key": str(entry.get("series_key") or ""),
            "series_name": str(entry.get("series_name") or ""),
            "series_aliases": aliases,
            "series_expected_elements": [
                str(value).lower()
                for value in entry.get("expected_elements") or []
                if value
            ],
            "series_lifecycle": str(entry.get("lifecycle") or "complete"),
            "series_detection": "registry",
        }
    return {}


def enrich_info_series(
    info: dict[str, Any],
    object_type: str,
    original_name: str | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_name = str(
        original_name
        or info.get("original_name")
        or info.get("name")
        or ""
    )
    if source_name:
        info["original_name"] = source_name
    metadata = series_metadata(object_type, source_name, record)
    if metadata:
        info.update(metadata)
    elif not info.get("series_key"):
        for key in SERIES_INFO_KEYS - {"original_name"}:
            info.pop(key, None)
    return info


def enrich_record_series(
    record: dict[str, Any],
    object_type: str,
    original_name: str | None = None,
    *,
    copy_record: bool = False,
) -> dict[str, Any]:
    updated = copy.deepcopy(record) if copy_record else record
    info = updated.get("info")
    if not isinstance(info, dict):
        info = {}
        updated["info"] = info
    enrich_info_series(info, object_type, original_name, updated)
    return updated


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                values.append(value)
    return values


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _source_url(record: dict[str, Any]) -> str:
    return str(_record_info(record).get("source_url") or "")


def _original_name(record: dict[str, Any]) -> str:
    info = _record_info(record)
    return str(info.get("original_name") or info.get("name") or "")


def series_source_urls(data_dir: Path, object_type: str) -> set[str]:
    urls: set[str] = set()
    for path in (data_dir / object_type).glob("*/raw.jsonl"):
        urls.update(
            source_url
            for record in _read_jsonl(path)
            if (source_url := _source_url(record))
        )
    return urls


def _japanese_runs(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return re.findall(r"[\u3040-\u30ff\u3400-\u9fffー]+", normalized)


def _candidate_signatures(record: dict[str, Any]) -> set[str]:
    name = unicodedata.normalize("NFKC", _original_name(record))
    signatures: set[str] = set()
    bracket = re.match(r"^\[([^]]+)\]", name)
    if bracket and "・" in bracket.group(1):
        stem = bracket.group(1).split("・", 1)[0] + "・"
        if len(stem) >= 3:
            signatures.add(stem)

    code = re.match(r"^([A-Z]{2})\d{2}", name)
    if code:
        signatures.add(f"{code.group(1)}##")

    texts = [name, *_effect_names(record)]
    for text in texts:
        for run in _japanese_runs(text):
            if "・" in run:
                prefix = run.split("・", 1)[0] + "・"
                if len(prefix) >= 3:
                    signatures.add(prefix)
    return signatures


def _signature_key(object_type: str, signature: str) -> str:
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"{object_type}:auto:{digest}"


def detect_series_candidates(
    records: Iterable[dict[str, Any]],
    object_type: str,
) -> list[dict[str, Any]]:
    values = list(records)
    groups: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(values):
        info = _record_info(record)
        if info.get("series_key") and info.get("series_detection") != "auto-high":
            continue
        for signature in _candidate_signatures(record):
            groups[signature].add(index)

    candidates: list[dict[str, Any]] = []
    for signature, indexes in groups.items():
        elements = {
            str(_record_info(values[index]).get("element") or "").lower()
            for index in indexes
        } - {""}
        if len(elements) < 2 or len(indexes) > 24:
            continue
        counts = [
            sum(
                str(_record_info(values[index]).get("element") or "").lower()
                == element
                for index in indexes
            )
            for element in elements
        ]
        balanced = bool(counts) and max(counts) - min(counts) <= 1
        confidence = "high" if len(elements) >= 4 and balanced else "medium"
        candidates.append(
            {
                "series_key": _signature_key(object_type, signature),
                "object_type": object_type,
                "signature": signature,
                "confidence": confidence,
                "elements": sorted(elements),
                "members": sorted(
                    _source_url(values[index]) for index in indexes if _source_url(values[index])
                ),
                "member_details": sorted(
                    (
                        {
                            "source_url": _source_url(values[index]),
                            "original_name": _original_name(values[index]),
                            "element": str(
                                _record_info(values[index]).get("element") or ""
                            ).lower(),
                        }
                        for index in indexes
                    ),
                    key=lambda item: (item["element"], item["original_name"]),
                ),
                "member_indexes": sorted(indexes),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["confidence"] == "high",
            len(item["elements"]),
            len(item["signature"]),
        ),
        reverse=True,
    )
    accepted: list[dict[str, Any]] = []
    seen_members: set[tuple[int, ...]] = set()
    for candidate in candidates:
        member_key = tuple(candidate["member_indexes"])
        if member_key in seen_members:
            continue
        seen_members.add(member_key)
        accepted.append(candidate)
    return accepted


def reconcile_series_data(
    data_dir: Path,
    object_type: str,
    *,
    changed_source_urls: Iterable[str] = (),
    allow_auto_attach: bool = False,
) -> dict[str, Any]:
    changed_urls = {str(value) for value in changed_source_urls if value}
    paths = sorted((data_dir / object_type).glob("*/raw.jsonl"))
    records_by_path = {path: _read_jsonl(path) for path in paths}
    all_records = [record for records in records_by_path.values() for record in records]
    registry_matches = 0
    for record in all_records:
        before = str(_record_info(record).get("series_key") or "")
        enrich_record_series(record, object_type)
        after = str(_record_info(record).get("series_key") or "")
        if after and after != before:
            registry_matches += 1

    candidates = detect_series_candidates(all_records, object_type)
    auto_attached = 0
    if allow_auto_attach and changed_urls:
        for candidate in candidates:
            if candidate["confidence"] != "high":
                continue
            if not changed_urls.intersection(candidate["members"]):
                continue
            existing_metadata = next(
                (
                    {
                        key: value
                        for key, value in _record_info(all_records[index]).items()
                        if key in SERIES_INFO_KEYS - {"original_name"}
                    }
                    for index in candidate["member_indexes"]
                    if _record_info(all_records[index]).get("series_detection")
                    == "auto-high"
                ),
                None,
            )
            for index in candidate["member_indexes"]:
                record = all_records[index]
                info = _record_info(record)
                if info.get("series_key"):
                    continue
                info.update(
                    existing_metadata
                    or {
                        "series_key": candidate["series_key"],
                        "series_name": candidate["signature"],
                        "series_aliases": [candidate["signature"]],
                        "series_expected_elements": list(CORE_ELEMENTS),
                        "series_lifecycle": (
                            "complete"
                            if set(CORE_ELEMENTS) <= set(candidate["elements"])
                            else "releasing"
                        ),
                        "series_detection": "auto-high",
                    }
                )
                auto_attached += 1
            lifecycle = (
                "complete"
                if set(CORE_ELEMENTS) <= set(candidate["elements"])
                else "releasing"
            )
            for index in candidate["member_indexes"]:
                info = _record_info(all_records[index])
                if info.get("series_detection") == "auto-high":
                    info["series_lifecycle"] = lifecycle
            candidate["auto_attached"] = True

    for path, records in records_by_path.items():
        original = _read_jsonl(path)
        if records != original:
            _write_jsonl(path, records)

    manifest_path = data_dir / SERIES_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest[object_type] = [
        {key: value for key, value in candidate.items() if key != "member_indexes"}
        for candidate in candidates
    ]
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return {
        "registry_matches": registry_matches,
        "auto_attached": auto_attached,
        "high_confidence_candidates": sum(
            candidate["confidence"] == "high" for candidate in candidates
        ),
        "medium_confidence_candidates": sum(
            candidate["confidence"] == "medium" for candidate in candidates
        ),
        "manifest": str(manifest_path),
    }
