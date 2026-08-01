from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kami.paths import DATA_DIR, OBJECT_ELEMENTS  # noqa: E402
from kami.series import SERIES_INFO_KEYS, enrich_record_series  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                record = json.loads(value)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _source_identity(record: dict[str, Any]) -> tuple[str, str]:
    info = record.get("info") if isinstance(record.get("info"), dict) else {}
    return (
        str(info.get("source_url") or ""),
        str(info.get("original_name") or info.get("name") or ""),
    )


def _backfill_object_type(object_type: str) -> dict[str, int]:
    raw_series: dict[str, dict[str, Any]] = {}
    raw_files = 0
    translated_files = 0
    records_changed = 0

    for element in OBJECT_ELEMENTS[object_type]:
        path = DATA_DIR / object_type / element / "raw.jsonl"
        records = _read_jsonl(path)
        if not records:
            continue
        changed = False
        for record in records:
            source_url, original_name = _source_identity(record)
            before = json.dumps(record.get("info") or {}, ensure_ascii=False)
            enrich_record_series(record, object_type, original_name)
            info = record.get("info") if isinstance(record.get("info"), dict) else {}
            after = json.dumps(record.get("info") or {}, ensure_ascii=False)
            if source_url:
                raw_series[source_url] = {
                    key: value for key, value in info.items() if key in SERIES_INFO_KEYS
                }
            if before != after:
                records_changed += 1
                changed = True
        if changed:
            _write_jsonl(path, records)
            raw_files += 1

    for element in OBJECT_ELEMENTS[object_type]:
        directory = DATA_DIR / object_type / element / "translated"
        for path in sorted(directory.glob("*.jsonl")):
            records = _read_jsonl(path)
            changed = False
            for record in records:
                source_url, stored_original_name = _source_identity(record)
                raw_metadata = raw_series.get(source_url, {})
                original_name = str(
                    raw_metadata.get("original_name") or stored_original_name
                )
                before = json.dumps(record.get("info") or {}, ensure_ascii=False)
                info = record.get("info") if isinstance(record.get("info"), dict) else {}
                record["info"] = info
                for key, value in raw_metadata.items():
                    info[key] = value
                enrich_record_series(record, object_type, original_name)
                after = json.dumps(record.get("info") or {}, ensure_ascii=False)
                if before != after:
                    records_changed += 1
                    changed = True
            if changed:
                _write_jsonl(path, records)
                translated_files += 1

    return {
        "raw_files": raw_files,
        "translated_files": translated_files,
        "records_changed": records_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill extensible series metadata into raw and translated data"
    )
    parser.add_argument(
        "--object-type",
        action="append",
        choices=("weapon", "eidolon"),
        dest="object_types",
        help="Object type to update; repeat to select both (default: both)",
    )
    args = parser.parse_args()
    selected = args.object_types or ["weapon", "eidolon"]
    for object_type in dict.fromkeys(selected):
        result = _backfill_object_type(object_type)
        print(
            f"{object_type}: {result['records_changed']} records changed "
            f"across {result['raw_files']} raw and "
            f"{result['translated_files']} translated files"
        )


if __name__ == "__main__":
    main()
