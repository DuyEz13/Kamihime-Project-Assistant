from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.documents import Document

from ..data_store import load_catalog_items


CatalogLoader = Callable[[str], list[dict[str, Any]]]
OBJECT_TYPES = ("kamihime", "eidolon", "weapon")


def _value(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _row_text(row: dict[str, Any]) -> str:
    return " | ".join(
        f"{str(key).replace('_', ' ').title()}: {_value(value)}"
        for key, value in row.items()
        if key not in {"icon", "show_type", "type_rowspan", "show_name", "name_rowspan"}
        and value not in (None, "")
    )


def _metadata(item: dict[str, Any], section: str, index: int) -> dict[str, Any]:
    object_type = str(item.get("object_type") or "kamihime")
    slug = str(item.get("slug") or "")
    return {
        "document_id": f"{object_type}:{slug}:{section}:{index}",
        "object_type": object_type,
        "slug": slug,
        "name": str(item.get("name") or ""),
        "normalized_name": str(item.get("name") or "").casefold(),
        "element": str(item.get("element") or ""),
        "series_key": str(item.get("series_key") or ""),
        "series_name": str(item.get("series_name") or ""),
        "series_lifecycle": str(item.get("series_lifecycle") or "complete"),
        "series_catalog_elements": list(item.get("series_catalog_elements") or []),
        "series_catalog_member_count": int(item.get("series_catalog_member_count") or 0),
        "section": section,
        "local_url": f"/objects/{object_type}/{slug}",
        "source_url": str((item.get("info") or {}).get("source_url") or ""),
    }


def _base_lines(item: dict[str, Any]) -> list[str]:
    values = [
        ("Name", item.get("name")),
        ("Original Name", item.get("original_name")),
        ("Object Type", item.get("object_type")),
        ("Series", item.get("series_name")),
        ("Series Key", item.get("series_key")),
        ("Series Lifecycle", item.get("series_lifecycle")),
        (
            "Series Aliases",
            ", ".join(item.get("series_aliases") or []),
        ),
    ]
    display_info = item.get("display_info")
    if isinstance(display_info, dict):
        values.extend(
            (str(key), value)
            for key, value in display_info.items()
            if value not in (None, "")
        )
    values.extend(
        [
            ("Element", item.get("element")),
            ("Release Date", item.get("release_date")),
            ("Acquisition Method", item.get("acquisition_method")),
        ]
    )
    lines: list[str] = []
    seen: set[str] = set()
    for key, value in values:
        normalized_key = " ".join(str(key).casefold().split())
        if normalized_key in seen or value in (None, ""):
            continue
        seen.add(normalized_key)
        lines.append(f"{key}: {_value(value)}")
    flavor = item.get("flavor")
    if flavor:
        lines.append(f"Flavor: {flavor}")
    return lines


def _section_document(
    item: dict[str, Any],
    section: str,
    rows: list[dict[str, Any]],
) -> Document | None:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return None
    object_label = str(item.get("object_type") or "object").title()
    lines = [
        f"{object_label}: {item.get('name')}",
        f"Element: {item.get('element')}",
        f"{section} (complete section, {len(valid_rows)} rows):",
    ]
    lines.extend(
        f"Row {index}: {_row_text(row)}"
        for index, row in enumerate(valid_rows, start=1)
    )
    return Document(
        page_content="\n".join(lines),
        metadata=_metadata(item, section, 0),
    )


def _append_section(
    docs: list[Document],
    item: dict[str, Any],
    section: str,
    rows: list[dict[str, Any]],
) -> None:
    document = _section_document(item, section, rows)
    if document is not None:
        docs.append(document)


def object_documents(item: dict[str, Any]) -> list[Document]:
    docs = [
        Document(
            page_content="\n".join(_base_lines(item)),
            metadata=_metadata(item, "basic", 0),
        )
    ]
    object_type = str(item.get("object_type") or "kamihime")

    if object_type == "kamihime":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for section in item.get("skill_sections") or []:
            if not isinstance(section, dict):
                continue
            section_type = str(section.get("type") or "skill")
            for row in section.get("rows") or []:
                if isinstance(row, dict):
                    grouped.setdefault(section_type, []).append(row)
        for section_type, rows in grouped.items():
            _append_section(docs, item, section_type, rows)
    elif object_type == "weapon":
        _append_section(docs, item, "Stats", item.get("stats") or [])
        _append_section(docs, item, "Burst Effects", item.get("bursts") or [])
        _append_section(
            docs,
            item,
            "Weapon Skills",
            item.get("weapon_skills") or [],
        )
    elif object_type == "eidolon":
        _append_section(docs, item, "Stats", item.get("stats") or [])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in item.get("eidolon_effects") or []:
            if not isinstance(row, dict):
                continue
            section = str(row.get("type") or "eidolon_effect")
            grouped.setdefault(section, []).append(row)
        for section, rows in grouped.items():
            _append_section(docs, item, section, rows)
    return docs


def catalog_documents(
    object_types: Iterable[str] = OBJECT_TYPES,
    loader: CatalogLoader = load_catalog_items,
) -> list[Document]:
    documents: list[Document] = []
    for object_type in object_types:
        for item in loader(object_type):
            documents.extend(object_documents(item))
    return documents


def documents_fingerprint(documents: list[Document]) -> str:
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(str(doc.metadata.get("document_id", "")).encode("utf-8"))
        digest.update(doc.page_content.encode("utf-8"))
    return digest.hexdigest()[:16]
