from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from .documents import CatalogLoader


@dataclass(frozen=True)
class CatalogSelection:
    items: list[dict[str, Any]]
    matching_count: int
    valid_date_count: int
    latest_dates: dict[str, date]


def parse_release_date(value: object) -> date | None:
    match = re.fullmatch(
        r"\s*(\d{2}|\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*",
        str(value or ""),
    )
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def select_latest_catalog_items(
    object_types: Sequence[str],
    elements: Sequence[str],
    loader: CatalogLoader,
) -> CatalogSelection:
    selected_elements = {value.casefold() for value in elements if value}
    matching_count = 0
    valid_date_count = 0
    latest_dates: dict[str, date] = {}
    winners: dict[str, list[dict[str, Any]]] = {}

    for object_type in dict.fromkeys(object_types):
        for item in loader(object_type):
            element = str(item.get("element") or "").casefold()
            if selected_elements and element not in selected_elements:
                continue
            matching_count += 1
            released = parse_release_date(item.get("release_date"))
            if released is None:
                continue
            valid_date_count += 1
            current = latest_dates.get(object_type)
            if current is None or released > current:
                latest_dates[object_type] = released
                winners[object_type] = [item]
            elif released == current:
                winners.setdefault(object_type, []).append(item)

    items = [
        item
        for object_type in dict.fromkeys(object_types)
        for item in sorted(
            winners.get(object_type, []),
            key=lambda value: (
                str(value.get("name") or "").casefold(),
                str(value.get("slug") or ""),
            ),
        )
    ]
    return CatalogSelection(
        items=items,
        matching_count=matching_count,
        valid_date_count=valid_date_count,
        latest_dates=latest_dates,
    )
