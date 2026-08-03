from datetime import date

import pytest

from kami.agent.catalog_query import (
    parse_release_date,
    select_latest_catalog_items,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("26/07/30", date(2026, 7, 30)),
        ("2026/07/30", date(2026, 7, 30)),
        ("26-07-30", date(2026, 7, 30)),
        ("2026-07-30", date(2026, 7, 30)),
        ("-", None),
        ("2026/13/40", None),
    ],
)
def test_parse_release_date(value, expected):
    assert parse_release_date(value) == expected


def test_select_latest_filters_type_and_element_and_keeps_all_ties():
    records = {
        "kamihime": [
            {"name": "Older Water", "slug": "older", "object_type": "kamihime", "element": "water", "release_date": "26/06/01"},
            {"name": "Latest B", "slug": "latest-b", "object_type": "kamihime", "element": "water", "release_date": "26/07/30"},
            {"name": "Latest A", "slug": "latest-a", "object_type": "kamihime", "element": "water", "release_date": "2026/07/30"},
            {"name": "Newer Fire", "slug": "fire", "object_type": "kamihime", "element": "fire", "release_date": "26/08/01"},
        ]
    }
    selection = select_latest_catalog_items(
        ["kamihime"],
        ["water"],
        lambda object_type: records.get(object_type, []),
    )

    assert [item["slug"] for item in selection.items] == ["latest-a", "latest-b"]
    assert selection.matching_count == 3
    assert selection.valid_date_count == 3
    assert selection.latest_dates == {"kamihime": date(2026, 7, 30)}


def test_select_latest_reports_matching_records_without_valid_dates():
    item = {"name": "Unknown", "slug": "unknown", "object_type": "eidolon", "element": "water", "release_date": "-"}
    selection = select_latest_catalog_items(
        ["eidolon"],
        ["water"],
        lambda _object_type: [item],
    )

    assert selection.items == []
    assert selection.matching_count == 1
    assert selection.valid_date_count == 0


def test_select_latest_computes_a_maximum_per_explicit_object_type():
    records = {
        "kamihime": [{"name": "K", "slug": "k", "object_type": "kamihime", "element": "water", "release_date": "26/07/30"}],
        "weapon": [{"name": "W", "slug": "w", "object_type": "weapon", "element": "water", "release_date": "26/06/01"}],
    }
    selection = select_latest_catalog_items(
        ["kamihime", "weapon"],
        ["water"],
        lambda object_type: records.get(object_type, []),
    )

    assert {item["slug"] for item in selection.items} == {"k", "w"}


def test_select_latest_tracks_match_and_valid_date_counts_per_type():
    records = {
        "kamihime": [
            {
                "name": "K",
                "slug": "k",
                "object_type": "kamihime",
                "element": "water",
                "release_date": "26/07/30",
            }
        ],
        "eidolon": [
            {
                "name": "E",
                "slug": "e",
                "object_type": "eidolon",
                "element": "water",
                "release_date": "-",
            }
        ],
        "weapon": [],
    }

    selection = select_latest_catalog_items(
        ["kamihime", "eidolon", "weapon"],
        ["water"],
        lambda object_type: records[object_type],
    )

    assert selection.matching_counts == {
        "kamihime": 1,
        "eidolon": 1,
        "weapon": 0,
    }
    assert selection.valid_date_counts == {
        "kamihime": 1,
        "eidolon": 0,
        "weapon": 0,
    }
