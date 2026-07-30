import json

from kami import data_store
from kami.paths import object_raw_path, object_translation_path


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_load_object_records_prefers_available_translation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(data_store, "DATA_DIR", tmp_path)
    raw_path = object_raw_path(tmp_path, "weapon", "phantom")
    translated_path = object_translation_path(
        tmp_path,
        "weapon",
        "phantom",
        "deepl",
    )
    _write_jsonl(
        raw_path,
        [
            {
                "info": {
                    "name": "幻の武器",
                    "object_type": "weapon",
                    "element": "phantom",
                }
            }
        ],
    )
    _write_jsonl(
        translated_path,
        [
            {
                "info": {
                    "name": "Phantom Weapon",
                    "object_type": "weapon",
                    "element": "phantom",
                }
            }
        ],
    )

    records = data_store.load_object_records("weapons", "phantom")

    assert records[0]["info"]["name"] == "Phantom Weapon"


def test_load_object_records_keeps_object_types_separate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(data_store, "DATA_DIR", tmp_path)
    _write_jsonl(
        object_raw_path(tmp_path, "eidolon", "fire"),
        [{"info": {"name": "Eidolon", "element": "fire"}}],
    )
    _write_jsonl(
        object_raw_path(tmp_path, "weapon", "fire"),
        [{"info": {"name": "Weapon", "element": "fire"}}],
    )

    assert data_store.load_object_records("eidolon") == [
        {"info": {"name": "Eidolon", "element": "fire"}}
    ]


def test_kamihime_loader_supports_legacy_layout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(data_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        data_store,
        "LEGACY_RAW_DATA_DIR",
        tmp_path / "raw",
    )
    legacy_raw = tmp_path / "raw" / "kamihime_fire_raw.jsonl"
    legacy_translation = (
        tmp_path
        / "translated"
        / "deepl"
        / "kamihime_fire_en.jsonl"
    )
    _write_jsonl(
        legacy_raw,
        [{"info": {"name": "火の神姫", "element": "fire"}}],
    )
    _write_jsonl(
        legacy_translation,
        [{"info": {"name": "Fire Kamihime", "element": "fire"}}],
    )

    assert data_store.load_object_records("kamihime", "fire") == [
        {"info": {"name": "Fire Kamihime", "element": "fire"}}
    ]


def test_catalog_view_models_are_partitioned_by_object_type(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(data_store, "DATA_DIR", tmp_path)
    for object_type in ("eidolon", "weapon"):
        _write_jsonl(
            object_raw_path(tmp_path, object_type, "phantom"),
            [
                {
                    "info": {
                        "name": "Shared Name",
                        "object_type": object_type,
                        "element": "phantom",
                        "release_date": "26/07/30",
                        "acquisition_method": "Test",
                    },
                    "stats": [{"limit_break": "4", "HP": "1000"}],
                }
            ],
        )

    eidolon = data_store.load_catalog_items("eidolon", "phantom")[0]
    weapon = data_store.load_catalog_items("weapon", "phantom")[0]

    assert eidolon["slug"] == weapon["slug"] == "shared-name"
    assert eidolon["object_type"] == "eidolon"
    assert weapon["object_type"] == "weapon"
    assert data_store.get_catalog_item("eidolon", "shared-name") == eidolon


def test_eidolon_effect_rows_merge_repeated_type_and_name():
    rows = data_store._prepare_eidolon_effects(
        [
            {"type": "Summon Effect", "name": "Phantom Call"},
            {"type": "Main Effect", "name": "Phantom Power", "requirements": "-"},
            {"type": "Main Effect", "name": "Phantom Power", "requirements": "1"},
            {"type": "Main Effect", "name": "Phantom Power", "requirements": "2"},
            {"type": "Sub Effect", "name": "Phantom Ward", "requirements": "-"},
            {"type": "Sub Effect", "name": "Phantom Ward", "requirements": "1"},
        ]
    )

    assert rows[0]["show_type"] is True
    assert rows[0]["type_rowspan"] == 1
    assert rows[1]["show_type"] is True
    assert rows[1]["type_rowspan"] == 3
    assert rows[2]["show_type"] is False
    assert rows[1]["show_name"] is True
    assert rows[1]["name_rowspan"] == 3
    assert rows[3]["show_name"] is False
    assert rows[4]["show_type"] is True
    assert rows[4]["type_rowspan"] == 2
    assert rows[4]["show_name"] is True
    assert rows[4]["name_rowspan"] == 2


def test_eidolon_catalog_splits_summon_and_passive_effect_tables(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(data_store, "DATA_DIR", tmp_path)
    _write_jsonl(
        object_raw_path(tmp_path, "eidolon", "fire"),
        [
            {
                "info": {"name": "Test Eidolon", "element": "fire"},
                "eidolon_effects": [
                    {"type": "Summon Effect", "name": "Flame Call"},
                    {"type": "Main Effect", "name": "Flame Power"},
                    {
                        "type": "Main Effect",
                        "name": "Flame Power",
                        "requirements": "1",
                    },
                ],
            }
        ],
    )

    item = data_store.load_catalog_items("eidolon", "fire")[0]

    assert [row["name"] for row in item["eidolon_summon_effects"]] == [
        "Flame Call"
    ]
    assert len(item["eidolon_passive_effects"]) == 2
    assert item["eidolon_passive_effects"][0]["type_rowspan"] == 2
    assert item["eidolon_passive_effects"][0]["name_rowspan"] == 2
