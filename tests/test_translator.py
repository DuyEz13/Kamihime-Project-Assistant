import json

from kami.translator import (
    KEY_TRANSLATIONS,
    LocalJapaneseEnglishTranslator,
    SUPPORTED_TORCH_VERSION,
    SUPPORTED_TRANSFORMERS_VERSION,
    _extract_json_array,
)


def test_translator_preserves_structural_values_and_translates_schema(tmp_path):
    translator = LocalJapaneseEnglishTranslator(tmp_path, model_name="test-model")
    translations = {
        "闇魔": "Dark Fiend",
        "闇属性ダメージ": "Dark-element damage",
    }
    for source, translated in translations.items():
        translator.cache[translator.translation_cache_key(source)] = {
            "source": source,
            "translation": translated,
        }

    record = {
        "info": {
            "name": "闇魔",
            "source_url": "https://example.test/闇魔",
            "img": "https://example.test/image.jpg",
            "element": "dark",
            "実装日": "16/10/07",
        },
        "skill": [
            {
                "バースト": "闇魔",
                "習得条件": "-",
                "効果": "闇属性ダメージ",
            }
        ],
    }

    translated = translator.translate_records([record])[0]

    assert translated["info"]["name"] == "Dark Fiend"
    assert translated["info"]["source_url"] == record["info"]["source_url"]
    assert translated["info"]["img"] == record["info"]["img"]
    assert translated["info"]["element"] == "dark"
    assert translated["info"]["Release Date"] == "16/10/07"
    assert translated["skill"][0]["Burst"] == "Dark Fiend"
    assert translated["skill"][0]["Effect"] == "Dark-element damage"
    assert KEY_TRANSLATIONS["習得条件"] in translated["skill"][0]


def test_translate_file_writes_valid_jsonl_atomically(tmp_path):
    source = tmp_path / "kamihime_fire_raw.jsonl"
    destination = tmp_path / "kamihime_fire_en.jsonl"
    source.write_text(
        json.dumps(
            {"info": {"name": "Test", "element": "fire"}},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    translator = LocalJapaneseEnglishTranslator(tmp_path, model_name="test-model")

    assert translator.translate_file(source, destination) == 1
    assert json.loads(destination.read_text(encoding="utf-8"))["info"]["name"] == "Test"


def test_cached_translation_reports_progress_and_device(tmp_path):
    translator = LocalJapaneseEnglishTranslator(tmp_path, model_name="test-model")
    source = "炎の女神"
    translator.cache[translator.translation_cache_key(source)] = {
        "source": source,
        "translation": "Goddess of Flame",
    }
    events = []

    translated = translator.translate_records(
        [{"info": {"name": source, "element": "fire"}}],
        events.append,
    )

    assert translated[0]["info"]["name"] == "Goddess of Flame"
    assert events[-1]["progress"] == 100
    assert events[-1]["processed"] == events[-1]["total"] == 1
    assert events[-1]["device"] in {"CPU"} or events[-1]["device"].startswith("GPU:")


def test_prompt_contains_glossary_and_translation_memory(tmp_path):
    glossary = tmp_path / "glossary.json"
    glossary.write_text(
        json.dumps({"攻撃UP": "increases Attack"}, ensure_ascii=False),
        encoding="utf-8",
    )
    translator = LocalJapaneseEnglishTranslator(
        tmp_path,
        model_name="test-model",
        glossary_path=glossary,
    )
    memory_source = "味方全体の攻撃UP"
    translator.cache[translator.translation_cache_key(memory_source)] = {
        "source": memory_source,
        "translation": "Increases all allies' Attack",
    }

    messages = translator._build_messages(["自分の攻撃UP"])

    assert '"攻撃UP": "increases Attack"' in messages[1]["content"]
    assert memory_source in messages[1]["content"]
    assert "Increases all allies' Attack" in messages[1]["content"]


def test_translate_texts_uses_cached_values(tmp_path):
    translator = LocalJapaneseEnglishTranslator(tmp_path, model_name="test-model")
    source = "味方全体の攻撃UP"
    translator.cache[translator.translation_cache_key(source)] = {
        "source": source,
        "translation": "Increases all allies' attack",
    }

    assert translator.translate_texts([source]) == [
        "Increases all allies' attack"
    ]


def test_extract_json_array_accepts_fenced_model_output():
    output = '```json\n[{"id": 0, "translation": "Massive Fire damage"}]\n```'

    assert _extract_json_array(output) == [
        {"id": 0, "translation": "Massive Fire damage"}
    ]


def test_glossary_change_invalidates_translation_cache(tmp_path):
    first_glossary = tmp_path / "first.json"
    second_glossary = tmp_path / "second.json"
    first_glossary.write_text('{"攻撃UP": "increases Attack"}', encoding="utf-8")
    second_glossary.write_text('{"攻撃UP": "Attack Up"}', encoding="utf-8")

    first = LocalJapaneseEnglishTranslator(
        tmp_path,
        model_name="test-model",
        glossary_path=first_glossary,
    )
    second = LocalJapaneseEnglishTranslator(
        tmp_path,
        model_name="test-model",
        glossary_path=second_glossary,
    )

    assert first.translation_cache_key("攻撃UP") != second.translation_cache_key(
        "攻撃UP"
    )


def test_autoawq_compatibility_versions_are_pinned():
    assert SUPPORTED_TORCH_VERSION == "2.6.0"
    assert SUPPORTED_TRANSFORMERS_VERSION == "4.51.3"
