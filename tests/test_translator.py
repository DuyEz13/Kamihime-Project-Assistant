import json

from kami.translator import (
    KEY_TRANSLATIONS,
    LocalJapaneseEnglishTranslator,
    _cache_key,
)


def test_translator_preserves_structural_values_and_translates_schema(tmp_path):
    translator = LocalJapaneseEnglishTranslator(tmp_path, model_name="test-model")
    translations = {
        "闇魔": "Dark Fiend",
        "闇属性ダメージ": "Dark-element damage",
    }
    for source, translated in translations.items():
        translator.cache[_cache_key(translator.model_name, source)] = {
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
    translator.cache[_cache_key(translator.model_name, source)] = {
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


def test_madlad_input_includes_english_target_prefix(tmp_path):
    translator = LocalJapaneseEnglishTranslator(
        tmp_path,
        model_name="google/madlad400-3b-mt",
    )

    assert translator._model_input("こんにちは") == "<2en> こんにちは"


def test_translate_texts_uses_cached_values(tmp_path):
    translator = LocalJapaneseEnglishTranslator(tmp_path, model_name="test-model")
    source = "味方全体の攻撃UP"
    translator.cache[_cache_key(translator.model_name, source)] = {
        "source": source,
        "translation": "Increases all allies' attack",
    }

    assert translator.translate_texts([source]) == [
        "Increases all allies' attack"
    ]
