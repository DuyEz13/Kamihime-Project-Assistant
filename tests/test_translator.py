import json
from types import SimpleNamespace

from kami.translator import (
    DeepLJapaneseEnglishTranslator,
    GoogleJapaneseEnglishTranslator,
    KEY_TRANSLATIONS,
    LocalJapaneseEnglishTranslator,
    SUPPORTED_TORCH_VERSION,
    SUPPORTED_TRANSFORMERS_VERSION,
    _extract_json_array,
    create_translator,
    translate_elements,
    translate_object_elements,
)
from kami.paths import (
    element_raw_path,
    element_translation_path,
    object_raw_path,
    object_translation_path,
)
from kami.data_store import _display_info


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
            "unlock_weapon_url": "https://example.test/武器/SSR/闇剣",
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
    assert (
        translated["info"]["unlock_weapon_url"]
        == record["info"]["unlock_weapon_url"]
    )
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


def test_translate_elements_uses_raw_and_translated_subdirectories(tmp_path):
    source = element_raw_path(tmp_path, "fire")
    destination = element_translation_path(tmp_path, "fire", "qwen")
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {"info": {"name": "Test", "element": "fire"}},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert translate_elements(tmp_path, ["fire"], provider="qwen") == {"fire": 1}
    assert destination.exists()
    assert json.loads(destination.read_text(encoding="utf-8"))["info"]["name"] == "Test"


def test_translation_paths_are_split_by_provider(tmp_path):
    assert element_translation_path(
        tmp_path,
        "fire",
        "deepl",
    ) == tmp_path / "kamihime" / "fire" / "translated" / "deepl.jsonl"
    assert object_translation_path(
        tmp_path,
        "weapon",
        "fire",
        "google",
    ) == tmp_path / "weapon" / "fire" / "translated" / "google.jsonl"


def test_translate_object_elements_keeps_object_data_separate(tmp_path):
    source = object_raw_path(tmp_path, "eidolon", "phantom")
    destination = object_translation_path(
        tmp_path,
        "eidolon",
        "phantom",
        "qwen",
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "info": {
                    "name": "Test Eidolon",
                    "object_type": "eidolon",
                    "element": "phantom",
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert translate_object_elements(
        tmp_path,
        "eidolon",
        ["phantom"],
        provider="qwen",
    ) == {"phantom": 1}
    assert destination.exists()


def test_display_info_hides_internal_element_metadata():
    info = {
        "Element": "Water",
        "element": "water",
        "release_date": "25/05/15",
        "acquisition_method": "Internal source list value",
        "Rarity": "SSR",
    }

    assert _display_info(info) == {
        "Element": "Water",
        "Rarity": "SSR",
    }


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


def test_create_translator_selects_deepl_provider(tmp_path):
    translator = create_translator(tmp_path, provider="deepl")

    assert isinstance(translator, DeepLJapaneseEnglishTranslator)
    assert translator.device_label == "DeepL API"
    assert translator.model_name.startswith("deepl-api:")


def test_create_translator_selects_google_provider(tmp_path):
    translator = create_translator(tmp_path, provider="google")

    assert isinstance(translator, GoogleJapaneseEnglishTranslator)
    assert translator.device_label == "Google Translate API"
    assert translator.model_name == "google-translate-api:v2"


def test_deepl_and_qwen_use_separate_cache_namespaces(tmp_path):
    qwen = create_translator(tmp_path, provider="qwen", model_name="test-model")
    deepl = create_translator(tmp_path, provider="deepl")
    google = create_translator(tmp_path, provider="google")

    assert qwen.translation_cache_key("ニケ") != deepl.translation_cache_key(
        "ニケ"
    )
    assert google.translation_cache_key("sample") != deepl.translation_cache_key(
        "sample"
    )


def test_deepl_requires_auth_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    translator = DeepLJapaneseEnglishTranslator(tmp_path)

    try:
        translator._load_client()
    except RuntimeError as exc:
        assert "DEEPL_AUTH_KEY" in str(exc)
    else:
        raise AssertionError("DeepL client should require an API key")


def _deepl_glossary(
    name="KamiWiki JA-EN",
    glossary_id="glossary-id",
    entries=True,
    creation_time="2026-01-01",
):
    dictionaries = (
        [SimpleNamespace(source_lang="JA", target_lang="EN")]
        if entries
        else []
    )
    return SimpleNamespace(
        name=name,
        glossary_id=glossary_id,
        dictionaries=dictionaries,
        creation_time=creation_time,
    )


def test_deepl_reuses_unchanged_managed_glossary(tmp_path, monkeypatch):
    glossary = _deepl_glossary()

    class Client:
        def list_multilingual_glossaries(self):
            return [glossary]

        def get_multilingual_glossary_entries(self, selected, source, target):
            assert selected is glossary
            assert (source, target) == ("JA", "EN")
            return SimpleNamespace(
                dictionaries=[
                    SimpleNamespace(
                        source_lang="JA",
                        target_lang="EN",
                        entries="\n".join(
                            f"{key}\t{value}"
                            for key, value in translator.glossary.items()
                        ),
                    )
                ]
            )

        def replace_multilingual_glossary_dictionary(self, *_args):
            raise AssertionError("Unchanged glossary must not be replaced")

    monkeypatch.delenv("DEEPL_GLOSSARY_ID", raising=False)
    translator = DeepLJapaneseEnglishTranslator(tmp_path)
    translator._client = Client()

    assert translator._ensure_glossary() is glossary


def test_deepl_migrates_legacy_name_and_replaces_changed_entries(
    tmp_path,
    monkeypatch,
):
    legacy = _deepl_glossary(name="KamiWiki JA-EN old-hash")
    renamed = _deepl_glossary()
    calls = []

    class Client:
        def list_multilingual_glossaries(self):
            return [legacy]

        def update_multilingual_glossary_name(self, selected, name):
            calls.append(("rename", selected, name))
            return renamed

        def get_multilingual_glossary_entries(self, *_args):
            return SimpleNamespace(
                dictionaries=[
                    SimpleNamespace(
                        source_lang="JA",
                        target_lang="EN",
                        entries={"old": "term"},
                    )
                ]
            )

        def replace_multilingual_glossary_dictionary(
            self,
            selected,
            dictionary,
        ):
            calls.append(("replace", selected, dictionary.entries))

    monkeypatch.delenv("DEEPL_GLOSSARY_ID", raising=False)
    translator = DeepLJapaneseEnglishTranslator(tmp_path)
    translator._client = Client()

    assert translator._ensure_glossary() is renamed
    assert calls == [
        ("rename", legacy, "KamiWiki JA-EN"),
        ("replace", renamed, translator.glossary),
    ]


def test_deepl_creates_stable_managed_glossary(tmp_path, monkeypatch):
    created = _deepl_glossary()
    calls = []

    class Client:
        def list_multilingual_glossaries(self):
            return []

        def create_multilingual_glossary(self, name, dictionaries):
            calls.append((name, dictionaries[0].entries))
            return created

        def get_multilingual_glossary_entries(self, *_args):
            return SimpleNamespace(
                dictionaries=[
                    SimpleNamespace(
                        source_lang="JA",
                        target_lang="EN",
                        entries=translator.glossary,
                    )
                ]
            )

    monkeypatch.delenv("DEEPL_GLOSSARY_ID", raising=False)
    translator = DeepLJapaneseEnglishTranslator(tmp_path)
    translator._client = Client()

    assert translator._ensure_glossary() is created
    assert calls == [("KamiWiki JA-EN", translator.glossary)]


def test_deepl_syncs_configured_glossary_id(tmp_path, monkeypatch):
    configured = _deepl_glossary(name="Shared glossary")
    calls = []

    class Client:
        def get_multilingual_glossary(self, glossary_id):
            calls.append(("get", glossary_id))
            return configured

        def get_multilingual_glossary_entries(self, *_args):
            return SimpleNamespace(
                dictionaries=[
                    SimpleNamespace(
                        source_lang="JA",
                        target_lang="EN",
                        entries={"old": "term"},
                    )
                ]
            )

        def replace_multilingual_glossary_dictionary(
            self,
            selected,
            dictionary,
        ):
            calls.append(("replace", selected, dictionary.entries))

    monkeypatch.setenv("DEEPL_GLOSSARY_ID", "configured-id")
    translator = DeepLJapaneseEnglishTranslator(tmp_path)
    translator._client = Client()

    assert translator._ensure_glossary() is configured
    assert calls == [
        ("get", "configured-id"),
        ("replace", configured, translator.glossary),
    ]


def test_deepl_translates_batch_and_caches_results(tmp_path):
    class Result:
        def __init__(self, text):
            self.text = text

    class Client:
        def translate_text(self, texts, **options):
            assert options["source_lang"] == "JA"
            assert options["target_lang"] == "EN-US"
            assert options["glossary"] == "glossary-id"
            return [Result("Nike"), Result("Raging state")]

    translator = DeepLJapaneseEnglishTranslator(tmp_path)
    translator._client = Client()
    translator._deepl_glossary = "glossary-id"

    translated = translator.translate_texts(["ニケ", "レイジング状態"])

    assert translated == ["Nike", "Raging state"]


def test_google_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_TRANSLATE_API_KEY", raising=False)
    translator = GoogleJapaneseEnglishTranslator(tmp_path)

    try:
        translator._api_key()
    except RuntimeError as exc:
        assert "GOOGLE_TRANSLATE_API_KEY" in str(exc)
    else:
        raise AssertionError("Google translator should require an API key")


def test_google_translates_batch_and_caches_results(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "translations": [
                        {"translatedText": "Nike"},
                        {"translatedText": "Raging state"},
                    ]
                }
            }

    class Client:
        def post(self, url, *, params, json):
            assert params["key"] == "google-key"
            assert json["source"] == "ja"
            assert json["target"] == "en"
            assert json["q"] == [
                "\u30cb\u30b1",
                "\u30ec\u30a4\u30b8\u30f3\u30b0\u72b6\u614b",
            ]
            return Response()

    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "google-key")
    translator = GoogleJapaneseEnglishTranslator(tmp_path)
    translator._client = Client()

    nike = "\u30cb\u30b1"
    raging_state = "\u30ec\u30a4\u30b8\u30f3\u30b0\u72b6\u614b"
    translated = translator.translate_texts([nike, raging_state])

    assert translated == ["Nike", "Raging state"]
    assert translator._cached_translation(nike) == "Nike"
    assert translator._cached_translation(raging_state) == "Raging state"


def google_translates_batch_and_caches_results_legacy_mojibake(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "translations": [
                        {"translatedText": "Nike"},
                        {"translatedText": "Raging state"},
                    ]
                }
            }

    class Client:
        def post(self, url, *, params, json):
            assert params["key"] == "google-key"
            assert json["source"] == "ja"
            assert json["target"] == "en"
            assert len(json["q"]) == 2
            return Response()

    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "google-key")
    translator = GoogleJapaneseEnglishTranslator(tmp_path)
    translator._client = Client()

    translated = translator.translate_texts(["ãƒ‹ã‚±", "ãƒ¬ã‚¤ã‚¸ãƒ³ã‚°çŠ¶æ…‹"])

    assert translated == ["Nike", "Raging state"]
    assert translator._cached_translation("ニケ") == "Nike"
    assert translator._cached_translation("レイジング状態") == "Raging state"
