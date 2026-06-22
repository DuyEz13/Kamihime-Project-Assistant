import difflib
import html
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from .paths import (
    TRANSLATION_PROVIDERS,
    element_raw_path,
    element_translation_path,
    normalize_translation_provider,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_PROVIDER = "qwen"
SUPPORTED_TORCH_VERSION = "2.6.0"
SUPPORTED_TRANSFORMERS_VERSION = "4.51.3"
CACHE_FILE = ".translation_cache.json"
DEFAULT_GLOSSARY_FILE = Path(__file__).with_name("translation_glossary.json")
PROMPT_VERSION = "qwen-kamihime-v1"
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
SENTENCE_END_RE = re.compile(r"(?<=[。！？\n])")
ProgressCallback = Callable[[dict[str, Any]], None]

KEY_TRANSLATIONS = {
    "レアリティ": "Rarity",
    "最大レベル": "Max Level",
    "属性": "Element",
    "得意武器": "Preferred Weapon",
    "タイプ": "Type",
    "攻撃力": "Attack",
    "実装日": "Release Date",
    "入手方法": "Acquisition Method",
    "解放武器": "Unlock Weapon",
    "真化ボーナス": "Finalization Bonus",
    "進化": "Evolution",
    "バースト": "Burst",
    "アビリティ": "Ability",
    "アシスト": "Assist",
    "習得条件": "Acquisition Requirements",
    "使用間隔": "Usage Interval",
    "効果時間": "Effect Duration",
    "効果": "Effect",
}

PRESERVED_VALUE_KEYS = {
    "source_url",
    "img",
    "image",
    "icon",
    "list_image",
    "element",
    "release_date",
}

SYSTEM_PROMPT = """You are the English localization engine for the game Kamihime Project.
Translate Japanese game data into concise, natural English.

Consistency is more important than stylistic variety:
- Always use the supplied glossary exactly for recurring game concepts.
- Follow translation-memory examples when the same wording or concept appears.
- Use the same English phrase for the same Japanese effect across every record.
- Preserve all numbers, percentages, operators, turn notation (for example 3T),
  level notation, slashes, stars, parentheses, and game abbreviations.
- Do not explain, summarize, censor, or add information.
- Transliterate proper names consistently when no established English name is known.
- Return only a valid JSON array. Each item must contain the original integer "id"
  and one string field named "translation".
"""


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _atomic_write_jsonl(records: Iterable[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path} line {line_number} is not a JSON object"
                )
            records.append(value)
    return records


def _cache_key(namespace: str, text: str) -> str:
    value = f"{namespace}\0{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in SENTENCE_END_RE.split(text):
        if not sentence:
            continue
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    return chunks


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON array")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("Model response JSON was not an array")
    return [item for item in value if isinstance(item, dict)]


class LocalJapaneseEnglishTranslator:
    def __init__(
        self,
        data_dir: Path,
        model_name: str | None = None,
        glossary_path: Path | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.model_name = (
            model_name
            or os.getenv("KAMI_TRANSLATION_MODEL")
            or DEFAULT_MODEL
        )
        self.batch_size = max(
            1, int(os.getenv("KAMI_TRANSLATION_BATCH_SIZE", "8"))
        )
        self.max_chars = max(
            80, int(os.getenv("KAMI_TRANSLATION_MAX_CHARS", "500"))
        )
        self.max_new_tokens = max(
            128, int(os.getenv("KAMI_TRANSLATION_MAX_NEW_TOKENS", "2048"))
        )
        self.memory_examples = max(
            0, int(os.getenv("KAMI_TRANSLATION_MEMORY_EXAMPLES", "6"))
        )
        self.memory_scan = max(
            self.memory_examples,
            int(os.getenv("KAMI_TRANSLATION_MEMORY_SCAN", "500")),
        )
        configured_glossary = os.getenv("KAMI_TRANSLATION_GLOSSARY")
        self.glossary_path = glossary_path or (
            Path(configured_glossary)
            if configured_glossary
            else DEFAULT_GLOSSARY_FILE
        )
        self.glossary = {
            str(key): str(value)
            for key, value in _load_json_object(self.glossary_path).items()
        }
        glossary_fingerprint = hashlib.sha256(
            json.dumps(
                self.glossary,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        self.cache_namespace = (
            f"{self.model_name}:{PROMPT_VERSION}:{glossary_fingerprint}"
        )
        self.cache_path = data_dir / CACHE_FILE
        self.cache = _load_json_object(self.cache_path)
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        self.device_label = "CPU"

    def translation_cache_key(self, text: str) -> str:
        return _cache_key(self.cache_namespace, text)

    def _resolve_device(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Local translation dependencies are missing. "
                "Run `uv sync --extra translation`."
            ) from exc

        requested_device = os.getenv("KAMI_TRANSLATION_DEVICE", "auto").lower()
        cuda_available = torch.cuda.is_available()
        if requested_device == "auto":
            self._device = "cuda" if cuda_available else "cpu"
        elif requested_device == "cuda" and not cuda_available:
            raise RuntimeError(
                "KAMI_TRANSLATION_DEVICE=cuda, but CUDA is unavailable"
            )
        elif requested_device not in {"cpu", "cuda"}:
            raise ValueError(
                "KAMI_TRANSLATION_DEVICE must be auto, cpu, or cuda"
            )
        else:
            self._device = requested_device

        if "awq" in self.model_name.casefold() and self._device != "cuda":
            raise RuntimeError(
                "The AWQ translation model requires an NVIDIA CUDA GPU. "
                "Run it on Colab with a GPU runtime."
            )

        self._torch = torch
        if self._device == "cuda":
            self.device_label = f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            self.device_label = "CPU"

    def _load_model(
        self,
        progress_callback: ProgressCallback | None = None,
        total: int = 0,
        processed: int = 0,
    ) -> None:
        if self._model is not None:
            return
        self._resolve_device()
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen translation dependencies are missing. "
                "Run `uv sync --extra translation`."
            ) from exc

        torch_version = self._torch.__version__.split("+", 1)[0]
        if (
            torch_version != SUPPORTED_TORCH_VERSION
            or transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION
        ):
            raise RuntimeError(
                "AutoAWQ requires the pinned compatibility stack: "
                f"torch=={SUPPORTED_TORCH_VERSION} and "
                f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}. "
                f"Found torch=={self._torch.__version__} and "
                f"transformers=={transformers.__version__}. "
                "Recreate .venv and run `uv sync --extra translation`."
            )

        if progress_callback:
            progress_callback(
                {
                    "phase": "loading",
                    "progress": round(processed * 100 / total) if total else 0,
                    "processed": processed,
                    "total": total,
                    "device": self.device_label,
                    "model": self.model_name,
                }
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._torch.float16,
            device_map="auto" if self._device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self._device == "cpu":
            self._model.to("cpu")
        self._model.eval()

    def _cached_translation(self, text: str) -> str | None:
        entry = self.cache.get(self.translation_cache_key(text))
        if not isinstance(entry, dict) or entry.get("source") != text:
            return None
        translation = entry.get("translation")
        return translation if isinstance(translation, str) else None

    def _matching_glossary(self, texts: list[str]) -> dict[str, str]:
        combined = "\n".join(texts)
        return {
            source: target
            for source, target in self.glossary.items()
            if source in combined
        }

    def _translation_memory(self, texts: list[str]) -> list[dict[str, str]]:
        if self.memory_examples == 0:
            return []

        candidates: list[tuple[float, str, str]] = []
        entries = list(self.cache.values())[-self.memory_scan :]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source = entry.get("source")
            translation = entry.get("translation")
            if not isinstance(source, str) or not isinstance(translation, str):
                continue
            score = max(
                difflib.SequenceMatcher(None, source, text).ratio()
                for text in texts
            )
            if score >= 0.45:
                candidates.append((score, source, translation))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            {"Japanese": source, "English": translation}
            for _score, source, translation in candidates[: self.memory_examples]
        ]

    def _build_messages(self, texts: list[str]) -> list[dict[str, str]]:
        payload = [
            {"id": index, "Japanese": text}
            for index, text in enumerate(texts)
        ]
        user_prompt = (
            "Canonical glossary:\n"
            + json.dumps(
                self._matching_glossary(texts),
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nRelevant translation-memory examples:\n"
            + json.dumps(
                self._translation_memory(texts),
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nTranslate these entries:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _generate_batch(self, texts: list[str]) -> list[str]:
        messages = self._build_messages(texts)
        model_inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        output = self._tokenizer.decode(
            generated[0][model_inputs.shape[-1] :],
            skip_special_tokens=True,
        )
        parsed = _extract_json_array(output)
        by_id = {
            item.get("id"): item.get("translation")
            for item in parsed
            if isinstance(item.get("id"), int)
            and isinstance(item.get("translation"), str)
        }
        if len(by_id) != len(texts):
            raise ValueError(
                f"Model returned {len(by_id)} of {len(texts)} translations"
            )
        return [by_id[index].strip() for index in range(len(texts))]

    def _translate_batch(self, texts: list[str]) -> list[str]:
        try:
            return self._generate_batch(texts)
        except (json.JSONDecodeError, ValueError):
            if len(texts) == 1:
                raise
            return [
                self._generate_batch([text])[0]
                for text in texts
            ]

    def _translate_missing(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        unique_texts = list(dict.fromkeys(texts))
        missing = [
            text for text in unique_texts if not self._cached_translation(text)
        ]
        total = len(unique_texts)
        processed = total - len(missing)
        if progress_callback:
            progress_callback(
                {
                    "phase": "preparing",
                    "progress": round(processed * 100 / total) if total else 100,
                    "processed": processed,
                    "total": total,
                    "device": self.device_label,
                    "model": self.model_name,
                }
            )
        if not missing:
            return

        self._resolve_device()
        self._load_model(progress_callback, total, processed)
        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset : offset + self.batch_size]
            translations = self._translate_batch(batch)
            for source, translation in zip(batch, translations):
                self.cache[self.translation_cache_key(source)] = {
                    "source": source,
                    "translation": translation,
                    "model": self.model_name,
                    "prompt_version": PROMPT_VERSION,
                }
            _atomic_write_json(self.cache_path, self.cache)
            processed += len(batch)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "translating",
                        "progress": round(processed * 100 / total),
                        "processed": processed,
                        "total": total,
                        "device": self.device_label,
                        "model": self.model_name,
                    }
                )

    def _translatable_chunks(self, text: str) -> list[str]:
        if not JAPANESE_RE.search(text):
            return []
        return [
            chunk
            for chunk in _chunk_text(text, self.max_chars)
            if JAPANESE_RE.search(chunk)
        ]

    def _collect_texts(
        self,
        value: Any,
        parent_key: str | None = None,
    ) -> list[str]:
        if isinstance(value, dict):
            texts: list[str] = []
            for key, child in value.items():
                if key not in KEY_TRANSLATIONS and JAPANESE_RE.search(key):
                    texts.extend(self._translatable_chunks(key))
                if key not in PRESERVED_VALUE_KEYS:
                    texts.extend(self._collect_texts(child, key))
            return texts
        if isinstance(value, list):
            return [
                text
                for child in value
                for text in self._collect_texts(child, parent_key)
            ]
        if isinstance(value, str) and parent_key not in PRESERVED_VALUE_KEYS:
            return self._translatable_chunks(value)
        return []

    def _translate_text(self, text: str) -> str:
        if not JAPANESE_RE.search(text):
            return text
        return "".join(
            self._cached_translation(chunk) or chunk
            if JAPANESE_RE.search(chunk)
            else chunk
            for chunk in _chunk_text(text, self.max_chars)
        )

    def _translate_value(
        self,
        value: Any,
        parent_key: str | None = None,
    ) -> Any:
        if isinstance(value, dict):
            translated: dict[str, Any] = {}
            for key, child in value.items():
                translated_key = KEY_TRANSLATIONS.get(key)
                if translated_key is None:
                    translated_key = self._translate_text(key)
                translated[translated_key] = (
                    child
                    if key in PRESERVED_VALUE_KEYS
                    else self._translate_value(child, key)
                )
            return translated
        if isinstance(value, list):
            return [
                self._translate_value(child, parent_key)
                for child in value
            ]
        if isinstance(value, str) and parent_key not in PRESERVED_VALUE_KEYS:
            return self._translate_text(value)
        return value

    def collect_texts(self, records: list[dict]) -> list[str]:
        return [
            text
            for record in records
            for text in self._collect_texts(record)
        ]

    def translate_records(
        self,
        records: list[dict],
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict]:
        self._translate_missing(
            self.collect_texts(records),
            progress_callback,
        )
        return [self._translate_value(record) for record in records]

    def translate_texts(
        self,
        texts: Iterable[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[str]:
        source_texts = list(texts)
        chunks = [
            chunk
            for text in source_texts
            for chunk in self._translatable_chunks(text)
        ]
        self._translate_missing(chunks, progress_callback)
        return [self._translate_text(text) for text in source_texts]

    def translate_file(self, source: Path, destination: Path) -> int:
        records = _read_jsonl(source)
        translated = self.translate_records(records)
        _atomic_write_jsonl(translated, destination)
        return len(translated)


class DeepLJapaneseEnglishTranslator(LocalJapaneseEnglishTranslator):
    def __init__(
        self,
        data_dir: Path,
        glossary_path: Path | None = None,
    ) -> None:
        self.model_type = os.getenv(
            "DEEPL_MODEL_TYPE",
            "prefer_quality_optimized",
        )
        super().__init__(
            data_dir,
            model_name=f"deepl-api:{self.model_type}",
            glossary_path=glossary_path,
        )
        self.batch_size = max(
            1, int(os.getenv("DEEPL_TRANSLATION_BATCH_SIZE", "50"))
        )
        self.device_label = "DeepL API"
        self._client = None
        self._deepl_glossary = None

    def _resolve_device(self) -> None:
        self._device = "api"
        self.device_label = "DeepL API"

    def _load_client(self) -> None:
        if self._client is not None:
            return
        auth_key = os.getenv("DEEPL_AUTH_KEY")
        if not auth_key:
            raise RuntimeError(
                "Set DEEPL_AUTH_KEY before using the DeepL provider"
            )
        try:
            import deepl
        except ImportError as exc:
            raise RuntimeError(
                "The DeepL SDK is missing. Run `uv sync --extra deepl`."
            ) from exc
        self._client = deepl.DeepLClient(auth_key)

    def _ensure_glossary(self) -> Any | None:
        if self._deepl_glossary is not None:
            return self._deepl_glossary
        configured_id = os.getenv("DEEPL_GLOSSARY_ID")
        if configured_id:
            self._deepl_glossary = configured_id
            return configured_id
        if not self.glossary:
            return None

        self._load_client()
        try:
            from deepl.api_data import (
                MultilingualGlossaryDictionaryEntries,
            )

            glossary_name = (
                "KamiWiki JA-EN "
                + self.cache_namespace.rsplit(":", 1)[-1]
            )
            for glossary in self._client.list_multilingual_glossaries():
                if glossary.name == glossary_name:
                    self._deepl_glossary = glossary
                    return glossary

            dictionary = MultilingualGlossaryDictionaryEntries(
                "JA",
                "EN",
                self.glossary,
            )
            self._deepl_glossary = (
                self._client.create_multilingual_glossary(
                    glossary_name,
                    [dictionary],
                )
            )
            return self._deepl_glossary
        except Exception as exc:
            if os.getenv("DEEPL_REQUIRE_GLOSSARY", "1") != "0":
                raise RuntimeError(
                    "Could not create or access the DeepL JA-EN glossary. "
                    "Set DEEPL_GLOSSARY_ID to an existing glossary or "
                    "DEEPL_REQUIRE_GLOSSARY=0 to continue without one."
                ) from exc
            return None

    def _deepl_context(self, texts: list[str]) -> str:
        matching = self._matching_glossary(texts)
        terms = "; ".join(
            f"{source} means {target}"
            for source, target in matching.items()
        )
        base = (
            "Kamihime Project game localization. Use consistent terminology "
            "for character names, combat states, buffs, debuffs, abilities, "
            "Burst effects, and turn notation."
        )
        return f"{base} Canonical terms: {terms}" if terms else base

    def _translate_missing(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        unique_texts = list(dict.fromkeys(texts))
        missing = [
            text for text in unique_texts if not self._cached_translation(text)
        ]
        total = len(unique_texts)
        processed = total - len(missing)
        self._resolve_device()
        if progress_callback:
            progress_callback(
                {
                    "phase": "preparing",
                    "progress": round(processed * 100 / total) if total else 100,
                    "processed": processed,
                    "total": total,
                    "device": self.device_label,
                    "model": self.model_name,
                }
            )
        if not missing:
            return

        self._load_client()
        glossary = self._ensure_glossary()
        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset : offset + self.batch_size]
            options: dict[str, Any] = {
                "source_lang": "JA",
                "target_lang": "EN-US",
                "preserve_formatting": True,
                "context": self._deepl_context(batch),
                "model_type": self.model_type,
            }
            if glossary is not None:
                options["glossary"] = glossary
            results = self._client.translate_text(batch, **options)
            if not isinstance(results, list):
                results = [results]
            if len(results) != len(batch):
                raise RuntimeError(
                    f"DeepL returned {len(results)} of {len(batch)} translations"
                )
            for source, result in zip(batch, results):
                self.cache[self.translation_cache_key(source)] = {
                    "source": source,
                    "translation": result.text.strip(),
                    "model": self.model_name,
                    "provider": "deepl",
                }
            _atomic_write_json(self.cache_path, self.cache)
            processed += len(batch)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "translating",
                        "progress": round(processed * 100 / total),
                        "processed": processed,
                        "total": total,
                        "device": self.device_label,
                        "model": self.model_name,
                    }
                )

    def usage(self) -> dict[str, int | None]:
        self._load_client()
        usage = self._client.get_usage()
        if not usage.character.valid:
            return {"count": None, "limit": None}
        return {
            "count": usage.character.count,
            "limit": usage.character.limit,
        }


class GoogleJapaneseEnglishTranslator(LocalJapaneseEnglishTranslator):
    def __init__(
        self,
        data_dir: Path,
        glossary_path: Path | None = None,
    ) -> None:
        super().__init__(
            data_dir,
            model_name="google-translate-api:v2",
            glossary_path=glossary_path,
        )
        self.batch_size = max(
            1, int(os.getenv("GOOGLE_TRANSLATE_BATCH_SIZE", "50"))
        )
        self.device_label = "Google Translate API"
        self._client = httpx.Client(timeout=60)

    def _resolve_device(self) -> None:
        self._device = "api"
        self.device_label = "Google Translate API"

    def _api_key(self) -> str:
        api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_TRANSLATE_API_KEY before using the Google provider"
            )
        return api_key

    def _translate_batch_google(self, texts: list[str]) -> list[str]:
        response = self._client.post(
            "https://translation.googleapis.com/language/translate/v2",
            params={"key": self._api_key()},
            json={
                "q": texts,
                "source": "ja",
                "target": "en",
                "format": "text",
            },
        )
        response.raise_for_status()
        payload = response.json()
        translations = (
            payload.get("data", {})
            .get("translations", [])
        )
        if len(translations) != len(texts):
            raise RuntimeError(
                f"Google returned {len(translations)} of {len(texts)} translations"
            )
        return [
            html.unescape(str(item.get("translatedText", "")).strip())
            for item in translations
        ]

    def _translate_missing(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        unique_texts = list(dict.fromkeys(texts))
        missing = [
            text for text in unique_texts if not self._cached_translation(text)
        ]
        total = len(unique_texts)
        processed = total - len(missing)
        self._resolve_device()
        if progress_callback:
            progress_callback(
                {
                    "phase": "preparing",
                    "progress": round(processed * 100 / total) if total else 100,
                    "processed": processed,
                    "total": total,
                    "device": self.device_label,
                    "model": self.model_name,
                }
            )
        if not missing:
            return

        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset : offset + self.batch_size]
            translations = self._translate_batch_google(batch)
            for source, translation in zip(batch, translations):
                self.cache[self.translation_cache_key(source)] = {
                    "source": source,
                    "translation": translation,
                    "model": self.model_name,
                    "provider": "google",
                }
            _atomic_write_json(self.cache_path, self.cache)
            processed += len(batch)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "translating",
                        "progress": round(processed * 100 / total),
                        "processed": processed,
                        "total": total,
                        "device": self.device_label,
                        "model": self.model_name,
                    }
                )


def create_translator(
    data_dir: Path,
    provider: str | None = None,
    model_name: str | None = None,
) -> LocalJapaneseEnglishTranslator:
    selected = normalize_translation_provider(
        provider or os.getenv("KAMI_TRANSLATION_PROVIDER") or DEFAULT_PROVIDER
    )
    if selected == "qwen":
        return LocalJapaneseEnglishTranslator(
            data_dir,
            model_name=model_name,
        )
    if selected == "deepl":
        if model_name:
            raise ValueError("--model is only supported by the qwen provider")
        return DeepLJapaneseEnglishTranslator(data_dir)
    if selected == "google":
        if model_name:
            raise ValueError("--model is only supported by the qwen provider")
        return GoogleJapaneseEnglishTranslator(data_dir)
    raise ValueError(
        "KAMI_TRANSLATION_PROVIDER must be one of: "
        + ", ".join(TRANSLATION_PROVIDERS)
    )


def translate_elements(
    data_dir: Path,
    elements: Iterable[str],
    progress_callback: ProgressCallback | None = None,
    provider: str | None = None,
) -> dict[str, int]:
    selected_provider = normalize_translation_provider(
        provider or os.getenv("KAMI_TRANSLATION_PROVIDER") or DEFAULT_PROVIDER
    )
    translator = create_translator(data_dir, provider=selected_provider)
    element_records: dict[str, list[dict]] = {}
    all_texts: list[str] = []
    normalized_elements = [element.strip().lower() for element in elements]
    for normalized in normalized_elements:
        source = element_raw_path(data_dir, normalized)
        if not source.exists():
            raise FileNotFoundError(f"Raw data file not found: {source}")
        records = _read_jsonl(source)
        element_records[normalized] = records
        all_texts.extend(translator.collect_texts(records))

    translator._translate_missing(all_texts, progress_callback)

    counts: dict[str, int] = {}
    for normalized, records in element_records.items():
        translated = [
            translator._translate_value(record)
            for record in records
        ]
        _atomic_write_jsonl(
            translated,
            element_translation_path(data_dir, normalized, selected_provider),
        )
        counts[normalized] = len(translated)
    return counts


def translate_jsonl(source: Path, destination: Path) -> int:
    """Translate a JSONL file locally for backward compatibility."""
    translator = create_translator(destination.parent)
    return translator.translate_file(source, destination)
