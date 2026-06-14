import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_MODEL = "google/madlad400-3b-mt"
CACHE_FILE = ".translation_cache.json"
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
    "list_image",
    "element",
    "release_date",
}


def element_translation_path(data_dir: Path, element: str) -> Path:
    return data_dir / f"kamihime_{element}_en.jsonl"


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


def _cache_key(model_name: str, text: str) -> str:
    value = f"{model_name}\0{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
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


class LocalJapaneseEnglishTranslator:
    def __init__(
        self,
        data_dir: Path,
        model_name: str | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.model_name = (
            model_name
            or os.getenv("KAMI_TRANSLATION_MODEL")
            or DEFAULT_MODEL
        )
        self.batch_size = max(
            1, int(os.getenv("KAMI_TRANSLATION_BATCH_SIZE", "1"))
        )
        self.num_beams = max(
            1, int(os.getenv("KAMI_TRANSLATION_BEAMS", "4"))
        )
        self.max_chars = max(
            80, int(os.getenv("KAMI_TRANSLATION_MAX_CHARS", "350"))
        )
        self.cache_path = data_dir / CACHE_FILE
        self.cache = _load_cache(self.cache_path)
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        self.device_label = "CPU"

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
        if requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "KAMI_TRANSLATION_DEVICE=cuda, but CUDA is unavailable. "
                "Install a CUDA-enabled PyTorch build or use CPU."
            )
        elif requested_device not in {"cpu", "cuda"}:
            raise ValueError(
                "KAMI_TRANSLATION_DEVICE must be auto, cpu, or cuda"
            )
        else:
            self._device = requested_device

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
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local translation dependencies are missing. "
                "Run `uv sync --extra translation`."
            ) from exc

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
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_options: dict[str, Any] = {}
        if self._device == "cuda":
            model_options["torch_dtype"] = self._torch.float16
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            **model_options,
        )
        self._model.to(self._device)
        self._model.eval()

    def _model_input(self, text: str) -> str:
        if "madlad400" in self.model_name.casefold():
            return f"<2en> {text}"
        return text

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

        self._load_model(progress_callback, total, processed)
        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset : offset + self.batch_size]
            encoded = self._tokenizer(
                [self._model_input(text) for text in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
            }
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=512,
                    num_beams=self.num_beams,
                )
            translations = self._tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            for source, translation in zip(batch, translations):
                self.cache[_cache_key(self.model_name, source)] = {
                    "source": source,
                    "translation": translation.strip(),
                }
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

        _atomic_write_json(self.cache_path, self.cache)

    def _cached_translation(self, text: str) -> str | None:
        entry = self.cache.get(_cache_key(self.model_name, text))
        if not isinstance(entry, dict) or entry.get("source") != text:
            return None
        translation = entry.get("translation")
        return translation if isinstance(translation, str) else None

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
        chunks = _chunk_text(text, self.max_chars)
        return "".join(
            self._cached_translation(chunk) or chunk
            if JAPANESE_RE.search(chunk)
            else chunk
            for chunk in chunks
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
                if key in PRESERVED_VALUE_KEYS:
                    translated[translated_key] = child
                else:
                    translated[translated_key] = self._translate_value(
                        child,
                        key,
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
        return [
            self._translate_value(record)
            for record in records
        ]

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


def translate_elements(
    data_dir: Path,
    elements: Iterable[str],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    translator = LocalJapaneseEnglishTranslator(data_dir)
    element_records: dict[str, list[dict]] = {}
    all_texts: list[str] = []
    normalized_elements = [element.strip().lower() for element in elements]
    for normalized in normalized_elements:
        source = data_dir / f"kamihime_{normalized}_raw.jsonl"
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
            element_translation_path(data_dir, normalized),
        )
        counts[normalized] = len(translated)
    return counts


def translate_jsonl(source: Path, destination: Path) -> int:
    """Translate a JSONL file locally for backward compatibility."""
    translator = LocalJapaneseEnglishTranslator(destination.parent)
    return translator.translate_file(source, destination)
