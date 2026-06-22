from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "kami" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRANSLATED_DATA_DIR = DATA_DIR / "translated"
TRANSLATION_PROVIDERS = ("deepl", "google", "qwen")
DEFAULT_TRANSLATION_PROVIDER = "qwen"


def element_raw_path(data_dir: Path, element: str) -> Path:
    return data_dir / "raw" / f"kamihime_{element}_raw.jsonl"


def normalize_translation_provider(provider: str | None = None) -> str:
    selected = (
        provider
        or os.getenv("KAMI_RENDER_TRANSLATION_PROVIDER")
        or os.getenv("KAMI_TRANSLATION_PROVIDER")
        or DEFAULT_TRANSLATION_PROVIDER
    ).strip().lower()
    if selected not in TRANSLATION_PROVIDERS:
        valid = ", ".join(TRANSLATION_PROVIDERS)
        raise ValueError(f"Unknown translation provider '{selected}'. Valid: {valid}")
    return selected


def translation_provider_order(provider: str | None = None) -> list[str]:
    selected = normalize_translation_provider(provider)
    return [selected] + [
        candidate for candidate in TRANSLATION_PROVIDERS if candidate != selected
    ]


def element_translation_path(
    data_dir: Path,
    element: str,
    provider: str | None = None,
) -> Path:
    selected = normalize_translation_provider(provider)
    return data_dir / "translated" / selected / f"kamihime_{element}_en.jsonl"
