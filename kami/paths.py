from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "kami" / "data"
LEGACY_RAW_DATA_DIR = DATA_DIR / "raw"
LEGACY_TRANSLATED_DATA_DIR = DATA_DIR / "translated"
# Backward-compatible names for callers that still inspect the old layout.
RAW_DATA_DIR = LEGACY_RAW_DATA_DIR
TRANSLATED_DATA_DIR = LEGACY_TRANSLATED_DATA_DIR
TRANSLATION_PROVIDERS = ("deepl", "google", "qwen")
DEFAULT_TRANSLATION_PROVIDER = "qwen"
OBJECT_TYPES = ("kamihime", "eidolon", "weapon")
OBJECT_TYPE_ALIASES = {
    "kami": "kamihime",
    "kamihime": "kamihime",
    "kamihimes": "kamihime",
    "eidolon": "eidolon",
    "eidolons": "eidolon",
    "weapon": "weapon",
    "weapons": "weapon",
}
BASE_ELEMENTS = ("fire", "water", "wind", "thunder", "light", "dark")
PHANTOM_ELEMENTS = (*BASE_ELEMENTS, "phantom")
OBJECT_ELEMENTS = {
    "kamihime": BASE_ELEMENTS,
    "eidolon": PHANTOM_ELEMENTS,
    "weapon": PHANTOM_ELEMENTS,
}


def normalize_object_type(object_type: str | None = None) -> str:
    selected = (object_type or "kamihime").strip().lower()
    normalized = OBJECT_TYPE_ALIASES.get(selected)
    if normalized is None:
        valid = ", ".join(OBJECT_TYPES)
        raise ValueError(f"Unknown object type '{selected}'. Valid: {valid}")
    return normalized


def object_element_dir(
    data_dir: Path,
    object_type: str,
    element: str,
) -> Path:
    selected = normalize_object_type(object_type)
    normalized_element = element.strip().lower()
    valid_elements = OBJECT_ELEMENTS[selected]
    if normalized_element not in valid_elements:
        valid = ", ".join(valid_elements)
        raise ValueError(
            f"Unknown {selected} element '{normalized_element}'. Valid: {valid}"
        )
    return data_dir / selected / normalized_element


def object_raw_path(
    data_dir: Path,
    object_type: str,
    element: str,
) -> Path:
    return object_element_dir(data_dir, object_type, element) / "raw.jsonl"


def element_raw_path(data_dir: Path, element: str) -> Path:
    """Return the Kamihime raw path for backward-compatible callers."""
    return object_raw_path(data_dir, "kamihime", element)


def legacy_element_raw_path(data_dir: Path, element: str) -> Path:
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
    """Return the Kamihime translation path for backward-compatible callers."""
    return object_translation_path(data_dir, "kamihime", element, provider)


def object_translation_path(
    data_dir: Path,
    object_type: str,
    element: str,
    provider: str | None = None,
) -> Path:
    selected = normalize_translation_provider(provider)
    return (
        object_element_dir(data_dir, object_type, element)
        / "translated"
        / f"{selected}.jsonl"
    )


def legacy_element_translation_path(
    data_dir: Path,
    element: str,
    provider: str | None = None,
) -> Path:
    selected = normalize_translation_provider(provider)
    return data_dir / "translated" / selected / f"kamihime_{element}_en.jsonl"
