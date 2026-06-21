from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "kami" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRANSLATED_DATA_DIR = DATA_DIR / "translated"


def element_raw_path(data_dir: Path, element: str) -> Path:
    return data_dir / "raw" / f"kamihime_{element}_raw.jsonl"


def element_translation_path(data_dir: Path, element: str) -> Path:
    return data_dir / "translated" / f"kamihime_{element}_en.jsonl"
