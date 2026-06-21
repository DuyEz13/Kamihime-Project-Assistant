"""Translate a small sample without overwriting KamiWiki element data."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kami.paths import DATA_DIR, element_raw_path  # noqa: E402
from kami.translator import create_translator  # noqa: E402


ELEMENTS = ("fire", "water", "wind", "thunder", "light", "dark")


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _progress(status: dict) -> None:
    print(
        (
            f"\r{status['phase'].capitalize()}: {status['progress']:3d}% "
            f"({status['processed']}/{status['total']}) "
            f"on {status['device']}"
        ),
        end="",
        flush=True,
    )
    if status["progress"] == 100:
        print()


def _sample_texts(
    translator,
    element: str,
    count: int,
    seed: int,
) -> list[str]:
    source = element_raw_path(DATA_DIR, element)
    if not source.exists():
        raise FileNotFoundError(f"Raw element data not found: {source}")

    texts = list(
        dict.fromkeys(translator.collect_texts(_read_jsonl(source)))
    )
    if not texts:
        raise RuntimeError(f"No Japanese text was found in {source}")

    random.Random(seed).shuffle(texts)
    return texts[:count]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test Japanese-to-English translation on a few values without "
            "modifying element English files."
        )
    )
    parser.add_argument(
        "--element",
        choices=ELEMENTS,
        default="fire",
        help="Raw element file used for sampling (default: fire).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of unique Japanese samples to translate (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random sampling seed (default: 42).",
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Japanese text to translate. Repeat for multiple values.",
    )
    parser.add_argument(
        "--provider",
        choices=("qwen", "deepl"),
        help="Translation provider. Defaults to KAMI_TRANSLATION_PROVIDER.",
    )
    parser.add_argument(
        "--model",
        help="Override KAMI_TRANSLATION_MODEL for the qwen provider.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override KAMI_TRANSLATION_DEVICE for this test.",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    load_dotenv(ROOT_DIR / ".env")
    if args.device:
        os.environ["KAMI_TRANSLATION_DEVICE"] = args.device

    translator = create_translator(
        DATA_DIR,
        provider=args.provider,
        model_name=args.model,
    )
    source_texts = args.text or _sample_texts(
        translator,
        args.element,
        args.count,
        args.seed,
    )

    print(f"Model: {translator.model_name}")
    translations = translator.translate_texts(source_texts, _progress)
    print(f"Device: {translator.device_label}")
    if hasattr(translator, "usage"):
        usage = translator.usage()
        print(
            "DeepL usage: "
            f"{usage['count']}/{usage['limit']} source characters"
        )

    for index, (source, translated) in enumerate(
        zip(source_texts, translations),
        start=1,
    ):
        print(f"\n[{index}] Japanese:\n{source}")
        print(f"English:\n{translated}")

    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
