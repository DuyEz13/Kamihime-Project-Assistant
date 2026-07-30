"""Crawl Kamihime, Eidolon, or Weapon raw JSONL data."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kami.crawler import (  # noqa: E402
    crawl_all_object_elements,
    crawl_object_element_to_jsonl,
    update_all_object_elements_latest,
    update_object_element_latest,
)
from kami.paths import DATA_DIR, OBJECT_ELEMENTS, OBJECT_TYPES  # noqa: E402


def _progress(status: dict) -> None:
    object_type = status.get("object_type") or ""
    element = status.get("element") or ""
    processed = int(status.get("processed") or 0)
    total = int(status.get("total") or 0)
    current = status.get("character") or ""
    print(
        f"\r{object_type}/{element}: {processed}/{total} {current}",
        end="",
        flush=True,
    )
    if total and processed >= total:
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl raw SSR object data without running the translation pipeline."
        )
    )
    parser.add_argument(
        "object_type",
        choices=OBJECT_TYPES,
        help="Wiki object catalog to crawl.",
    )
    parser.add_argument(
        "--element",
        choices=OBJECT_ELEMENTS["eidolon"],
        help="Crawl one element. Omit to use the configured element list.",
    )
    parser.add_argument(
        "--mode",
        choices=("latest", "database"),
        default="latest",
        help=(
            "latest reuses existing details; database recrawls every detail "
            "(default: latest)."
        ),
    )
    parser.add_argument(
        "--source-url",
        help="Override the list URL; only valid together with --element.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    if (
        args.element
        and args.element not in OBJECT_ELEMENTS[args.object_type]
    ):
        parser.error(
            f"{args.object_type} does not support the {args.element} element"
        )
    if args.source_url and not args.element:
        parser.error("--source-url requires --element")

    if args.element:
        if args.mode == "database":
            result = {
                args.element: crawl_object_element_to_jsonl(
                    args.object_type,
                    args.element,
                    DATA_DIR,
                    source_url=args.source_url,
                    progress_callback=_progress,
                )
            }
        else:
            result = {
                args.element: update_object_element_latest(
                    args.object_type,
                    args.element,
                    DATA_DIR,
                    source_url=args.source_url,
                    progress_callback=_progress,
                )
            }
    elif args.mode == "database":
        result = crawl_all_object_elements(
            args.object_type,
            DATA_DIR,
            _progress,
        )
    else:
        result = update_all_object_elements_latest(
            args.object_type,
            DATA_DIR,
            _progress,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
