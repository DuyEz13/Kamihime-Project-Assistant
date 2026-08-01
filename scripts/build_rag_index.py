from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kami.agent.retrieval import (  # noqa: E402
    build_rag_index,
    resolve_embedding_device,
)


_LAST_PHASE = ""


def _progress(status: dict) -> None:
    global _LAST_PHASE
    phase = str(status.get("phase") or "working")
    message = str(status.get("message") or phase.replace("_", " ").title())
    processed = int(status.get("processed") or 0)
    total = int(status.get("total") or 0)
    progress = max(0, min(100, int(status.get("progress") or 0)))

    if total > 0:
        width = 36
        filled = round(width * progress / 100)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r[{bar}] {progress:3d}%  {processed}/{total}  {message}",
            end="\n" if phase == "complete" else "",
            flush=True,
        )
    elif phase != _LAST_PHASE:
        print(f"\n{message}...", flush=True)
    _LAST_PHASE = phase


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the local hybrid Qdrant index for KamiWiki"
    )
    parser.add_argument(
        "--object-type",
        action="append",
        choices=("kamihime", "eidolon", "weapon"),
        dest="object_types",
        help="Object type to index; repeat to select multiple (default: all)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dense embedding device: auto, cpu, cuda, or cuda:<index>",
    )
    args = parser.parse_args()
    device = resolve_embedding_device(args.device)
    print(f"Dense embedding device: {device}", flush=True)
    result = build_rag_index(
        args.object_types or ("kamihime", "eidolon", "weapon"),
        device=device,
        progress_callback=_progress,
    )
    print(
        f"Built {result['collection']} with {result['documents']} documents "
        f"for {', '.join(result['object_types'])}."
    )


if __name__ == "__main__":
    main()
