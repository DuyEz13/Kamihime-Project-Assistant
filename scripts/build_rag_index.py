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


def _completion_message(result: dict) -> str:
    message = f"Built {result['collection']} with {result['documents']} documents."
    cache = result.get("embedding_cache") or {}
    if cache:
        message += (
            " Embedding cache: "
            f"dense {cache.get('dense_hits', 0)} hit/{cache.get('dense_misses', 0)} miss; "
            f"sparse {cache.get('sparse_hits', 0)} hit/{cache.get('sparse_misses', 0)} miss."
        )
    return message


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
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
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
        default="cuda",
        help="Dense embedding device: cuda (default), cpu, auto, or cuda:<index>",
    )
    args = parser.parse_args()
    device = resolve_embedding_device(args.device)
    print(f"Dense embedding device: {device}", flush=True)
    result = build_local_index(args.object_types, device)
    print(_completion_message(result))


def build_local_index(object_types=None, device="auto") -> dict:
    from kami.local_data import LocalDataStore, read_json
    from kami.local_index import build_pending_index

    store = LocalDataStore()
    with store.lock():
        selected = object_types or ("kamihime", "eidolon", "weapon")
        if set(selected) != {"kamihime", "eidolon", "weapon"}:
            raise ValueError("Build all object types to activate a local data release for chat")
        store.ensure_wiki_release()
        result = build_pending_index(store, device=device, force=True, progress_callback=_progress)
        if result.get("index_state") != "ready":
            raise RuntimeError(result.get("message", "Index build did not complete"))
        _, target = store.chat_target()
        return read_json(target / "manifest.json")


if __name__ == "__main__":
    main()
