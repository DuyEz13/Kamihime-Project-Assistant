"""Backward-compatible entry point for the agentic RAG index builder."""

from __future__ import annotations

import os

from .agent.retrieval import build_rag_index


def build_index(
    jsonl_path: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Build the unified Kamihime/Eidolon/Weapon hybrid index.

    ``jsonl_path`` and ``output_dir`` remain accepted for callers of the old
    FAISS helper. Catalog data and the index directory now come from the shared
    application paths, so those two arguments are intentionally ignored.
    """
    del jsonl_path, output_dir
    if model_name:
        os.environ["KAMI_RAG_EMBED_MODEL"] = model_name
    return build_rag_index()


if __name__ == "__main__":
    result = build_index()
    print(
        f"Built {result['collection']} with {result['documents']} documents."
    )
