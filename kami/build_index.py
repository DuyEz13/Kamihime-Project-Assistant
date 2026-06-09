import json
import os
from typing import Any, List

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from .data_loader import load_jsonl


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _record_to_text(record: dict) -> str:
    info = record.get("info") if isinstance(record.get("info"), dict) else {}
    skills = record.get("skill") if isinstance(record.get("skill"), list) else []

    lines: List[str] = []
    if info:
        lines.append("Character info:")
        lines.extend(f"{key}: {_stringify_value(value)}" for key, value in info.items())

    if skills:
        lines.append("Skills:")
        for skill in skills:
            if isinstance(skill, dict):
                lines.append("; ".join(f"{key}: {_stringify_value(value)}" for key, value in skill.items()))
            else:
                lines.append(_stringify_value(skill))

    flavor = record.get("flavor")
    if flavor:
        lines.append(f"Flavor: {_stringify_value(flavor)}")

    if lines:
        return "\n".join(lines)

    text = record.get("text") or record.get("content") or record.get("body") or record.get("description")
    return _stringify_value(text or record)


def _record_metadata(record: dict, index: int) -> dict:
    info = record.get("info") if isinstance(record.get("info"), dict) else {}
    title = info.get("name") or record.get("title") or record.get("name") or ""
    return {
        "source": record.get("id") or record.get("url") or f"record-{index}",
        "title": title,
        "image": info.get("img") or "",
    }


def build_index(jsonl_path: str, output_dir: str = "vectorstore", model_name: str = "all-MiniLM-L6-v2") -> None:
    os.makedirs(output_dir, exist_ok=True)

    records = list(load_jsonl(jsonl_path))
    texts: List[str] = []
    metadatas: List[dict] = []

    for i, r in enumerate(records):
        texts.append(_record_to_text(r))
        metadatas.append(_record_metadata(r, i))

    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    vect = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vect.save_local(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS vectorstore from kami JSONL")
    parser.add_argument("jsonl", help="Path to all_kami_data.jsonl")
    parser.add_argument("--out", default="vectorstore", help="Output directory for vectorstore")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    args = parser.parse_args()
    build_index(args.jsonl, args.out, args.model)
