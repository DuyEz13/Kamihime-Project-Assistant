from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import threading
import unicodedata
import uuid
from difflib import SequenceMatcher
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..data_store import DATA_DIR, load_catalog_items
from .documents import (
    OBJECT_TYPES,
    CatalogLoader,
    catalog_documents,
    documents_fingerprint,
    object_documents,
)
from .schemas import EntityQuery, Evidence


INDEX_DIR = DATA_DIR / "rag_index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
OBJECT_CANDIDATES = 7
OBJECT_CANDIDATES_BY_TYPE = {
    "kamihime": 7,
    "eidolon": 7,
    "weapon": 24,
}
INDEX_SCHEMA_VERSION = 4
_INDEX_LOCK = threading.RLock()
LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", normalized.casefold()))


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _fold_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _tokens(value: str) -> set[str]:
    return {token for token in normalize_text(value).split() if len(token) > 1}


def _base_name(value: str) -> str:
    return normalize_text(re.sub(r"^\s*\[[^]]+\]\s*", "", value))


def _candidate_limit(object_type: str) -> int:
    default = OBJECT_CANDIDATES_BY_TYPE.get(object_type, OBJECT_CANDIDATES)
    specific = f"KAMI_RAG_OBJECT_CANDIDATES_{object_type.upper()}"
    if os.getenv(specific) is not None:
        return _env_int(specific, default)
    if os.getenv("KAMI_RAG_OBJECT_CANDIDATES") is not None:
        return _env_int("KAMI_RAG_OBJECT_CANDIDATES", default)
    return default


def series_alias_score(query: str, aliases: Iterable[str]) -> float:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return 0.0
    query_tokens = normalized_query.split()
    best = 0.0
    for alias in aliases:
        normalized_alias = normalize_text(str(alias))
        if not normalized_alias:
            continue
        query_numbers = set(re.findall(r"\d+", normalized_query))
        alias_numbers = set(re.findall(r"\d+", normalized_alias))
        if alias_numbers and query_numbers and not alias_numbers <= query_numbers:
            continue
        if normalized_alias == normalized_query:
            return 1.0
        if normalized_alias in normalized_query or normalized_query in normalized_alias:
            best = max(best, 0.96)
        alias_tokens = normalized_alias.split()
        width = len(alias_tokens)
        windows = [
            " ".join(query_tokens[index : index + width])
            for index in range(max(1, len(query_tokens) - width + 1))
        ]
        for window in windows:
            if len(normalized_alias) >= 5:
                best = max(best, SequenceMatcher(None, normalized_alias, window).ratio())
    return best


def _series_aliases(item: dict[str, Any]) -> list[str]:
    aliases = [str(value) for value in item.get("series_aliases") or [] if value]
    series_name = str(item.get("series_name") or "")
    if series_name and series_name not in aliases:
        aliases.append(series_name)
    return aliases


def exact_series_matches(
    query: str,
    object_types: Iterable[str],
    loader: CatalogLoader = load_catalog_items,
) -> dict[str, tuple[str, str]]:
    normalized_query = normalize_text(query)
    candidates: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for item in _all_objects(object_types, loader):
        series_key = str(item.get("series_key") or "")
        if not series_key or series_key in seen:
            continue
        seen.add(series_key)
        for alias in _series_aliases(item):
            normalized_alias = normalize_text(alias)
            if contains_normalized_phrase(normalized_query, normalized_alias):
                candidates.append(
                    (
                        series_key,
                        str(item.get("series_name") or alias),
                        str(item.get("object_type") or ""),
                        normalized_alias,
                    )
                )
    maximal = [
        candidate
        for candidate in candidates
        if not any(
            candidate[3] != other[3] and candidate[3] in other[3]
            for other in candidates
        )
    ]
    return {
        key: (name, object_type)
        for key, name, object_type, _alias in maximal
    }


def _all_objects(
    object_types: Iterable[str],
    loader: CatalogLoader,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for object_type in object_types:
        values.extend(loader(object_type))
    return values


def resolve_object_variants(
    mention: str,
    object_types: Iterable[str],
    element: str | None = None,
    loader: CatalogLoader = load_catalog_items,
    limit: int | None = None,
    series_key: str | None = None,
) -> list[tuple[dict[str, Any], float]]:
    query = normalize_text(mention)
    folded_query = _fold_text(mention)
    if not query and not folded_query:
        return []
    query_tokens = set(query.split())
    objects = _all_objects(object_types, loader)
    exact_series_keys = set(
        exact_series_matches(mention, object_types, loader)
    )
    exact_object_keys = {
        (
            str(item.get("object_type") or ""),
            str(item.get("slug") or ""),
        )
        for item in objects
        if str(item.get("object_type") or "") != "kamihime"
        and (
            query == normalize_text(str(item.get("name") or ""))
        or folded_query == _fold_text(str(item.get("name") or ""))
        )
    }
    scored: list[tuple[dict[str, Any], float]] = []
    for item in objects:
        if element and str(item.get("element") or "").casefold() != element.casefold():
            continue
        item_series_key = str(item.get("series_key") or "")
        if series_key and item_series_key != series_key:
            continue
        if exact_series_keys and item_series_key not in exact_series_keys:
            continue
        item_key = (
            str(item.get("object_type") or ""),
            str(item.get("slug") or ""),
        )
        if exact_object_keys and item_key not in exact_object_keys:
            continue
        name = str(item.get("name") or "")
        normalized_name = normalize_text(name)
        folded_name = _fold_text(name)
        base = _base_name(name)
        name_tokens = set(normalized_name.split())
        score = 0.0
        if query and query == normalized_name:
            score = 120.0
        elif folded_query == folded_name:
            score = 120.0
        elif query and query == base:
            score = 110.0
        elif query and query in normalized_name:
            score = 85.0 + len(query_tokens & name_tokens)
        elif query_tokens and query_tokens <= name_tokens:
            score = 70.0 + len(query_tokens)
        aliases = _series_aliases(item)
        for alias in aliases:
            normalized_alias = normalize_text(alias)
            folded_alias = _fold_text(alias)
            if (
                (query and query == normalized_alias)
                or folded_query == folded_alias
            ):
                score = max(score, 115.0)
            elif query and normalized_alias and query in normalized_alias:
                score = max(score, 90.0 + len(query_tokens))
            elif folded_query and folded_query in folded_alias:
                score = max(score, 90.0)
            elif query and normalized_alias and normalized_alias in query:
                score = max(score, 88.0 + len(set(normalized_alias.split())))
            elif folded_alias and folded_alias in folded_query:
                score = max(score, 88.0)
        alias_similarity = series_alias_score(mention, aliases)
        if alias_similarity >= 0.88:
            score = max(score, 100.0 + alias_similarity * 10.0)
        if score:
            if element and str(item.get("element") or "").casefold() == element.casefold():
                score += 8.0
            scored.append((item, score))
    scored.sort(key=lambda pair: str(pair[0].get("name") or ""))
    scored.sort(
        key=lambda pair: str(pair[0].get("release_date") or ""),
        reverse=True,
    )
    scored.sort(key=lambda pair: pair[1], reverse=True)
    if limit is not None:
        return scored[:limit]
    selected: list[tuple[dict[str, Any], float]] = []
    used: dict[str, int] = defaultdict(int)
    for pair in scored:
        object_type = str(pair[0].get("object_type") or "kamihime")
        if used[object_type] >= _candidate_limit(object_type):
            continue
        selected.append(pair)
        used[object_type] += 1
    return selected


class SentenceTransformerEmbeddingAdapter(Embeddings):
    def __init__(self, model_name: str, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "RAG embeddings require `uv sync --extra rag`"
            ) from exc
        self.model_name = model_name
        self.device = resolve_embedding_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def _prefix(self, text: str, kind: str) -> str:
        if "e5" in self.model_name.casefold():
            return f"{kind}: {text}"
        return text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        values = self.model.encode(
            [self._prefix(text, "passage") for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [value.tolist() for value in values]

    def embed_query(self, text: str) -> list[float]:
        values = self.model.encode(
            [self._prefix(text, "query")],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return values[0].tolist()


def _embedding_model() -> str:
    return os.getenv(
        "KAMI_RAG_EMBED_MODEL",
        "intfloat/multilingual-e5-base",
    )


def resolve_embedding_device(requested: str | None = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "RAG embeddings require PyTorch; run `uv sync --extra rag`"
        ) from exc

    value = (requested or os.getenv("KAMI_RAG_DEVICE") or "auto").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if re.fullmatch(r"cuda(?::\d+)?", value):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for RAG embeddings, but this environment "
                "has no CUDA-enabled PyTorch device. Install a CUDA PyTorch "
                "build and verify `torch.cuda.is_available()` first."
            )
        if ":" in value:
            device_index = int(value.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device_index} is unavailable; detected "
                    f"{torch.cuda.device_count()} CUDA device(s)."
                )
        return value
    raise ValueError("RAG device must be auto, cpu, cuda, or cuda:<index>")


def _sparse_model() -> str:
    return os.getenv("KAMI_RAG_SPARSE_MODEL", "Qdrant/bm25")


def _read_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def index_available() -> bool:
    manifest = _read_manifest()
    return bool(
        manifest.get("collection")
        and int(manifest.get("schema_version") or 0) == INDEX_SCHEMA_VERSION
        and INDEX_DIR.exists()
    )


def build_rag_index(
    object_types: Iterable[str] = OBJECT_TYPES,
    loader: CatalogLoader = load_catalog_items,
    device: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    try:
        from langchain_qdrant import (
            FastEmbedSparse,
            QdrantVectorStore,
            RetrievalMode,
        )
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid indexing requires `uv sync --extra rag`"
        ) from exc

    selected_types = tuple(dict.fromkeys(object_types))
    documents = catalog_documents(selected_types, loader)
    if not documents:
        raise RuntimeError("No catalog documents were available for RAG indexing")
    if progress_callback:
        progress_callback(
            {
                "phase": "preparing",
                "processed": len(documents),
                "total": len(documents),
                "progress": 100,
                "message": f"Prepared {len(documents)} documents",
            }
        )
    content_fingerprint = documents_fingerprint(documents)
    resolved_device = resolve_embedding_device(device)
    fingerprint = hashlib.sha256(
        (
            f"{INDEX_SCHEMA_VERSION}:{content_fingerprint}:{_embedding_model()}:"
            f"{_sparse_model()}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    collection = f"kamiwiki_{fingerprint}"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with _INDEX_LOCK:
        current_manifest = _read_manifest()
        if (
            current_manifest.get("collection") == collection
            and int(current_manifest.get("documents") or 0) == len(documents)
        ):
            if progress_callback:
                progress_callback(
                    {
                        "phase": "complete",
                        "processed": len(documents),
                        "total": len(documents),
                        "progress": 100,
                        "cached": True,
                        "message": "RAG index is already up to date",
                    }
                )
            return current_manifest
        if progress_callback:
            progress_callback(
                {
                    "phase": "loading_models",
                    "processed": 0,
                    "total": 0,
                    "progress": 0,
                    "message": f"Loading embedding models on {resolved_device}",
                }
            )
        dense = SentenceTransformerEmbeddingAdapter(
            _embedding_model(),
            device=resolved_device,
        )
        sparse = FastEmbedSparse(model_name=_sparse_model())
        vector_size = len(dense.embed_query("Kamihime Project"))
        client = QdrantClient(path=str(INDEX_DIR / "qdrant"))
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
            client.create_collection(
                collection_name=collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            store = QdrantVectorStore(
                client=client,
                collection_name=collection,
                embedding=dense,
                sparse_embedding=sparse,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name="dense",
                sparse_vector_name="sparse",
            )
            ids = [
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        str(doc.metadata["document_id"]),
                    )
                )
                for doc in documents
            ]
            batch_size = _env_int("KAMI_RAG_INDEX_BATCH_SIZE", 64)
            total = len(documents)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "indexing",
                        "processed": 0,
                        "total": total,
                        "progress": 0,
                        "message": "Embedding and indexing documents",
                    }
                )
            for offset in range(0, total, batch_size):
                end = min(offset + batch_size, total)
                store.add_documents(
                    documents[offset:end],
                    ids=ids[offset:end],
                )
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "indexing",
                            "processed": end,
                            "total": total,
                            "progress": round(end * 100 / total),
                            "message": "Embedding and indexing documents",
                        }
                    )

            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "collection": collection,
                "fingerprint": fingerprint,
                "documents": len(documents),
                "object_types": list(selected_types),
                "embedding_model": _embedding_model(),
                "sparse_model": _sparse_model(),
                "build_device": resolved_device,
            }
            temporary = MANIFEST_PATH.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(MANIFEST_PATH)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "complete",
                        "processed": total,
                        "total": total,
                        "progress": 100,
                        "cached": False,
                        "message": "RAG index build complete",
                    }
                )
            return manifest
        finally:
            client.close()


def refresh_rag_index(object_type: str | None = None) -> dict[str, Any] | None:
    del object_type
    if not _read_manifest() and os.getenv("KAMI_RAG_AUTO_BUILD", "0") != "1":
        return None
    return build_rag_index()


@lru_cache(maxsize=4)
def _query_embedding(
    model_name: str,
    device: str,
) -> SentenceTransformerEmbeddingAdapter:
    return SentenceTransformerEmbeddingAdapter(model_name, device=device)


@lru_cache(maxsize=2)
def _query_sparse_embedding(model_name: str):
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=model_name)


def _qdrant_search(
    query: str,
    object_types: list[str],
    object_keys: list[tuple[str, str]] | None,
    k: int,
) -> list[tuple[Document, float]]:
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient, models

    with _INDEX_LOCK:
        manifest = _read_manifest()
        collection = str(manifest.get("collection") or "")
        if not collection:
            return []
        dense = _query_embedding(
            str(manifest.get("embedding_model") or _embedding_model()),
            resolve_embedding_device(),
        )
        sparse = _query_sparse_embedding(
            str(manifest.get("sparse_model") or _sparse_model())
        )
        client = QdrantClient(path=str(INDEX_DIR / "qdrant"))
        try:
            store = QdrantVectorStore(
                client=client,
                collection_name=collection,
                embedding=dense,
                sparse_embedding=sparse,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name="dense",
                sparse_vector_name="sparse",
            )
            must = [
                models.FieldCondition(
                    key="metadata.object_type",
                    match=models.MatchAny(any=object_types),
                )
            ]
            if object_keys:
                should = [
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.object_type",
                                match=models.MatchValue(value=object_type),
                            ),
                            models.FieldCondition(
                                key="metadata.slug",
                                match=models.MatchValue(value=slug),
                            ),
                        ]
                    )
                    for object_type, slug in object_keys
                ]
                query_filter = models.Filter(must=must, should=should)
            else:
                query_filter = models.Filter(must=must)
            return store.similarity_search_with_score(
                query,
                k=k,
                filter=query_filter,
            )
        finally:
            client.close()


def _lexical_score(query: str, doc: Document) -> float:
    query_tokens = _tokens(query)
    content_tokens = _tokens(doc.page_content)
    overlap = len(query_tokens & content_tokens)
    score = float(overlap)
    name = normalize_text(str(doc.metadata.get("name") or ""))
    folded_query = normalize_text(query)
    if name and name in folded_query:
        score += 20.0
    elif folded_query and folded_query in name:
        score += 12.0
    if str(doc.metadata.get("section") or "") != "basic":
        score += 0.1
    return score


def _fallback_search(
    query: str,
    object_types: list[str],
    objects: list[dict[str, Any]],
    k: int,
) -> list[tuple[Document, float]]:
    documents: list[Document] = []
    for item in objects:
        documents.extend(object_documents(item))
    scored = [(doc, _lexical_score(query, doc)) for doc in documents]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


@lru_cache(maxsize=1)
def _reranker_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        os.getenv(
            "KAMI_RAG_RERANK_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    )


def _rerank(query: str, values: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    if not values or os.getenv("KAMI_RAG_RERANK", "1") == "0":
        return values
    try:
        model = _reranker_model()
        scores = model.predict([(query, doc.page_content) for doc, _ in values])
    except Exception:
        return values
    reranked = [
        (doc, float(score) + original * 0.05)
        for (doc, original), score in zip(values, scores)
    ]
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def _section_key(value: str) -> str:
    return normalize_text(value).replace(" ", "_")


def _requested_sections(query: str, object_type: str) -> set[str] | None:
    words = set(normalize_text(query).split())
    selected: set[str] = set()

    if object_type == "kamihime":
        if words & {"stat", "stats", "hp", "attack", "level", "rarity", "release", "acquisition", "weapon"}:
            selected.add("basic")
        if words & {"burst", "ougi"}:
            selected.add("burst")
        if words & {"ability", "abilities", "skill", "skills", "cooldown", "interval", "duration"}:
            selected.update({"ability", "skill"})
        if words & {"assist", "passive"}:
            selected.add("assist")
    elif object_type == "weapon":
        if words & {"stat", "stats", "hp", "attack", "level", "rarity"}:
            selected.add("stats")
        if words & {"burst", "bursts", "ougi"}:
            selected.add("burst_effects")
        if words & {"skill", "skills", "effect", "effects"}:
            selected.add("weapon_skills")
        if words & {"unlock", "kamihime", "release", "acquisition", "type"}:
            selected.add("basic")
    elif object_type == "eidolon":
        if words & {"stat", "stats", "hp", "attack", "level", "rarity"}:
            selected.add("stats")
        if words & {"summon", "summoning", "interval", "duration"}:
            selected.add("summon_effect")
        if "main" in words:
            selected.add("main_effect")
        if "sub" in words:
            selected.add("sub_effect")
        if words & {"effect", "effects"} and not selected & {
            "summon_effect",
            "main_effect",
            "sub_effect",
        }:
            selected.update({"summon_effect", "main_effect", "sub_effect"})
        if words & {"release", "acquisition", "element", "returns"}:
            selected.add("basic")

    return selected or None


def _series_section_policy(
    items: list[dict[str, Any]],
    query: str,
) -> tuple[set[str] | None, str, str]:
    if not items:
        return None, "full_series", ""
    object_type = str(items[0].get("object_type") or "")
    if _requested_sections(query, object_type) is not None:
        return None, "explicit_sections", ""
    documents = [doc for item in items for doc in object_documents(item)]
    budget = _env_int("KAMI_RAG_SERIES_CONTEXT_CHARS", 48_000, 4_000)
    if sum(len(doc.page_content) for doc in documents) <= budget:
        return None, "full_series", ""

    priorities = {
        "eidolon": ("basic", "stats", "summon_effect", "main_effect", "sub_effect"),
        "weapon": ("basic", "stats", "burst_effects", "weapon_skills"),
        "kamihime": ("basic", "burst", "ability", "assist", "skill"),
    }.get(object_type, ("basic",))
    allowed: set[str] = set()
    used = 0
    for section in priorities:
        round_documents = [
            doc
            for doc in documents
            if _section_key(str(doc.metadata.get("section") or "")) == section
        ]
        round_size = sum(len(doc.page_content) for doc in round_documents)
        if section == "basic" or used + round_size <= budget:
            allowed.add(section)
            used += round_size
    return (
        allowed,
        "budgeted_overview",
        f"Series detail was limited to a {budget}-character context budget.",
    )


def _effect_signature(doc: Document) -> str:
    section = _section_key(str(doc.metadata.get("section") or ""))
    if section in {"", "basic", "stats"}:
        return ""
    lines = [line for line in doc.page_content.splitlines() if line.startswith("Row ")]
    text = "\n".join(lines)
    text = re.sub(r"\bName:.*?(?=\s+\|\s+|$)", "", text)
    text = re.sub(
        r"\b(fire|water|wind|thunder|light|dark|phantom)(?:-element(?:al)?)?\b",
        "<element>",
        text,
        flags=re.IGNORECASE,
    )
    return normalize_text(text)


def _hydrate_item(
    item: dict[str, Any],
    query: str,
    score: float,
    retrieval_mode: str,
    series_overview: bool = False,
    allowed_series_sections: set[str] | None = None,
    selection_mode: str = "full_entity",
    omission_reason: str = "",
    shared_effect_signatures: set[str] | None = None,
) -> list[Evidence]:
    documents = object_documents(item)
    available_sections = [
        str(doc.metadata.get("section") or "") for doc in documents
    ]
    requested = _requested_sections(query, str(item.get("object_type") or ""))
    if requested is None and series_overview and allowed_series_sections is not None:
        selected_documents = [
            doc for doc in documents
            if _section_key(str(doc.metadata.get("section") or ""))
            in allowed_series_sections
        ]
    elif requested is None:
        selected_documents = documents
    else:
        selected_documents = [
            doc
            for doc in documents
            if _section_key(str(doc.metadata.get("section") or "")) == "basic"
            or _section_key(str(doc.metadata.get("section") or "")) in requested
        ]
    included_sections = [
        str(doc.metadata.get("section") or "") for doc in selected_documents
    ]
    coverage_complete = set(included_sections) == set(available_sections)
    omitted_sections = [
        section for section in available_sections if section not in included_sections
    ]
    effective_omission_reason = omission_reason
    if omitted_sections and requested is not None:
        effective_omission_reason = (
            "Only sections explicitly requested by the user were loaded."
        )
    series_expected_elements = [
        str(value).lower()
        for value in item.get("series_expected_elements") or []
        if value
    ]
    return [
        _to_evidence(
            doc,
            score + _lexical_score(query, doc),
            available_sections=available_sections,
            included_sections=included_sections,
            coverage_complete=coverage_complete,
            retrieval_mode=retrieval_mode,
            series_expected_elements=series_expected_elements,
            series_lifecycle=str(item.get("series_lifecycle") or "complete"),
            series_catalog_elements=list(item.get("series_catalog_elements") or []),
            series_catalog_member_count=int(item.get("series_catalog_member_count") or 0),
            selection_mode=(
                "explicit_sections" if requested is not None else selection_mode
            ),
            omitted_sections=omitted_sections,
            omission_reason=(effective_omission_reason if omitted_sections else ""),
            effect_group_id=(
                f"{item.get('series_key')}:"
                f"{_section_key(str(doc.metadata.get('section') or ''))}:"
                f"{hashlib.sha1(_effect_signature(doc).encode('utf-8')).hexdigest()[:10]}"
                if item.get("series_key") and _effect_signature(doc)
                else ""
            ),
            effect_is_shared=(
                bool(_effect_signature(doc))
                and _effect_signature(doc) in (shared_effect_signatures or set())
            ),
            effect_variant_fields=(
                ["element", "effect_name", "translation_wording"]
                if _effect_signature(doc)
                and _effect_signature(doc) in (shared_effect_signatures or set())
                else []
            ),
        )
        for doc in selected_documents
    ]


def _to_evidence(
    doc: Document,
    score: float,
    *,
    available_sections: list[str] | None = None,
    included_sections: list[str] | None = None,
    coverage_complete: bool = False,
    retrieval_mode: str = "semantic",
    series_expected_elements: list[str] | None = None,
    series_lifecycle: str = "complete",
    series_catalog_elements: list[str] | None = None,
    series_catalog_member_count: int = 0,
    selection_mode: str = "full_entity",
    omitted_sections: list[str] | None = None,
    omission_reason: str = "",
    effect_group_id: str = "",
    effect_is_shared: bool = False,
    effect_variant_fields: list[str] | None = None,
) -> Evidence:
    metadata = doc.metadata
    section = str(metadata.get("section") or "")
    return {
        "content": doc.page_content,
        "score": round(float(score), 6),
        "object_type": str(metadata.get("object_type") or ""),
        "slug": str(metadata.get("slug") or ""),
        "name": str(metadata.get("name") or ""),
        "element": str(metadata.get("element") or ""),
        "section": section,
        "local_url": str(metadata.get("local_url") or ""),
        "available_sections": list(available_sections or [section]),
        "included_sections": list(included_sections or [section]),
        "coverage_complete": coverage_complete,
        "retrieval_mode": retrieval_mode,
        "selection_mode": selection_mode,
        "omitted_sections": list(omitted_sections or []),
        "omission_reason": omission_reason,
        "effect_group_id": effect_group_id,
        "effect_is_shared": effect_is_shared,
        "effect_variant_fields": list(effect_variant_fields or []),
        "series_key": str(metadata.get("series_key") or ""),
        "series_name": str(metadata.get("series_name") or ""),
        "series_elements": [],
        "series_expected_elements": list(series_expected_elements or []),
        "series_lifecycle": series_lifecycle,
        "series_catalog_elements": list(series_catalog_elements or []),
        "series_catalog_member_count": series_catalog_member_count,
        "series_retrieved_member_count": 0,
        "series_unreleased_elements": [],
        "series_missing_elements": [],
        "series_coverage_complete": False,
    }


def _annotate_series_coverage(evidence: list[Evidence]) -> list[Evidence]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        series_key = item.get("series_key") or ""
        if series_key:
            grouped[series_key].append(item)
        else:
            item["series_elements"] = [item["element"]] if item["element"] else []
            item["series_coverage_complete"] = True

    for values in grouped.values():
        retrieved_slugs = {item["slug"] for item in values if item.get("slug")}
        elements = list(
            dict.fromkeys(
                item["element"] for item in values if item.get("element")
            )
        )
        expected = list(
            dict.fromkeys(
                element
                for item in values
                for element in item.get("series_expected_elements") or []
            )
        )
        catalog = list(
            dict.fromkeys(
                element
                for item in values
                for element in item.get("series_catalog_elements") or []
            )
        ) or expected
        lifecycle = next(
            (item.get("series_lifecycle") for item in values if item.get("series_lifecycle")),
            "complete",
        )
        missing = [element for element in catalog if element not in elements]
        unreleased = (
            [element for element in expected if element not in catalog]
            if lifecycle == "releasing"
            else []
        )
        catalog_count = max(
            [item.get("series_catalog_member_count") or 0 for item in values]
            or [len(catalog)]
        ) or len(catalog)
        complete = bool(catalog_count) and len(retrieved_slugs) >= catalog_count
        for item in values:
            item["series_elements"] = elements
            item["series_expected_elements"] = expected
            item["series_missing_elements"] = missing
            item["series_lifecycle"] = str(lifecycle)
            item["series_catalog_elements"] = catalog
            item["series_catalog_member_count"] = catalog_count
            item["series_retrieved_member_count"] = len(retrieved_slugs)
            item["series_unreleased_elements"] = unreleased
            item["series_coverage_complete"] = complete
    return evidence


def retrieve_entity(
    entity: EntityQuery,
    target_types: list[str],
    loader: CatalogLoader = load_catalog_items,
) -> list[Evidence]:
    object_types = [entity.object_type] if entity.object_type else target_types
    object_types = [value for value in object_types if value in OBJECT_TYPES]
    if not object_types:
        object_types = list(OBJECT_TYPES)
    mention = entity.name or entity.mention
    variants = resolve_object_variants(
        mention,
        object_types,
        entity.element,
        loader,
        None,
        entity.series_key,
    )
    all_objects = _all_objects(object_types, loader)
    query = entity.retrieval_query or mention
    retrieval_k = _env_int("KAMI_RAG_RETRIEVAL_K", 20)

    if variants:
        evidence: list[Evidence] = []
        series_counts: dict[str, int] = defaultdict(int)
        series_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item, _score in variants:
            if item.get("series_key"):
                key = str(item["series_key"])
                series_counts[key] += 1
                series_items[key].append(item)
        series_policies = {
            key: _series_section_policy(items, query)
            for key, items in series_items.items()
            if len(items) > 1
        }
        signature_members: dict[tuple[str, str], set[str]] = defaultdict(set)
        for key, items in series_items.items():
            if len(items) <= 1:
                continue
            for item in items:
                for doc in object_documents(item):
                    signature = _effect_signature(doc)
                    if signature:
                        signature_members[(key, signature)].add(
                            str(item.get("slug") or "")
                        )
        shared_by_series: dict[str, set[str]] = defaultdict(set)
        for (key, signature), members in signature_members.items():
            if len(members) > 1:
                shared_by_series[key].add(signature)
        for item, identity_score in variants:
            series_key = str(item.get("series_key") or "")
            policy = series_policies.get(
                series_key,
                (None, "full_entity", ""),
            )
            evidence.extend(
                _hydrate_item(
                    item,
                    query,
                    identity_score,
                    "resolved_entity",
                    series_counts.get(series_key, 0) > 1,
                    policy[0],
                    policy[1],
                    policy[2],
                    shared_by_series.get(series_key, set()),
                )
            )
        return _annotate_series_coverage(evidence)

    retrieval_mode = "hybrid"
    try:
        values = (
            _qdrant_search(query, object_types, None, retrieval_k)
            if index_available()
            else []
        )
    except Exception as exc:
        LOGGER.warning("Hybrid RAG search failed; using lexical fallback: %s", exc)
        values = []
    if not values:
        retrieval_mode = "lexical_fallback"
        values = _fallback_search(query, object_types, all_objects, retrieval_k)
    values = _rerank(query, values)
    grouped: dict[tuple[str, str], list[tuple[Document, float]]] = defaultdict(list)
    for doc, score in values:
        key = (
            str(doc.metadata.get("object_type") or ""),
            str(doc.metadata.get("slug") or ""),
        )
        grouped[key].append((doc, score))
    ranked_groups = sorted(
        grouped.items(),
        key=lambda pair: max(value[1] for value in pair[1]),
        reverse=True,
    )
    used_by_type: dict[str, int] = defaultdict(int)
    limited_groups = []
    for group in ranked_groups:
        object_type = group[0][0]
        if used_by_type[object_type] >= _candidate_limit(object_type):
            continue
        limited_groups.append(group)
        used_by_type[object_type] += 1
    ranked_groups = limited_groups
    objects_by_key = {
        (str(item.get("object_type") or ""), str(item.get("slug") or "")): item
        for item in all_objects
    }
    evidence: list[Evidence] = []
    for key, group in ranked_groups:
        item = objects_by_key.get(key)
        group_score = max(value[1] for value in group)
        if item is not None:
            evidence.extend(
                _hydrate_item(item, query, group_score, retrieval_mode)
            )
        else:
            evidence.extend(
                _to_evidence(doc, score, retrieval_mode=retrieval_mode)
                for doc, score in group
            )
    return _annotate_series_coverage(evidence)


def retrieve_plan(
    entities: list[EntityQuery],
    target_types: list[str],
    standalone_question: str,
    loader: CatalogLoader = load_catalog_items,
) -> list[Evidence]:
    if not entities:
        entities = [
            EntityQuery(
                mention=standalone_question,
                retrieval_query=standalone_question,
            )
        ]
    results: list[Evidence] = []
    for entity in entities:
        results.extend(retrieve_entity(entity, target_types, loader))
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[Evidence] = []
    for item in results:
        key = (
            item["object_type"],
            item["slug"],
            item["section"],
            item["content"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
