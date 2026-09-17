"""Persistent document-embedding cache used by immutable local index builds."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector


CACHE_SCHEMA_VERSION = 1


def cache_key(namespace: str, text: str) -> str:
    payload = json.dumps(
        [namespace, text], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embedding_namespace(kind: str, contract: dict[str, Any]) -> str:
    payload = json.dumps(
        {"schema": CACHE_SCHEMA_VERSION, "kind": kind, **contract},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}:{hashlib.sha256(payload).hexdigest()}"


def _finite(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(value) for value in values)


def _encode_dense(vector: list[float]) -> bytes | None:
    values = [float(value) for value in vector]
    if not _finite(values):
        return None
    return struct.pack(f"<I{len(values)}f", len(values), *values)


def _decode_dense(payload: bytes, dimension: int) -> list[float] | None:
    if len(payload) < 4:
        return None
    (count,) = struct.unpack_from("<I", payload)
    if count != dimension or len(payload) != 4 + count * 4:
        return None
    values = list(struct.unpack_from(f"<{count}f", payload, 4))
    return values if _finite(values) else None


def _encode_sparse(vector: SparseVector) -> bytes | None:
    indices = [int(value) for value in vector.indices]
    values = [float(value) for value in vector.values]
    if (
        not indices
        or len(indices) != len(values)
        or len(indices) != len(set(indices))
        or any(value < 0 or value > 0xFFFFFFFF for value in indices)
        or not _finite(values)
    ):
        return None
    count = len(indices)
    return struct.pack(f"<I{count}I{count}f", count, *indices, *values)


def _decode_sparse(payload: bytes) -> SparseVector | None:
    if len(payload) < 4:
        return None
    (count,) = struct.unpack_from("<I", payload)
    if count == 0 or len(payload) != 4 + count * 8:
        return None
    indices = list(struct.unpack_from(f"<{count}I", payload, 4))
    values = list(struct.unpack_from(f"<{count}f", payload, 4 + count * 4))
    if len(indices) != len(set(indices)) or not _finite(values):
        return None
    return SparseVector(indices=indices, values=values)


class EmbeddingCache:
    def __init__(self, path: Path, max_bytes: int):
        self.path = Path(path)
        self.max_bytes = max(4096, int(max_bytes))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=2, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._pending_writes = 0
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                namespace TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                vector BLOB NOT NULL,
                checksum TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                accessed_at REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (namespace, content_hash)
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS embeddings_lru ON embeddings(accessed_at)"
        )
        self._connection.commit()
        row = self._connection.execute(
            "SELECT COALESCE(SUM(byte_size), 0) FROM embeddings"
        ).fetchone()
        self._logical_bytes = int(row[0] if row else 0)

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def logical_bytes(self) -> int:
        with self._lock:
            return self._logical_bytes

    def _get(self, kind: str, namespace: str, text: str) -> bytes | None:
        key = cache_key(namespace, text)
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT vector, checksum, kind, byte_size FROM embeddings "
                    "WHERE namespace = ? AND content_hash = ?",
                    (namespace, key),
                ).fetchone()
                if row is None:
                    return None
                payload, checksum, stored_kind = bytes(row[0]), str(row[1]), str(row[2])
                if stored_kind != kind or hashlib.sha256(payload).hexdigest() != checksum:
                    self._connection.execute(
                        "DELETE FROM embeddings WHERE namespace = ? AND content_hash = ?",
                        (namespace, key),
                    )
                    self._connection.commit()
                    self._pending_writes = 0
                    self._logical_bytes = max(0, self._logical_bytes - int(row[3]))
                    return None
                self._connection.execute(
                    "UPDATE embeddings SET accessed_at = ? "
                    "WHERE namespace = ? AND content_hash = ?",
                    (time.time(), namespace, key),
                )
                self._pending_writes += 1
                self._commit_if_needed()
                return payload
            except sqlite3.Error:
                return None

    def _put(self, kind: str, namespace: str, text: str, payload: bytes | None) -> bool:
        if payload is None:
            return False
        key = cache_key(namespace, text)
        now = time.time()
        with self._lock:
            try:
                existing = self._connection.execute(
                    "SELECT byte_size FROM embeddings "
                    "WHERE namespace = ? AND content_hash = ?",
                    (namespace, key),
                ).fetchone()
                self._connection.execute(
                    """
                    INSERT INTO embeddings
                        (namespace, content_hash, kind, vector, checksum, byte_size,
                         accessed_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, content_hash) DO UPDATE SET
                        kind=excluded.kind, vector=excluded.vector,
                        checksum=excluded.checksum, byte_size=excluded.byte_size,
                        accessed_at=excluded.accessed_at
                    """,
                    (
                        namespace,
                        key,
                        kind,
                        payload,
                        hashlib.sha256(payload).hexdigest(),
                        len(payload),
                        now,
                        now,
                    ),
                )
                self._pending_writes += 1
                previous_size = int(existing[0]) if existing else 0
                self._logical_bytes += len(payload) - previous_size
                if self._logical_bytes > self.max_bytes:
                    self.prune()
                else:
                    self._commit_if_needed()
                return True
            except sqlite3.Error:
                return False

    def get_dense(self, namespace: str, text: str, *, dimension: int) -> list[float] | None:
        payload = self._get("dense", namespace, text)
        if payload is None:
            return None
        vector = _decode_dense(payload, dimension)
        if vector is None:
            self.delete(namespace, text)
        return vector

    def put_dense(self, namespace: str, text: str, vector: list[float]) -> bool:
        return self._put("dense", namespace, text, _encode_dense(vector))

    def get_sparse(self, namespace: str, text: str) -> SparseVector | None:
        payload = self._get("sparse", namespace, text)
        if payload is None:
            return None
        vector = _decode_sparse(payload)
        if vector is None:
            self.delete(namespace, text)
        return vector

    def put_sparse(self, namespace: str, text: str, vector: SparseVector) -> bool:
        return self._put("sparse", namespace, text, _encode_sparse(vector))

    def delete(self, namespace: str, text: str) -> None:
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT byte_size FROM embeddings "
                    "WHERE namespace = ? AND content_hash = ?",
                    (namespace, cache_key(namespace, text)),
                ).fetchone()
                self._connection.execute(
                    "DELETE FROM embeddings WHERE namespace = ? AND content_hash = ?",
                    (namespace, cache_key(namespace, text)),
                )
                self._connection.commit()
                self._pending_writes = 0
                if row:
                    self._logical_bytes = max(0, self._logical_bytes - int(row[0]))
            except sqlite3.Error:
                return

    def prune(self) -> None:
        with self._lock:
            try:
                while self._logical_bytes > self.max_bytes:
                    rows = self._connection.execute(
                        "SELECT rowid, byte_size FROM embeddings "
                        "ORDER BY accessed_at ASC LIMIT 64"
                    ).fetchall()
                    if not rows:
                        break
                    self._connection.executemany(
                        "DELETE FROM embeddings WHERE rowid = ?",
                        [(int(row[0]),) for row in rows],
                    )
                    self._connection.commit()
                    self._pending_writes = 0
                    self._logical_bytes = max(
                        0,
                        self._logical_bytes - sum(int(row[1]) for row in rows),
                    )
            except sqlite3.Error:
                return

    def _commit_if_needed(self) -> None:
        if self._pending_writes >= 128:
            self._connection.commit()
            self._pending_writes = 0

    def close(self) -> None:
        with self._lock:
            self._connection.commit()
            self._pending_writes = 0
            self._connection.close()


@dataclass
class EmbeddingCacheStats:
    hits: int = 0
    misses: int = 0
    computed_unique: int = 0
    load_seconds: float = 0.0
    embed_seconds: float = 0.0


class _LazyModel:
    def __init__(self, factory: Callable[[], Any], stats: EmbeddingCacheStats):
        self.factory = factory
        self.stats = stats
        self.value: Any | None = None

    def get(self) -> Any:
        if self.value is None:
            started = time.perf_counter()
            self.value = self.factory()
            self.stats.load_seconds += time.perf_counter() - started
        return self.value


class CachedDenseEmbeddings(Embeddings):
    def __init__(self, *, cache: EmbeddingCache, namespace: str, dimension: int,
                 factory: Callable[[], Embeddings]):
        self.cache = cache
        self.namespace = namespace
        self.dimension = dimension
        self.stats = EmbeddingCacheStats()
        self._model = _LazyModel(factory, self.stats)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        unique = list(dict.fromkeys(texts))
        values: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in unique:
            vector = self.cache.get_dense(self.namespace, text, dimension=self.dimension)
            if vector is None:
                missing.append(text)
            else:
                values[text] = vector
        self.stats.hits += len(unique) - len(missing)
        self.stats.misses += len(missing)
        if missing:
            started = time.perf_counter()
            embedded = self._model.get().embed_documents(missing)
            self.stats.embed_seconds += time.perf_counter() - started
            if len(embedded) != len(missing):
                raise ValueError("Dense embedding model returned the wrong vector count")
            for text, vector in zip(missing, embedded, strict=True):
                normalized = [float(value) for value in vector]
                if len(normalized) != self.dimension or not _finite(normalized):
                    raise ValueError("Dense embedding model returned an invalid vector")
                values[text] = normalized
                self.cache.put_dense(self.namespace, text, normalized)
            self.stats.computed_unique += len(missing)
        return [values[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._model.get().embed_query(text)


class CachedSparseEmbeddings(SparseEmbeddings):
    def __init__(self, *, cache: EmbeddingCache, namespace: str,
                 factory: Callable[[], SparseEmbeddings]):
        self.cache = cache
        self.namespace = namespace
        self.stats = EmbeddingCacheStats()
        self._model = _LazyModel(factory, self.stats)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        unique = list(dict.fromkeys(texts))
        values: dict[str, SparseVector] = {}
        missing: list[str] = []
        for text in unique:
            vector = self.cache.get_sparse(self.namespace, text)
            if vector is None:
                missing.append(text)
            else:
                values[text] = vector
        self.stats.hits += len(unique) - len(missing)
        self.stats.misses += len(missing)
        if missing:
            started = time.perf_counter()
            embedded = self._model.get().embed_documents(missing)
            self.stats.embed_seconds += time.perf_counter() - started
            if len(embedded) != len(missing):
                raise ValueError("Sparse embedding model returned the wrong vector count")
            for text, vector in zip(missing, embedded, strict=True):
                payload = _encode_sparse(vector)
                if payload is None:
                    raise ValueError("Sparse embedding model returned an invalid vector")
                values[text] = vector
                self.cache.put_sparse(self.namespace, text, vector)
            self.stats.computed_unique += len(missing)
        return [values[text] for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._model.get().embed_query(text)
