"""Build immutable local GPU indexes and lease matching runtimes per chat turn."""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .local_data import DataRelease, LocalDataStore, PipelineBusy, data_scope, read_json, write_json
from .agent.retrieval import (
    RetrievalRuntime, _open_runtime_components, build_rag_index,
    expected_index_metadata, resolve_embedding_device, validate_index_metadata,
)
from .agent.embedding_cache import EmbeddingCache

LOGGER = logging.getLogger(__name__)


def _embedding_cache_context(store: LocalDataStore):
    enabled = os.getenv("KAMI_LOCAL_RAG_EMBED_CACHE", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return nullcontext(None)
    try:
        max_mb = max(1, int(os.getenv("KAMI_LOCAL_RAG_EMBED_CACHE_MAX_MB", "128")))
        return EmbeddingCache(
            store.root / "embedding_cache.sqlite3",
            max_bytes=max_mb * 1024 * 1024,
        )
    except Exception as error:
        LOGGER.warning("Embedding cache is unavailable; continuing without it: %s", type(error).__name__)
        return nullcontext(None)


def build_pending_index(store: LocalDataStore, *, device: str = "cuda", force: bool = False,
                        progress_callback: Callable | None = None) -> dict:
    """Caller owns the writer lock. Failures leave both published pointers intact."""
    release = store.wiki_release()
    if release is None or (not force and not store.status()["pending_index"]):
        return store.status()
    try:
        lock = store.index_lock()
    except PipelineBusy:
        return store.status()
    with lock:
        store.set_status(state="indexing", index_state="building", index_progress=0,
                         index_error_type=None,
                         dense_cache_hits=0, dense_cache_misses=0,
                         sparse_cache_hits=0, sparse_cache_misses=0,
                         embedded_unique=0,
                         message="Wiki updated. Building the new RAG index on GPU; chat uses the previous index.")
        index_id = uuid4().hex
        target = store.root / "indexes" / index_id
        def progress(value):
            metrics = {
                key: value[key]
                for key in (
                    "dense_cache_hits", "dense_cache_misses",
                    "sparse_cache_hits", "sparse_cache_misses",
                    "embedded_unique",
                )
                if key in value
            }
            store.set_status(index_progress=value.get("progress", 0),
                             index_phase=value.get("phase", "building"), **metrics)
            if progress_callback:
                progress_callback(value)
        try:
            selected_device = resolve_embedding_device(device)
            if device.startswith("cuda") and not selected_device.startswith("cuda"):
                raise RuntimeError("GPU indexing requires CUDA")
            with _embedding_cache_context(store) as embedding_cache:
                with data_scope(release):
                    manifest = build_rag_index(
                        device=selected_device, index_dir=target,
                        batch_size=max(1, int(os.getenv("KAMI_LOCAL_RAG_BATCH_SIZE", "8"))),
                        local_release_id=release.release_id, progress_callback=progress,
                        embedding_cache=embedding_cache,
                    )
                if embedding_cache is not None:
                    embedding_cache.prune()
            # build_rag_index publishes its manifest only after a hybrid smoke query.
            saved = read_json(target / "manifest.json")
            if saved != manifest or saved.get("local_release_id") != release.release_id:
                raise RuntimeError("Index manifest does not match the selected data release")
            store.activate_chat(release.release_id, index_id)
            store.prune_artifacts()
            store.set_status(state="completed", index_state="ready", index_progress=100,
                             message="Wiki and chatbot data updated.", index_finished_at=time.time())
        except Exception as exc:
            store.set_status(state="completed", index_state="failed", index_error_type=type(exc).__name__,
                             message=f"Wiki updated; GPU index build failed ({type(exc).__name__}). "
                                     "Chat still uses its previous index. The next update will retry indexing.")
        return store.status()


def open_local_runtime(index: Path, manifest: dict) -> RetrievalRuntime:
    expected = expected_index_metadata(catalog_fingerprint=manifest["catalog_fingerprint"],
                                       documents=int(manifest["documents"]))
    validate_index_metadata(manifest, expected)
    # Keep query embeddings on CPU by default, leaving the laptop's VRAM for builds.
    client, vector_store = _open_runtime_components(
        index, manifest, device=os.getenv("KAMI_LOCAL_RAG_QUERY_DEVICE", "cpu"))
    return RetrievalRuntime(client, vector_store, manifest["collection"], dict(manifest), index)


@dataclass
class RuntimeLease:
    release: DataRelease | None
    runtime: RetrievalRuntime | None
    index_dir: Path
    users: int = 0
    artifact_leases: tuple = ()


class LocalRuntimeManager:
    def __init__(self, store: LocalDataStore, *, opener=open_local_runtime):
        self.store = store
        self.opener = opener
        self._lock = threading.RLock()
        self._current: RuntimeLease | None = None
        self._retired: list[RuntimeLease] = []
        self._failed_key = None
        self._retry_at = 0.0

    def _select(self) -> RuntimeLease:
        release, index = self.store.chat_target()
        key = (release.release_id if release else "legacy", str(index))
        current = self._current
        if current and current.index_dir == index:
            # First wiki publication introduces a baseline snapshot without
            # changing the legacy index. Do not reopen its already-owned Qdrant path.
            return current
        if current and self._failed_key == key and time.monotonic() < self._retry_at:
            return current
        try:
            artifact_leases = []
            if index.parent == self.store.root / "indexes":
                artifact_leases.append(self.store.artifact_lease("index", index.name))
            if release:
                artifact_leases.append(self.store.artifact_lease("release", release.release_id))
            manifest = read_json(index / "manifest.json")
            if index.parent == self.store.root / "indexes" and (
                not manifest or not release or manifest.get("local_release_id") != release.release_id
            ):
                raise ValueError("Local index/data release mismatch")
            runtime = self.opener(index, manifest) if manifest else None
            selected = RuntimeLease(release, runtime, index, artifact_leases=tuple(artifact_leases))
        except Exception as exc:
            for lease in reversed(locals().get("artifact_leases", [])):
                lease.close()
            self._failed_key, self._retry_at = key, time.monotonic() + 30
            if current is None:
                raise
            LOGGER.warning("Keeping previous local chat runtime after activation failure (%s)", type(exc).__name__)
            return current
        self._current = selected
        self._failed_key = None
        if current:
            if current.users:
                self._retired.append(current)
            else:
                self._close_lease(current)
        return selected

    def _close_lease(self, lease: RuntimeLease):
        if lease.runtime:
            lease.runtime.client.close()
        for artifact_lease in reversed(lease.artifact_leases):
            artifact_lease.close()
        self.store.prune_artifacts()

    @contextmanager
    def acquire(self):
        with self._lock:
            lease = self._select()
            lease.users += 1
        try:
            yield lease
        finally:
            with self._lock:
                lease.users -= 1
                if not lease.users and lease in self._retired:
                    self._retired.remove(lease)
                    self._close_lease(lease)

    def close(self):
        with self._lock:
            leases = [*self._retired, *([self._current] if self._current else [])]
            for lease in leases:
                if lease.users:
                    raise RuntimeError("Cannot close local retrieval during an active chat")
                self._close_lease(lease)
            self._current = None
            self._retired.clear()
