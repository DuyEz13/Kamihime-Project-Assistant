"""Local web adapter for the durable scheduled/manual data runner."""
from __future__ import annotations

import threading

from .local_data import LocalDataStore, PipelineBusy
from .paths import DATA_DIR, normalize_object_type, normalize_translation_provider
from .pipeline_runner import run_update


DEFAULT_UPDATE_TRANSLATION_PROVIDER = "deepl"


def get_refresh_status() -> dict:
    return LocalDataStore(DATA_DIR).status()


def _start(mode: str, object_type: str, provider: str) -> bool:
    object_type = normalize_object_type(object_type)
    provider = normalize_translation_provider(provider)
    store = LocalDataStore(DATA_DIR)
    try:
        lock = store.lock()
    except PipelineBusy:
        return False
    try:
        store.set_status(state="starting", message="Starting data update…", mode=mode,
                         object_type=object_type, provider=provider, progress=0,
                         finished_at=None, crawl_progress={})
        thread = threading.Thread(
            target=run_update,
            kwargs=dict(mode=mode, object_type=object_type, provider=provider,
                        store=store, acquired_lock=lock, trigger="manual"),
            daemon=True,
        )
        thread.start()
    except BaseException:
        lock.close()
        raise
    return True


def start_update(mode: str, object_type: str = "kamihime",
                 provider: str = DEFAULT_UPDATE_TRANSLATION_PROVIDER) -> bool:
    if mode not in {"latest", "database"}:
        raise ValueError("Update mode must be latest or database")
    return _start(mode, object_type, provider)


def start_translation(provider: str, object_type: str = "kamihime") -> bool:
    return _start("translate", object_type, provider)


def start_refresh() -> bool:
    return start_update("database")
