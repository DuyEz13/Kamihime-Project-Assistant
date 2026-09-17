"""Synchronous local update runner shared by the web and scheduled CLI."""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .crawler import crawl_all_object_elements, update_all_object_elements_latest
from .local_data import (LocalDataStore, PipelineBusy, copy_catalog, copy_translation_cache, read_json,
                         write_json, DataRelease, data_scope)
from .paths import normalize_object_type, normalize_translation_provider, translation_provider_order
from .series import reconcile_series_data, series_source_urls
from .translator import translate_object_elements
from .local_index import build_pending_index


def _now():
    return datetime.now(timezone.utc).isoformat()


def _remove_stage(store: LocalDataStore, stage: Path) -> None:
    jobs = (store.root / "jobs").resolve()
    target = stage.resolve()
    if target.parent.parent != jobs or len(target.name) != 32:
        raise ValueError("Refusing to remove a directory outside pipeline jobs")
    if target.exists():
        shutil.rmtree(target)


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _identity(record: dict) -> tuple[str, str]:
    info = record.get("info", {})
    return str(info.get("source_url", "")), str(info.get("original_name") or info.get("name", ""))


def _changed_elements(source: Path, staged: Path, object_type: str, provider: str) -> list[str]:
    changed = []
    for path in sorted((staged / object_type).glob("*/raw.jsonl")):
        relative = path.relative_to(staged)
        old, new = _records(source / relative), _records(path)
        if not new:
            raise ValueError("Source returned an empty element")
        # Treat missing entries as a source anomaly, including partial HTML parses.
        if not {_identity(r) for r in old} <= {_identity(r) for r in new}:
            raise ValueError("Source list lost existing entries; review required")
        target = path.parent / "translated" / f"{provider}.jsonl"
        translated = _records(target)
        source_urls = [r.get("info", {}).get("source_url") for r in new]
        translated_urls = [r.get("info", {}).get("source_url") for r in translated]
        if old != new or not target.exists() or source_urls != translated_urls:
            changed.append(path.parent.name)
    return changed


def _validate(root: Path, object_type: str, elements: list[str], provider: str) -> None:
    skill_key = {"kamihime": "skill", "weapon": "weapon_skills", "eidolon": "eidolon_effects"}[object_type]
    for element in elements:
        directory = root / object_type / element
        raw = _records(directory / "raw.jsonl")
        translated = _records(directory / "translated" / f"{provider}.jsonl")
        if not raw or len(raw) != len(translated):
            raise ValueError("Translation record count does not match source")
        identities = [_identity(r) for r in raw]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate source entries")
        for original, rendered in zip(raw, translated, strict=True):
            info = rendered.get("info", {})
            if not info.get("name") or not info.get("source_url"):
                raise ValueError("Record is missing name or source URL")
            if info["source_url"] != original.get("info", {}).get("source_url"):
                raise ValueError("Translation changed source identity")
            if not original.get(skill_key) or not rendered.get(skill_key):
                raise ValueError("Source or translation is missing skill data")
            if len(original[skill_key]) != len(rendered[skill_key]):
                raise ValueError("Translation changed the number of skills")
    # Exercise the same normalizer/templates' view models before publication.
    from .data_store import load_catalog_items
    with data_scope(DataRelease(root, "validation", {object_type: provider})):
        items = load_catalog_items(object_type)
        if not items or len({item["slug"] for item in items}) != len(items):
            raise ValueError("Catalog normalization returned empty or duplicate slugs")


def run_update(*, mode: str = "latest", object_type: str = "kamihime",
               provider: str = "deepl", store: LocalDataStore | None = None,
               trigger: str = "manual", acquired_lock=None, restart: bool = False,
               build_index: bool = True) -> dict:
    store = store or LocalDataStore()
    object_type = normalize_object_type(object_type)
    provider = normalize_translation_provider(provider)
    if mode not in {"latest", "database", "translate"}:
        raise ValueError("Unknown update mode")
    try:
        lock = acquired_lock or store.lock()
    except PipelineBusy:
        return {"state": "busy", "message": "Another update or index build is running"}
    with lock:
        key = f"{object_type}-{mode}-{provider}"
        job = store.root / "jobs" / key
        checkpoint = read_json(job / "job.json")
        if restart and checkpoint:
            _remove_stage(store, job / checkpoint["data"])
            checkpoint = {}
            write_json(job / "job.json", {})
        pointer = store.pointers()
        # A crash after pointer publication must not replay the already-published run.
        if checkpoint.get("run_id") == pointer.get("run_id") and checkpoint:
            _remove_stage(store, job / checkpoint["data"])
            checkpoint = {}
        parent = pointer.get("wiki", "legacy")
        if checkpoint.get("parent") != parent:
            if checkpoint:
                _remove_stage(store, job / checkpoint["data"])
            checkpoint = {}
        if not checkpoint:
            run_id = uuid4().hex
            stage = job / run_id
            current = store.wiki_release()
            source = current.root if current else store.data_dir
            copy_catalog(source, stage)
            copy_translation_cache(store.data_dir, stage)
            checkpoint = {"run_id": run_id, "parent": parent, "stage": "crawl",
                          "data": run_id, "providers": dict(current.providers) if current else {
                              kind: normalize_translation_provider() for kind in ("kamihime", "eidolon", "weapon")}}
            write_json(job / "job.json", checkpoint)
        run_id = checkpoint["run_id"]
        stage = job / checkpoint["data"]
        store.set_status(state="starting", run_id=run_id, mode=mode, object_type=object_type,
                         provider=provider, trigger=trigger, started_at=_now(), finished_at=None,
                         message="Checking source lists…", progress=0, crawl_progress={},
                         error_type=None)
        try:
            if checkpoint["stage"] == "crawl":
                # Crawl failures restart from the active catalog, not a partially written stage.
                current = store.wiki_release()
                source = current.root if current else store.data_dir
                _remove_stage(store, stage)
                copy_catalog(source, stage)
                copy_translation_cache(store.data_dir, stage)
                urls_before = series_source_urls(stage, object_type)
                def crawl_progress(value):
                    status = store.status()
                    values = status.get("crawl_progress", {})
                    processed, total = value.get("processed", 0), value.get("total", 0)
                    values[value.get("element", "")] = {**value, "progress": round(processed * 100 / total) if total else 0}
                    store.set_status(state="updating", crawl_progress=values,
                                     message=f"Checking {object_type}: {processed}/{total} new detail pages")
                store.set_status(state="updating")
                if mode == "latest":
                    result = update_all_object_elements_latest(object_type, stage, crawl_progress)
                elif mode == "database":
                    counts = crawl_all_object_elements(object_type, stage, crawl_progress)
                    result = {e: {"entries": count} for e, count in counts.items()}
                else:
                    result = {}
                urls = [url for value in result.values() for url in value.get("new_source_urls", [])]
                if mode == "database":
                    urls = list(series_source_urls(stage, object_type) - urls_before)
                reconcile_series_data(stage, object_type, changed_source_urls=urls,
                                      allow_auto_attach=object_type in {"weapon", "eidolon"})
                elements = _changed_elements(source, stage, object_type, provider)
                if mode in {"translate", "database"}:
                    elements = [p.parent.name for p in sorted((stage / object_type).glob("*/raw.jsonl"))]
                # Selecting a different rendered provider also needs an explicit publication.
                priority = translation_provider_order(checkpoint["providers"].get(object_type))
                previous_providers = {
                    next((candidate for candidate in priority
                          if (path.parent / "translated" / f"{candidate}.jsonl").exists()), None)
                    for path in (source / object_type).glob("*/raw.jsonl")
                }
                if previous_providers != {provider}:
                    elements = [p.parent.name for p in (stage / object_type).glob("*/raw.jsonl")]
                if not elements:
                    store.set_status(state="no_change", message="No new or changed records.", finished_at=_now(), progress=100)
                    write_json(job / "job.json", {})
                    _remove_stage(store, stage)
                    if build_index:
                        build_pending_index(store)
                    write_json(store.root / "runs" / f"{run_id}.json", store.status())
                    store.prune_history()
                    return store.status()
                skill_key = {"kamihime": "skill", "weapon": "weapon_skills", "eidolon": "eidolon_effects"}[object_type]
                for element in elements:
                    if any(not record.get(skill_key) for record in _records(stage / object_type / element / "raw.jsonl")):
                        raise ValueError("Source is missing skill data; wait for the wiki to be completed")
                checkpoint.update(stage="translate", elements=elements, crawl=result)
                write_json(job / "job.json", checkpoint)
            elements = checkpoint["elements"]
            if checkpoint["stage"] == "translate":
                # Conservative input ceiling bounds unattended first-time imports.
                size = sum(len((stage / object_type / e / "raw.jsonl").read_text(encoding="utf-8")) for e in elements)
                limit = int(os.getenv("KAMI_AUTO_UPDATE_MAX_INPUT_CHARS", "2000000"))
                if trigger == "scheduled" and size > limit:
                    raise ValueError("Scheduled translation input exceeds configured ceiling")
                def translation_progress(value):
                    store.set_status(**{**value, "state": "translating",
                                        "message": f"Translating {object_type} with {provider}…"})
                store.set_status(state="translating", message=f"Translating {object_type} with {provider}…")
                translate_object_elements(stage, object_type, elements, translation_progress, provider=provider)
                checkpoint["stage"] = "validate"
                write_json(job / "job.json", checkpoint)
            store.set_status(state="validating", message="Checking translated catalog…")
            try:
                _validate(stage, object_type, elements, provider)
            except (ValueError, KeyError, TypeError):
                checkpoint["stage"] = "translate"
                write_json(job / "job.json", checkpoint)
                raise
            providers = {**checkpoint["providers"], object_type: provider}
            store.set_status(state="publishing", message="Publishing wiki data…")
            release = store.publish(stage, providers, run_id)
            write_json(job / "job.json", {})
            _remove_stage(store, stage)
            store.set_status(state="completed", message="Wiki data updated. Preparing the new RAG index.",
                             progress=100, finished_at=_now(), published_release=release.release_id)
        except Exception as exc:
            # Preserve checkpoint/cache for retry; provider exception text may contain credentials.
            store.set_status(state="failed", error_type=type(exc).__name__, finished_at=_now(),
                             message=f"Update stopped at {checkpoint['stage']} ({type(exc).__name__}). "
                                     "The previous wiki data is unchanged; retry to resume.")
        if build_index and store.status()["state"] == "completed":
            build_pending_index(store)
        result = store.status()
        write_json(store.root / "runs" / f"{run_id}.json", result)
        store.prune_history()
        return result


def run_updates(object_types, *, mode: str = "latest", provider: str = "deepl",
                store: LocalDataStore | None = None, trigger: str = "manual",
                restart: bool = False, build_index: bool = True) -> dict:
    """Update several catalogs and build their combined index once at the end."""
    store = store or LocalDataStore()
    selected = list(dict.fromkeys(object_types))
    results = []
    try:
        lock = store.lock()
    except PipelineBusy:
        return {**store.status(), "state": "busy", "object_types": selected,
                "catalog_results": []}

    @contextmanager
    def borrowed_lock():
        yield lock

    with lock:
        for object_type in selected:
            result = run_update(mode=mode, object_type=object_type, provider=provider,
                                store=store, trigger=trigger, restart=restart,
                                build_index=False, acquired_lock=borrowed_lock())
            results.append(result)
            if result.get("state") == "busy":
                break
        index_result = store.status()
        if build_index and results and not any(item.get("state") == "busy" for item in results):
            index_result = build_pending_index(store)
    states = {item.get("state") for item in results}
    state = "failed" if "failed" in states else "busy" if "busy" in states else (
        "completed" if "completed" in states else "no_change")
    return {**index_result, "state": state, "object_types": selected,
            "catalog_results": results}
