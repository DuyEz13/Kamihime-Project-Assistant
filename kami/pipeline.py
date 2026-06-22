import threading
import copy
from datetime import datetime, timezone

from .crawler import configured_elements, crawl_all_elements, update_all_elements_latest
from .data_store import DATA_DIR
from .paths import TRANSLATION_PROVIDERS
from .translator import translate_elements


_lock = threading.Lock()
_status = {
    "state": "idle",
    "message": "Ready",
    "started_at": None,
    "finished_at": None,
    "characters": None,
    "mode": None,
    "progress": None,
    "processed": None,
    "total": None,
    "device": None,
    "model": None,
    "crawl_progress": {},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_refresh_status() -> dict:
    with _lock:
        return copy.deepcopy(_status)


def _set_status(**values) -> None:
    with _lock:
        _status.update(values)


def _empty_crawl_progress() -> dict:
    return {
        element: {
            "processed": 0,
            "total": 0,
            "progress": 0,
            "character": "",
            "url": "",
        }
        for element in configured_elements()
    }


def _crawl_progress(progress: dict) -> None:
    element = str(progress.get("element") or "")
    if not element:
        return
    processed = int(progress.get("processed") or 0)
    total = int(progress.get("total") or 0)
    percent = round(processed * 100 / total) if total else 0
    character = str(progress.get("character") or "")
    with _lock:
        crawl_progress = copy.deepcopy(_status.get("crawl_progress") or {})
        crawl_progress[element] = {
            "processed": processed,
            "total": total,
            "progress": percent,
            "character": character,
            "url": str(progress.get("url") or ""),
        }
        totals = [
            item
            for item in crawl_progress.values()
            if int(item.get("total") or 0) > 0
        ]
        overall_processed = sum(int(item.get("processed") or 0) for item in totals)
        overall_total = sum(int(item.get("total") or 0) for item in totals)
        overall_percent = (
            round(overall_processed * 100 / overall_total)
            if overall_total
            else 0
        )
        suffix = f" Current: {character}" if character else ""
        _status.update(
            {
                "state": "updating",
                "crawl_progress": crawl_progress,
                "progress": overall_percent,
                "processed": overall_processed,
                "total": overall_total,
                "message": (
                    f"Crawling character details: "
                    f"{overall_processed}/{overall_total}"
                    f"{suffix}"
                ),
            }
        )


def _translation_progress(progress: dict) -> None:
    phase = progress["phase"]
    if phase == "loading":
        message = f"Loading translation model on {progress['device']}..."
    elif phase == "preparing":
        message = "Checking the translation cache..."
    else:
        message = (
            f"Translating with {progress['device']}: "
            f"{progress['processed']}/{progress['total']} text chunks"
        )
    _set_status(
        state="translating",
        message=message,
        **progress,
    )


def _run_update(mode: str) -> None:
    try:
        if mode == "latest":
            _set_status(
                state="updating",
                mode=mode,
                message="Checking element lists for new characters...",
                crawl_progress=_empty_crawl_progress(),
                progress=0,
                processed=0,
                total=0,
            )
            results = update_all_elements_latest(DATA_DIR, _crawl_progress)
            total = sum(result["entries"] for result in results.values())
            new_entries = sum(result["new_entries"] for result in results.values())
            crawled_details = sum(
                result["crawled_details"] for result in results.values()
            )
            removed_entries = sum(
                result["removed_entries"] for result in results.values()
            )
            _set_status(
                state="translating",
                mode=mode,
                message="Translating updated element data...",
                progress=0,
            )
            translated = translate_elements(
                DATA_DIR,
                results,
                _translation_progress,
            )
            message = (
                f"Latest update completed: {new_entries} new entries, "
                f"{crawled_details} new detail pages crawled, "
                f"{removed_entries} duplicate or stale entries removed, "
                f"{sum(translated.values())} records rendered in English"
            )
        elif mode == "database":
            _set_status(
                state="updating",
                mode=mode,
                message="Rebuilding the full character database...",
                crawl_progress=_empty_crawl_progress(),
                progress=0,
                processed=0,
                total=0,
            )
            counts = crawl_all_elements(DATA_DIR, _crawl_progress)
            _set_status(
                state="translating",
                mode=mode,
                message="Translating the rebuilt database...",
                progress=0,
            )
            translated = translate_elements(
                DATA_DIR,
                counts,
                _translation_progress,
            )
            total = sum(counts.values())
            summary = ", ".join(
                f"{element}: {count}" for element, count in counts.items()
            )
            message = (
                f"Database update completed and translated: {summary}"
            )
        else:
            raise ValueError(f"Unknown update mode: {mode}")

        _set_status(
            state="completed",
            message=message,
            characters=total,
            progress=100,
            finished_at=_now(),
        )
    except Exception as exc:
        detail = str(exc).strip()
        if not detail or detail == "Message:":
            detail = "No additional error details were provided"
        _set_status(
            state="failed",
            message=f"{type(exc).__name__}: {detail}",
            finished_at=_now(),
        )


def _run_translation(provider: str) -> None:
    try:
        elements = configured_elements()
        _set_status(
            state="translating",
            mode="translate",
            message=f"Translating existing raw database with {provider}...",
            progress=0,
            processed=0,
            total=0,
            crawl_progress={},
        )
        translated = translate_elements(
            DATA_DIR,
            elements,
            _translation_progress,
            provider=provider,
        )
        total = sum(translated.values())
        summary = ", ".join(
            f"{element}: {count}" for element, count in translated.items()
        )
        _set_status(
            state="completed",
            message=(
                f"Existing database translated with {provider}: {summary}"
            ),
            characters=total,
            progress=100,
            finished_at=_now(),
        )
    except Exception as exc:
        detail = str(exc).strip()
        if not detail or detail == "Message:":
            detail = "No additional error details were provided"
        _set_status(
            state="failed",
            message=f"{type(exc).__name__}: {detail}",
            finished_at=_now(),
        )


def start_update(mode: str) -> bool:
    with _lock:
        if _status["state"] in {"starting", "updating", "translating"}:
            return False
        _status.update(
            {
                "state": "starting",
                "mode": mode,
                "message": f"Starting {mode} update...",
                "started_at": _now(),
                "finished_at": None,
                "characters": None,
                "progress": None,
                "processed": None,
                "total": None,
                "device": None,
                "model": None,
                "crawl_progress": {},
            }
        )

    thread = threading.Thread(target=_run_update, args=(mode,), daemon=True)
    thread.start()
    return True


def start_translation(provider: str) -> bool:
    provider = provider.strip().lower()
    if provider not in TRANSLATION_PROVIDERS:
        raise ValueError(
            "Translation provider must be one of: "
            + ", ".join(TRANSLATION_PROVIDERS)
        )
    with _lock:
        if _status["state"] in {"starting", "updating", "translating"}:
            return False
        _status.update(
            {
                "state": "starting",
                "mode": "translate",
                "message": f"Starting {provider} translation...",
                "started_at": _now(),
                "finished_at": None,
                "characters": None,
                "progress": 0,
                "processed": 0,
                "total": 0,
                "device": None,
                "model": None,
                "crawl_progress": {},
            }
        )

    thread = threading.Thread(target=_run_translation, args=(provider,), daemon=True)
    thread.start()
    return True


def start_refresh() -> bool:
    """Backward-compatible alias for a full database update."""
    return start_update("database")
