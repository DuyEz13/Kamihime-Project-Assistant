import threading
from datetime import datetime, timezone

from .crawler import crawl_all_elements, update_all_elements_latest
from .data_store import DATA_DIR


_lock = threading.Lock()
_status = {
    "state": "idle",
    "message": "Ready",
    "started_at": None,
    "finished_at": None,
    "characters": None,
    "mode": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_refresh_status() -> dict:
    with _lock:
        return dict(_status)


def _set_status(**values) -> None:
    with _lock:
        _status.update(values)


def _run_update(mode: str) -> None:
    try:
        if mode == "latest":
            _set_status(
                state="updating",
                mode=mode,
                message="Checking element lists for new characters...",
            )
            results = update_all_elements_latest(DATA_DIR)
            total = sum(result["entries"] for result in results.values())
            new_entries = sum(result["new_entries"] for result in results.values())
            crawled_details = sum(
                result["crawled_details"] for result in results.values()
            )
            removed_entries = sum(
                result["removed_entries"] for result in results.values()
            )
            message = (
                f"Latest update completed: {new_entries} new entries, "
                f"{crawled_details} new detail pages crawled, "
                f"{removed_entries} duplicate or stale entries removed"
            )
        elif mode == "database":
            _set_status(
                state="updating",
                mode=mode,
                message="Rebuilding the full character database...",
            )
            counts = crawl_all_elements(DATA_DIR)
            total = sum(counts.values())
            summary = ", ".join(
                f"{element}: {count}" for element, count in counts.items()
            )
            message = f"Database update completed: {summary}"
        else:
            raise ValueError(f"Unknown update mode: {mode}")

        _set_status(
            state="completed",
            message=message,
            characters=total,
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
        if _status["state"] in {"starting", "updating"}:
            return False
        _status.update(
            {
                "state": "starting",
                "mode": mode,
                "message": f"Starting {mode} update...",
                "started_at": _now(),
                "finished_at": None,
                "characters": None,
            }
        )

    thread = threading.Thread(target=_run_update, args=(mode,), daemon=True)
    thread.start()
    return True


def start_refresh() -> bool:
    """Backward-compatible alias for a full database update."""
    return start_update("database")
