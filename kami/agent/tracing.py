from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

from ..paths import DATA_DIR
from .schemas import Evidence, QueryPlan


LOGGER = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text_fingerprint(value: str) -> dict[str, Any]:
    encoded = value.encode("utf-8")
    return {
        "chars": len(value),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


class ChatTrace:
    def __init__(
        self,
        session_id: str,
        client_id: str,
        provider: str,
        model: str,
    ) -> None:
        self._started = time.perf_counter()
        self._finalized = False
        self._include_content = _env_flag(
            "KAMI_CHAT_TRACE_INCLUDE_CONTENT",
            False,
        )
        self._path = Path(
            os.getenv("KAMI_CHAT_TRACE_PATH")
            or DATA_DIR / "chat_traces.jsonl"
        )
        self._max_bytes = _env_int(
            "KAMI_CHAT_TRACE_MAX_BYTES",
            10 * 1024 * 1024,
            1,
        )
        self._backup_count = _env_int(
            "KAMI_CHAT_TRACE_BACKUP_COUNT",
            5,
            0,
        )
        self._record: dict[str, Any] = {
            "schema_version": 1,
            "trace_id": uuid.uuid4().hex,
            "session_id": session_id,
            "client_id": client_id,
            "started_at": _now_iso(),
            "provider": provider,
            "model": model,
            "status": "running",
            "graph": {"nodes": []},
            "model_calls": [],
        }

    @property
    def trace_id(self) -> str:
        return str(self._record["trace_id"])

    @contextmanager
    def node(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        status = "ok"
        error_type = None
        try:
            yield
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            self._record["graph"]["nodes"].append(
                {
                    "name": name,
                    "status": status,
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                    "error_type": error_type,
                }
            )

    def record_request(self, message: str) -> None:
        request = _text_fingerprint(message)
        if self._include_content:
            request["message"] = message
        self._record["request"] = request

    def record_plan(self, plan: QueryPlan) -> None:
        if self._include_content:
            self._record["plan"] = plan.model_dump(mode="json")
            return
        self._record["plan"] = {
            "in_domain": plan.in_domain,
            "intent": plan.intent,
            "target_types": list(plan.target_types),
            "entities": [
                {
                    "name": entity.name,
                    "mention": entity.mention,
                    "object_type": entity.object_type,
                    "series_key": entity.series_key,
                    "element": entity.element,
                    "skills": list(entity.skills),
                }
                for entity in plan.entities
            ],
            "elements": list(plan.elements),
            "sort_by": plan.sort_by,
            "sort_order": plan.sort_order,
            "result_limit": plan.result_limit,
            "include_ties": plan.include_ties,
            "requested_sections": dict(plan.requested_sections),
            "needs_clarification": plan.needs_clarification,
        }

    def record_retrieval(
        self,
        diagnostics: dict[str, Any],
        evidence: list[Evidence] | list[dict[str, Any]],
    ) -> None:
        retrieval = dict(diagnostics)
        diagnostic_objects = retrieval.pop("evidence", None)
        if diagnostic_objects is not None:
            retrieval["objects"] = diagnostic_objects
        retrieval["evidence"] = []
        for source in evidence:
            item = dict(source)
            if not self._include_content:
                item.pop("content", None)
            retrieval["evidence"].append(item)
        self._record["retrieval"] = retrieval

    def record_model_call(self, event: dict[str, Any]) -> None:
        self._record["model_calls"].append(dict(event))

    def record_response(
        self,
        required_language: str,
        actual_language: str,
        answer: str,
        language_retry_count: int,
    ) -> None:
        response = {
            "required_language": required_language,
            "actual_language": actual_language,
            "language_retry_count": language_retry_count,
            "answer_chars": len(answer),
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        }
        if self._include_content:
            response["answer"] = answer
        self._record["response"] = response

    def record_error(self, exc: Exception) -> None:
        self._record["status"] = "error"
        self._record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    def _model_totals(self) -> dict[str, Any]:
        calls = list(self._record.get("model_calls") or [])
        reported = [call for call in calls if call.get("usage_available")]
        return {
            "model_call_count": len(calls),
            "calls_with_usage": len(reported),
            "reported_input_tokens": sum(
                int(call.get("input_tokens") or 0) for call in reported
            ),
            "reported_output_tokens": sum(
                int(call.get("output_tokens") or 0) for call in reported
            ),
            "reported_total_tokens": sum(
                int(call.get("total_tokens") or 0) for call in reported
            ),
            "model_duration_ms": round(
                sum(float(call.get("duration_ms") or 0) for call in calls),
                3,
            ),
        }

    def _write_record(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        message = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        with _WRITE_LOCK:
            handler = RotatingFileHandler(
                self._path,
                maxBytes=self._max_bytes,
                backupCount=self._backup_count,
                encoding="utf-8",
                delay=True,
            )
            try:
                handler.setFormatter(logging.Formatter("%(message)s"))
                log_record = logging.LogRecord(
                    name=__name__,
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=0,
                    msg=message,
                    args=(),
                    exc_info=None,
                )
                handler.emit(log_record)
            finally:
                handler.close()

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._record.get("status") == "running":
            self._record["status"] = "ok"
        self._record["finished_at"] = _now_iso()
        self._record["duration_ms"] = round(
            (time.perf_counter() - self._started) * 1000,
            3,
        )
        self._record["totals"] = self._model_totals()
        try:
            self._write_record(self._record)
        except Exception as exc:
            LOGGER.error("Could not write chat trace: %s", exc)


class NullChatTrace:
    trace_id = ""

    @contextmanager
    def node(self, _name: str) -> Iterator[None]:
        yield

    def record_request(self, _message: str) -> None:
        return None

    def record_plan(self, _plan: QueryPlan) -> None:
        return None

    def record_retrieval(
        self,
        _diagnostics: dict[str, Any],
        _evidence: list[Evidence] | list[dict[str, Any]],
    ) -> None:
        return None

    def record_model_call(self, _event: dict[str, Any]) -> None:
        return None

    def record_response(
        self,
        _required_language: str,
        _actual_language: str,
        _answer: str,
        _language_retry_count: int,
    ) -> None:
        return None

    def record_error(self, _exc: Exception) -> None:
        return None

    def finalize(self) -> None:
        return None


def create_chat_trace(
    session_id: str,
    client_id: str,
    provider: str,
    model: str,
) -> ChatTrace | NullChatTrace:
    if not _env_flag("KAMI_CHAT_TRACE_ENABLED", True):
        return NullChatTrace()
    return ChatTrace(session_id, client_id, provider, model)
