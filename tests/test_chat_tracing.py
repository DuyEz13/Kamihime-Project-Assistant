import json
import logging

from kami.agent.tracing import create_chat_trace


def _read_trace(path):
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_trace_writes_one_redacted_json_record(tmp_path, monkeypatch):
    path = tmp_path / "chat_traces.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(path))
    monkeypatch.setenv("KAMI_CHAT_TRACE_INCLUDE_CONTENT", "0")
    trace = create_chat_trace(
        "session-1",
        "client-1",
        "deepseek",
        "deepseek-chat",
    )

    with trace.node("retrieve"):
        trace.record_retrieval(
            {
                "mode": "catalog_latest",
                "reason_code": "selected_maximum_release_date",
            },
            [
                {
                    "name": "The Little Mermaid",
                    "object_type": "eidolon",
                    "slug": "the-little-mermaid",
                    "section": "Stats",
                    "score": 200.0,
                    "coverage_complete": True,
                    "content": "secret evidence",
                }
            ],
        )
    trace.record_response("vi", "vi", "secret final answer", 0)
    trace.finalize()

    record = _read_trace(path)
    assert record["retrieval"]["evidence"][0]["name"] == "The Little Mermaid"
    assert record["retrieval"]["evidence"][0]["section"] == "Stats"
    assert "content" not in record["retrieval"]["evidence"][0]
    assert "answer" not in record["response"]
    assert record["response"]["answer_chars"] == len("secret final answer")
    assert record["graph"]["nodes"][0]["name"] == "retrieve"
    assert record["graph"]["nodes"][0]["status"] == "ok"


def test_trace_content_is_opt_in(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(path))
    monkeypatch.setenv("KAMI_CHAT_TRACE_INCLUDE_CONTENT", "1")
    trace = create_chat_trace("s", "c", "gpt", "test-model")
    trace.record_retrieval(
        {},
        [
            {
                "name": "Nike",
                "section": "Ability",
                "content": "Restores HP",
            }
        ],
    )
    trace.record_response("en", "en", "Nike restores HP", 0)
    trace.finalize()

    record = _read_trace(path)
    assert record["retrieval"]["evidence"][0]["content"] == "Restores HP"
    assert record["response"]["answer"] == "Nike restores HP"


def test_disabled_trace_uses_noop_collector(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_ENABLED", "0")
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(path))
    trace = create_chat_trace("s", "c", "gpt", "test-model")

    with trace.node("plan"):
        pass
    trace.finalize()

    assert not path.exists()


def test_trace_aggregates_only_reported_tokens(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(path))
    trace = create_chat_trace("s", "c", "deepseek", "deepseek-chat")
    trace.record_model_call(
        {
            "step": "planner",
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "usage_available": True,
            "duration_ms": 5.0,
        }
    )
    trace.record_model_call(
        {
            "step": "answer",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "usage_available": False,
            "duration_ms": 8.0,
        }
    )
    trace.finalize()

    record = _read_trace(path)
    assert record["totals"]["reported_input_tokens"] == 10
    assert record["totals"]["reported_output_tokens"] == 2
    assert record["totals"]["reported_total_tokens"] == 12
    assert record["totals"]["calls_with_usage"] == 1
    assert record["totals"]["model_call_count"] == 2


def test_trace_records_errors_without_raw_traceback(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(path))
    trace = create_chat_trace("s", "c", "gpt", "test-model")
    trace.record_error(RuntimeError("provider failed"))
    trace.finalize()

    record = _read_trace(path)
    assert record["status"] == "error"
    assert record["error"] == {
        "type": "RuntimeError",
        "message": "provider failed",
    }
    assert "traceback" not in record["error"]


def test_trace_writer_failure_is_non_fatal(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    trace = create_chat_trace("s", "c", "gpt", "test-model")

    def fail_write(_record):
        raise OSError("disk full")

    monkeypatch.setattr(trace, "_write_record", fail_write)
    with caplog.at_level(logging.ERROR, logger="kami.agent.tracing"):
        trace.finalize()

    assert "disk full" in caplog.text
