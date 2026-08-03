import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from kami import chatbot


def _characters():
    return [
        {
            "slug": "nike",
            "name": "Nike",
            "element": "Water",
            "release_date": "2016/03/30",
            "acquisition_method": "Premium Gacha",
            "display_info": {
                "Rarity": "SSR",
                "Type": "Healer",
            },
            "skill_sections": [
                {
                    "type": "Ability",
                    "rows": [
                        {
                            "name": "Healing Stream",
                            "requirements": "-",
                            "interval": "6T",
                            "duration": "-",
                            "effect": "Restores HP to all allies.",
                        }
                    ],
                }
            ],
            "flavor": "A water healer.",
        }
    ]


def _characters_with_gacha_noise():
    return [
        {
            "slug": "phoenix-cry",
            "name": "Phoenix Cry",
            "element": "Fire",
            "release_date": "2024/05/01",
            "acquisition_method": "Limited-Time Gacha (Collaboration)",
            "display_info": {"Rarity": "SSR"},
            "skill_sections": [],
            "flavor": "",
        },
        {
            "slug": "ra",
            "name": "Ra",
            "element": "Light",
            "release_date": "2020/01/01",
            "acquisition_method": "Premium Gacha",
            "display_info": {"Rarity": "SSR"},
            "skill_sections": [],
            "flavor": "A gacha character with gacha-related text.",
        },
    ]


def test_retrieve_characters_scores_name_and_skill(monkeypatch):
    monkeypatch.setattr(chatbot, "load_characters", _characters)

    results = chatbot.retrieve_characters("What does Nike Healing Stream do?")

    assert results
    assert results[0][0]["name"] == "Nike"


def test_off_topic_question_is_refused_without_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", tmp_path / "chat_sessions.json")
    monkeypatch.setattr(chatbot, "load_characters", _characters)

    def fail_model(*_args, **_kwargs):
        raise AssertionError("model should not be called for off-topic questions")

    monkeypatch.setattr(chatbot, "_call_model", fail_model)

    response = chatbot.answer_chat("What is the capital of France?", provider="gpt")

    assert "Kamihime Project characters" in response["answer"]
    assert response["session_id"]


def test_domain_question_calls_model_and_persists_memory(tmp_path, monkeypatch):
    memory_path = tmp_path / "chat_sessions.json"
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", memory_path)
    monkeypatch.setattr(chatbot, "load_characters", _characters)

    calls = []

    def fake_model(provider, model, session_id, message, context, **_kwargs):
        calls.append((provider, model, session_id, message, context))
        return "Nike is a Water healer. Healing Stream restores HP to all allies."

    monkeypatch.setattr(chatbot, "_call_model", fake_model)

    response = chatbot.answer_chat("Tell me about Nike skills", provider="gpt")

    assert response["answer"].startswith("Nike is a Water healer")
    assert response["sources"][0]["name"] == "Nike"
    assert calls
    assert "Healing Stream" in calls[0][4]

    stored = json.loads(memory_path.read_text(encoding="utf-8"))
    messages = stored["sessions"][response["session_id"]]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert chatbot.list_sessions()[0]["title"] == "Tell me about Nike skills"


def test_vietnamese_question_persists_language_and_guards_model_prompt(
    tmp_path,
    monkeypatch,
):
    memory_path = tmp_path / "chat_sessions.json"
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", memory_path)
    monkeypatch.setattr(chatbot, "load_characters", _characters)
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    calls = []

    def fake_model(_provider, _model, _session_id, message, _context, **_kwargs):
        calls.append(message)
        return "Nike là Kamihime hệ Thủy."

    monkeypatch.setattr(chatbot, "_call_model", fake_model)

    response = chatbot.answer_chat(
        "Hãy cho tôi biết thông tin về Kamihime Nike",
        provider="gpt",
        client_id="vi-client",
    )

    assert response["answer"].startswith("Nike là")
    assert "Required response language: Vietnamese (vi)" in calls[0]
    stored = json.loads(memory_path.read_text(encoding="utf-8"))
    state = stored["sessions"][response["session_id"]]["state"]
    assert state["long_term"]["language"] == "vi"


def test_delete_session_removes_history(tmp_path, monkeypatch):
    memory_path = tmp_path / "chat_sessions.json"
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", memory_path)
    monkeypatch.setattr(chatbot, "load_characters", _characters)
    monkeypatch.setattr(
        chatbot,
        "_call_model",
        lambda *_args, **_kwargs: "Answer.",
    )

    response = chatbot.answer_chat("Tell me about Nike", provider="gpt")

    assert chatbot.list_sessions()
    assert chatbot.delete_session(response["session_id"]) is True
    assert chatbot.list_sessions() == []
    assert chatbot.delete_session(response["session_id"]) is False


def test_followup_question_uses_session_memory_for_rag(tmp_path, monkeypatch):
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", tmp_path / "chat_sessions.json")
    monkeypatch.setattr(chatbot, "load_characters", _characters)

    contexts = []

    def fake_model(_provider, _model, _session_id, _message, context, **_kwargs):
        contexts.append(context)
        return "Answer from retrieved context."

    monkeypatch.setattr(chatbot, "_call_model", fake_model)

    first = chatbot.answer_chat("Tell me about Nike", provider="gpt")
    second = chatbot.answer_chat(
        "What does she do?",
        session_id=first["session_id"],
        provider="gpt",
    )

    assert second["session_id"] == first["session_id"]
    assert "Healing Stream" in contexts[-1]


def test_followup_prioritizes_previous_answer_sources_over_generic_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", tmp_path / "chat_sessions.json")
    monkeypatch.setattr(chatbot, "load_characters", _characters_with_gacha_noise)

    contexts = []

    def fake_model(_provider, _model, _session_id, _message, context, **_kwargs):
        contexts.append(context)
        return "Answer from retrieved context."

    monkeypatch.setattr(chatbot, "_call_model", fake_model)

    first = chatbot.answer_chat("How do I get Phoenix Cry?", provider="gpt")
    second = chatbot.answer_chat(
        "When does that gacha happen?",
        session_id=first["session_id"],
        provider="gpt",
    )

    assert second["sources"][0]["name"] == "Phoenix Cry"
    assert "Name: Phoenix Cry" in contexts[-1]
    assert all(source["name"] != "Ra" for source in second["sources"])


def test_chat_provider_retries_transient_http_status(monkeypatch):
    monkeypatch.setenv("KAMI_CHAT_HTTP_RETRIES", "2")
    monkeypatch.setattr(chatbot.time, "sleep", lambda _seconds: None)
    request = httpx.Request("POST", "https://example.test/chat")
    responses = [
        httpx.Response(503, json={"error": {"message": "high demand"}}, request=request),
        httpx.Response(200, json={"ok": True}, request=request),
    ]

    class Client:
        def post(self, *_args, **_kwargs):
            return responses.pop(0)

    body = chatbot._post_json_with_retry(Client(), "https://example.test/chat")

    assert body == {"ok": True}
    assert responses == []


def test_chat_provider_reports_transient_error_after_retries(monkeypatch):
    monkeypatch.setenv("KAMI_CHAT_HTTP_RETRIES", "1")
    monkeypatch.setattr(chatbot.time, "sleep", lambda _seconds: None)
    request = httpx.Request("POST", "https://example.test/chat")

    class Client:
        def post(self, *_args, **_kwargs):
            return httpx.Response(
                503,
                json={"error": {"message": "high demand"}},
                request=request,
            )

    try:
        chatbot._post_json_with_retry(Client(), "https://example.test/chat")
    except httpx.HTTPStatusError as exc:
        message = chatbot._provider_error_message(exc)
    else:
        raise AssertionError("expected HTTPStatusError")

    assert "temporarily unavailable (503)" in message
    assert "high demand" in message


def test_answer_chat_writes_graph_and_retrieval_trace(tmp_path, monkeypatch):
    memory_path = tmp_path / "chat_sessions.json"
    trace_path = tmp_path / "chat_traces.jsonl"
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", memory_path)
    monkeypatch.setattr(chatbot, "load_characters", _characters)
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    monkeypatch.setenv("KAMI_CHAT_TRACE_ENABLED", "1")
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(trace_path))
    monkeypatch.setattr(
        chatbot,
        "_call_model",
        lambda *_args, **_kwargs: "Nike is a Water healer.",
    )

    response = chatbot.answer_chat("Tell me about Nike", provider="gpt")

    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert response["trace_id"] == record["trace_id"]
    assert record["status"] == "ok"
    assert record["plan"]["target_types"] == ["kamihime"]
    assert record["retrieval"]["reason_code"] == "selected_relevant_records"
    assert record["retrieval"]["evidence"][0]["name"] == "Nike"
    assert [node["name"] for node in record["graph"]["nodes"]] == [
        "plan",
        "retrieve",
        "answer",
    ]


def test_answer_chat_traces_language_retry_as_second_model_call(
    tmp_path,
    monkeypatch,
):
    trace_path = tmp_path / "chat_traces.jsonl"
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", tmp_path / "sessions.json")
    monkeypatch.setattr(chatbot, "load_characters", _characters)
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    monkeypatch.setenv("KAMI_CHAT_TRACE_ENABLED", "1")
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(trace_path))
    responses = [
        AIMessage(
            content="There are six members in this series.",
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 25,
            },
        ),
        AIMessage(
            content="Nike là Kamihime hệ Thủy.",
            usage_metadata={
                "input_tokens": 30,
                "output_tokens": 7,
                "total_tokens": 37,
            },
        ),
    ]

    class FakeModel:
        def invoke(self, _messages):
            return responses.pop(0)

    monkeypatch.setattr(
        "kami.agent.providers.create_chat_model",
        lambda *_args, **_kwargs: FakeModel(),
    )

    chatbot.answer_chat(
        "Hãy cho tôi biết thông tin về Kamihime Nike",
        provider="gpt",
    )

    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert [call["step"] for call in record["model_calls"]] == [
        "answer",
        "answer_language_retry",
    ]
    assert record["totals"]["reported_total_tokens"] == 62
    assert record["response"]["required_language"] == "vi"
    assert record["response"]["actual_language"] == "vi"
    assert record["response"]["language_retry_count"] == 1


def test_answer_chat_finalizes_trace_when_model_fails(tmp_path, monkeypatch):
    trace_path = tmp_path / "chat_traces.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_ENABLED", "1")
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", tmp_path / "sessions.json")
    monkeypatch.setattr(chatbot, "load_characters", _characters)

    def fail_model(*_args, **_kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(chatbot, "_call_model", fail_model)

    with pytest.raises(RuntimeError, match="provider failed"):
        chatbot.answer_chat("Tell me about Nike", provider="gpt")

    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "error"
    assert record["error"]["type"] == "RuntimeError"
    assert [node["name"] for node in record["graph"]["nodes"]] == [
        "plan",
        "retrieve",
        "answer",
    ]
    assert record["graph"]["nodes"][-1]["status"] == "error"
