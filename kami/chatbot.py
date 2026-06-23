from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .data_store import load_characters
from .paths import DATA_DIR


CHAT_MEMORY_PATH = DATA_DIR / "chat_sessions.json"
DOMAIN_KEYWORDS = {
    "kamihime",
    "kami",
    "character",
    "characters",
    "skill",
    "skills",
    "ability",
    "abilities",
    "burst",
    "assist",
    "element",
    "fire",
    "water",
    "wind",
    "thunder",
    "light",
    "dark",
    "rarity",
    "weapon",
    "hp",
    "attack",
    "release",
    "acquisition",
    "gacha",
    "buff",
    "debuff",
    "damage",
    "healer",
    "defense",
    "balance",
    "tricky",
}
FOLLOWUP_KEYWORDS = {
    "it",
    "its",
    "she",
    "her",
    "hers",
    "he",
    "him",
    "his",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
}
PROVIDER_LABELS = {
    "gpt": "GPT",
    "gemini": "Gemini",
}
TRANSIENT_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MEMORY_LOCK = threading.Lock()


@dataclass(frozen=True)
class ChatModel:
    provider: str
    label: str
    model: str
    configured: bool


@dataclass(frozen=True)
class RagSource:
    name: str
    element: str
    slug: str
    score: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(destination)


def _read_memory() -> dict[str, Any]:
    if not CHAT_MEMORY_PATH.exists():
        return {"sessions": {}}
    try:
        with CHAT_MEMORY_PATH.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"sessions": {}}
    if not isinstance(value, dict):
        return {"sessions": {}}
    sessions = value.get("sessions")
    if not isinstance(sessions, dict):
        value["sessions"] = {}
    return value


def _save_memory(memory: dict[str, Any]) -> None:
    _atomic_write_json(memory, CHAT_MEMORY_PATH)


def _tokenize(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'+.-]*", value.casefold())
        if len(token) > 1
    }


def _skill_name(row: dict[str, Any]) -> str:
    for key in ("name", "Burst", "Ability", "Assist", "burst", "ability", "assist"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "-"


def _character_text(character: dict[str, Any]) -> str:
    parts = [
        str(character.get("name") or ""),
        str(character.get("element") or ""),
        str(character.get("release_date") or ""),
        str(character.get("acquisition_method") or ""),
        str(character.get("flavor") or ""),
    ]

    display_info = character.get("display_info")
    if isinstance(display_info, dict):
        for key, value in display_info.items():
            parts.append(f"{key}: {value}")

    for section in character.get("skill_sections") or []:
        if not isinstance(section, dict):
            continue
        parts.append(str(section.get("type") or "Skill"))
        for row in section.get("rows") or []:
            if not isinstance(row, dict):
                continue
            parts.extend(
                [
                    _skill_name(row),
                    str(row.get("requirements") or ""),
                    str(row.get("interval") or ""),
                    str(row.get("duration") or ""),
                    str(row.get("effect") or ""),
                ]
            )
    return "\n".join(part for part in parts if part)


def _character_context(character: dict[str, Any]) -> str:
    lines = [
        f"Name: {character.get('name')}",
        f"Element: {character.get('element')}",
        f"Release Date: {character.get('release_date')}",
        f"Acquisition Method: {character.get('acquisition_method')}",
    ]
    display_info = character.get("display_info")
    if isinstance(display_info, dict):
        for key, value in display_info.items():
            if value not in (None, ""):
                lines.append(f"{key}: {value}")
    flavor = character.get("flavor")
    if flavor:
        lines.append(f"Flavor: {flavor}")

    for section in character.get("skill_sections") or []:
        if not isinstance(section, dict):
            continue
        section_type = section.get("type") or "Skill"
        for row in section.get("rows") or []:
            if not isinstance(row, dict):
                continue
            details = [
                f"{section_type}: {_skill_name(row)}",
                f"Acquisition: {row.get('requirements') or '-'}",
            ]
            if section_type == "Ability":
                details.extend(
                    [
                        f"Interval: {row.get('interval') or '-'}",
                        f"Duration: {row.get('duration') or '-'}",
                    ]
                )
            details.append(f"Effect: {row.get('effect') or '-'}")
            lines.append(" | ".join(details))
    return "\n".join(lines)


def _rag_top_k() -> int:
    try:
        return max(1, int(os.getenv("KAMI_RAG_TOP_K", "5")))
    except ValueError:
        return 5


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _sleep_before_retry(response: httpx.Response | None, attempt: int) -> None:
    retry_after = _retry_after_seconds(response) if response is not None else None
    if retry_after is not None:
        delay = retry_after
    else:
        base = _env_float("KAMI_CHAT_RETRY_BACKOFF_BASE", 1.5, 0.1)
        max_delay = _env_float("KAMI_CHAT_RETRY_BACKOFF_MAX", 12.0, 0.1)
        jitter = _env_float("KAMI_CHAT_RETRY_JITTER", 0.25, 0.0)
        delay = min(max_delay, base * (2 ** attempt))
        delay += random.uniform(0, jitter)
    time.sleep(delay)


def _post_json_with_retry(
    client: httpx.Client,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    retries = _env_int("KAMI_CHAT_HTTP_RETRIES", 3, 0)
    last_request_error: httpx.RequestError | None = None

    for attempt in range(retries + 1):
        try:
            response = client.post(url, **kwargs)
        except httpx.RequestError as exc:
            last_request_error = exc
            if attempt >= retries:
                raise
            _sleep_before_retry(None, attempt)
            continue

        if response.status_code not in TRANSIENT_HTTP_STATUS:
            response.raise_for_status()
            return response.json()

        if attempt >= retries:
            response.raise_for_status()
        _sleep_before_retry(response, attempt)

    if last_request_error:
        raise last_request_error
    raise RuntimeError("Chat provider request failed")


def retrieve_characters(query: str, limit: int | None = None) -> list[tuple[dict[str, Any], float]]:
    query_tokens = _tokenize(query)
    query_folded = query.casefold()
    if not query_tokens and not query_folded.strip():
        return []

    scored: list[tuple[dict[str, Any], float]] = []
    for character in load_characters():
        name = str(character.get("name") or "")
        element = str(character.get("element") or "")
        document = _character_text(character)
        document_tokens = _tokenize(document)
        overlap = len(query_tokens & document_tokens)
        score = float(overlap)

        name_folded = name.casefold()
        if name_folded and name_folded in query_folded:
            score += 12.0
        elif any(token in _tokenize(name) for token in query_tokens):
            score += 5.0

        if element and element.casefold() in query_tokens:
            score += 3.0
        if score > 0:
            scored.append((character, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: limit or _rag_top_k()]


def is_domain_question(message: str, sources: list[tuple[dict[str, Any], float]]) -> bool:
    tokens = _tokenize(message)
    if tokens & DOMAIN_KEYWORDS:
        return True
    return bool(sources and sources[0][1] >= 4)


def available_chat_models() -> list[ChatModel]:
    gpt_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    gemini_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    return [
        ChatModel(
            provider="gpt",
            label=PROVIDER_LABELS["gpt"],
            model=gpt_model,
            configured=bool(os.getenv("OPENAI_API_KEY")),
        ),
        ChatModel(
            provider="gemini",
            label=PROVIDER_LABELS["gemini"],
            model=gemini_model,
            configured=bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        ),
    ]


def _normalize_provider(provider: str | None) -> str:
    value = (provider or os.getenv("KAMI_CHAT_PROVIDER") or "gpt").strip().lower()
    if value in {"openai", "chatgpt"}:
        value = "gpt"
    if value not in PROVIDER_LABELS:
        valid = ", ".join(PROVIDER_LABELS)
        raise ValueError(f"Unknown chat provider '{provider}'. Valid providers: {valid}")
    return value


def _system_prompt() -> str:
    return (
        "You are KamiWiki Assistant, a factual assistant for Kamihime Project "
        "character information. Answer only questions about Kamihime Project "
        "characters, their elements, basic data, skills, effects, release data, "
        "and acquisition data. Use only the retrieved database context and the "
        "current conversation memory. If the user asks about anything unrelated, "
        "politely refuse in one short sentence. If the retrieved context does "
        "not contain the answer, say that the local database does not have enough "
        "information. Do not invent data."
    )


def _session_messages(session_id: str) -> list[dict[str, Any]]:
    memory = _read_memory()
    session = memory.get("sessions", {}).get(session_id)
    if not isinstance(session, dict):
        return []
    messages = session.get("messages")
    return messages if isinstance(messages, list) else []


def _session_title(session: dict[str, Any], fallback: str = "New chat") -> str:
    title = str(session.get("title") or "").strip()
    if title:
        return title
    messages = session.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                return content[:48] + ("..." if len(content) > 48 else "")
    return fallback


def list_sessions() -> list[dict[str, Any]]:
    with _MEMORY_LOCK:
        memory = _read_memory()
    sessions = memory.get("sessions", {})
    if not isinstance(sessions, dict):
        return []

    result = []
    for session_id, session in sessions.items():
        if not isinstance(session, dict):
            continue
        messages = session.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0
        result.append(
            {
                "session_id": str(session_id),
                "title": _session_title(session),
                "created_at": session.get("created_at") or "",
                "updated_at": session.get("updated_at")
                or session.get("created_at")
                or "",
                "message_count": message_count,
            }
        )

    result.sort(key=lambda item: item["updated_at"], reverse=True)
    return result


def get_session(session_id: str) -> dict[str, Any]:
    with _MEMORY_LOCK:
        memory = _read_memory()
    session = memory.get("sessions", {}).get(session_id, {})
    if not isinstance(session, dict):
        session = {}
    return {
        "session_id": session_id,
        "title": _session_title(session),
        "messages": session.get("messages") if isinstance(session.get("messages"), list) else [],
    }


def delete_session(session_id: str) -> bool:
    with _MEMORY_LOCK:
        memory = _read_memory()
        sessions = memory.get("sessions")
        if not isinstance(sessions, dict) or session_id not in sessions:
            return False
        del sessions[session_id]
        _save_memory(memory)
    return True


def _append_session_messages(
    session_id: str,
    new_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    limit = max(10, int(os.getenv("KAMI_CHAT_HISTORY_LIMIT", "100")))
    with _MEMORY_LOCK:
        memory = _read_memory()
        sessions = memory.setdefault("sessions", {})
        session = sessions.setdefault(
            session_id,
            {
                "created_at": _now_iso(),
                "messages": [],
            },
        )
        session["updated_at"] = _now_iso()
        if not session.get("title"):
            first_user = next(
                (
                    str(message.get("content") or "").strip()
                    for message in new_messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                "",
            )
            if first_user:
                session["title"] = first_user[:48] + ("..." if len(first_user) > 48 else "")
        messages = session.setdefault("messages", [])
        if not isinstance(messages, list):
            messages = []
            session["messages"] = messages
        messages.extend(new_messages)
        if len(messages) > limit:
            del messages[:-limit]
        _save_memory(memory)
        return list(messages)


def _memory_for_prompt(session_id: str) -> list[dict[str, str]]:
    limit = max(2, int(os.getenv("KAMI_CHAT_CONTEXT_MESSAGES", "16")))
    messages = _session_messages(session_id)[-limit:]
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def _build_context(sources: list[tuple[dict[str, Any], float]]) -> str:
    chunks = []
    for index, (character, _score) in enumerate(sources, start=1):
        chunks.append(f"[Character {index}]\n{_character_context(character)}")
    return "\n\n".join(chunks)


def _openai_answer(
    model: str,
    session_id: str,
    message: str,
    context: str,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    messages: list[dict[str, str]] = [{"role": "system", "content": _system_prompt()}]
    messages.extend(_memory_for_prompt(session_id))
    messages.append(
        {
            "role": "user",
            "content": (
                "Retrieved character database context:\n"
                f"{context or '(No relevant character context found.)'}\n\n"
                f"User question: {message}"
            ),
        }
    )
    with httpx.Client(timeout=60) as client:
        body = _post_json_with_retry(
            client,
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            },
        )
    return str(body["choices"][0]["message"]["content"]).strip()


def _gemini_answer(
    model: str,
    session_id: str,
    message: str,
    context: str,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")
    history = "\n".join(
        f"{item['role']}: {item['content']}" for item in _memory_for_prompt(session_id)
    )
    prompt = (
        f"Conversation memory:\n{history or '(empty)'}\n\n"
        "Retrieved character database context:\n"
        f"{context or '(No relevant character context found.)'}\n\n"
        f"User question: {message}"
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    with httpx.Client(timeout=60) as client:
        body = _post_json_with_retry(
            client,
            url,
            params={"key": api_key},
            json={
                "systemInstruction": {"parts": [{"text": _system_prompt()}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
        )
    parts = body["candidates"][0]["content"].get("parts", [])
    return "".join(str(part.get("text", "")) for part in parts).strip()


def _call_model(provider: str, model: str, session_id: str, message: str, context: str) -> str:
    if provider == "gpt":
        return _openai_answer(model, session_id, message, context)
    if provider == "gemini":
        return _gemini_answer(model, session_id, message, context)
    raise ValueError(f"Unsupported chat provider: {provider}")


def _provider_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        status_code = exc.response.status_code
        provider_message = ""
        try:
            body = exc.response.json()
        except json.JSONDecodeError:
            body = {}
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                provider_message = str(error.get("message") or "")
            elif isinstance(error, str):
                provider_message = error
        if status_code in TRANSIENT_HTTP_STATUS:
            detail = f" ({provider_message})" if provider_message else ""
            return (
                f"The chat provider is temporarily unavailable "
                f"({status_code}) after retrying. Please try again in a moment.{detail}"
            )
        return f"Chat provider returned HTTP {status_code}. {provider_message}".strip()
    if isinstance(exc, httpx.RequestError):
        return "Could not reach the chat provider after retrying. Please try again."
    return str(exc)


def answer_chat(
    message: str,
    session_id: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Message cannot be empty")

    selected_provider = _normalize_provider(provider)
    model_map = {model.provider: model for model in available_chat_models()}
    selected_model = model_map[selected_provider].model
    current_session_id = session_id or uuid.uuid4().hex
    sources = retrieve_characters(clean_message)
    message_tokens = _tokenize(clean_message)
    if (message_tokens & FOLLOWUP_KEYWORDS) and _session_messages(current_session_id):
        memory_text = "\n".join(
            item["content"] for item in _memory_for_prompt(current_session_id)
        )
        memory_sources = retrieve_characters(f"{clean_message}\n{memory_text}")
        known_slugs = {character.get("slug") for character, _score in sources}
        for character, score in memory_sources:
            if character.get("slug") not in known_slugs:
                sources.append((character, score))
                known_slugs.add(character.get("slug"))
        sources.sort(key=lambda item: item[1], reverse=True)
        sources = sources[:_rag_top_k()]
    source_payload = [
        RagSource(
            name=str(character.get("name") or ""),
            element=str(character.get("element") or ""),
            slug=str(character.get("slug") or ""),
            score=score,
        )
        for character, score in sources
    ]

    if not is_domain_question(clean_message, sources):
        answer = (
            "I can only answer questions about Kamihime Project characters in "
            "the local database."
        )
    else:
        context = _build_context(sources)
        try:
            answer = _call_model(
                selected_provider,
                selected_model,
                current_session_id,
                clean_message,
                context,
            )
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise RuntimeError(_provider_error_message(exc)) from exc

    _append_session_messages(
        current_session_id,
        [
            {
                "role": "user",
                "content": clean_message,
                "created_at": _now_iso(),
            },
            {
                "role": "assistant",
                "content": answer,
                "created_at": _now_iso(),
                "provider": selected_provider,
                "model": selected_model,
                "sources": [source.__dict__ for source in source_payload],
            },
        ],
    )
    return {
        "session_id": current_session_id,
        "answer": answer,
        "provider": selected_provider,
        "model": selected_model,
        "sources": [source.__dict__ for source in source_payload],
    }
