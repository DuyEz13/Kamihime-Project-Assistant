from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def database_path(legacy_json_path: Path) -> Path:
    return legacy_json_path.with_name("chat_memory.sqlite3")


def _connect(legacy_json_path: Path) -> sqlite3.Connection:
    path = database_path(legacy_json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL DEFAULT 'local-user',
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            state_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            sources_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, id);
        CREATE TABLE IF NOT EXISTS client_memory (
            client_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        """
    )
    _migrate_json(connection, legacy_json_path)
    return connection


def _migrate_json(connection: sqlite3.Connection, path: Path) -> None:
    count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    if count or not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(sessions, dict):
        return
    for session_id, session in sessions.items():
        if not isinstance(session, dict):
            continue
        created = str(session.get("created_at") or now_iso())
        updated = str(session.get("updated_at") or created)
        connection.execute(
            """INSERT OR IGNORE INTO sessions
            (session_id, title, summary, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(session_id),
                str(session.get("title") or ""),
                str(session.get("summary") or ""),
                json.dumps(session.get("state") or {}, ensure_ascii=False),
                created,
                updated,
            ),
        )
        for message in session.get("messages") or []:
            if not isinstance(message, dict):
                continue
            connection.execute(
                """INSERT INTO messages
                (session_id, role, content, created_at, provider, model, sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(session_id),
                    str(message.get("role") or ""),
                    str(message.get("content") or ""),
                    str(message.get("created_at") or created),
                    message.get("provider"),
                    message.get("model"),
                    json.dumps(message.get("sources") or [], ensure_ascii=False),
                ),
            )
    connection.commit()


def _message(row: sqlite3.Row) -> dict[str, Any]:
    value: dict[str, Any] = {
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
    if row["provider"]:
        value["provider"] = row["provider"]
    if row["model"]:
        value["model"] = row["model"]
    try:
        sources = json.loads(row["sources_json"] or "[]")
    except json.JSONDecodeError:
        sources = []
    if sources:
        value["sources"] = sources
    return value


def _mirror_json(connection: sqlite3.Connection, path: Path) -> None:
    sessions: dict[str, Any] = {}
    for row in connection.execute("SELECT * FROM sessions ORDER BY updated_at DESC"):
        messages = [
            _message(item)
            for item in connection.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id",
                (row["session_id"],),
            )
        ]
        try:
            state = json.loads(row["state_json"] or "{}")
        except json.JSONDecodeError:
            state = {}
        sessions[row["session_id"]] = {
            "title": row["title"],
            "summary": row["summary"],
            "state": state,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": messages,
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def get_messages(path: Path, session_id: str, limit: int | None = None) -> list[dict]:
    with _LOCK, _connect(path) as connection:
        if limit is None:
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """SELECT * FROM (
                SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?
                ) ORDER BY id""",
                (session_id, limit),
            ).fetchall()
        return [_message(row) for row in rows]


def get_state(path: Path, session_id: str) -> dict[str, Any]:
    with _LOCK, _connect(path) as connection:
        row = connection.execute(
            "SELECT state_json, summary FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            return {}
        try:
            state = json.loads(row["state_json"] or "{}")
        except json.JSONDecodeError:
            state = {}
        if row["summary"]:
            state["summary"] = row["summary"]
        return state


def get_client_state(path: Path, client_id: str) -> dict[str, Any]:
    if not client_id:
        return {}
    with _LOCK, _connect(path) as connection:
        row = connection.execute(
            "SELECT state_json FROM client_memory WHERE client_id=?",
            (client_id,),
        ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(row["state_json"] or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


def append_exchange(
    path: Path,
    session_id: str,
    client_id: str,
    user_message: dict,
    assistant_message: dict,
    state: dict[str, Any],
) -> None:
    created = str(user_message.get("created_at") or now_iso())
    updated = str(assistant_message.get("created_at") or now_iso())
    title_text = str(user_message.get("content") or "").strip()
    title = title_text[:48] + ("..." if len(title_text) > 48 else "")
    with _LOCK, _connect(path) as connection:
        summary = str(state.get("summary") or "")
        connection.execute(
            """INSERT INTO sessions
            (session_id, client_id, title, summary, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                client_id=excluded.client_id,
                updated_at=excluded.updated_at,
                summary=excluded.summary,
                state_json=excluded.state_json""",
            (
                session_id,
                client_id or "local-user",
                title,
                summary,
                json.dumps(state, ensure_ascii=False),
                created,
                updated,
            ),
        )
        long_term = state.get("long_term")
        if isinstance(long_term, dict):
            connection.execute(
                """INSERT INTO client_memory (client_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(client_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at""",
                (
                    client_id or "local-user",
                    json.dumps(long_term, ensure_ascii=False),
                    updated,
                ),
            )
        for message in (user_message, assistant_message):
            connection.execute(
                """INSERT INTO messages
                (session_id, role, content, created_at, provider, model, sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    message.get("role"),
                    message.get("content"),
                    message.get("created_at") or now_iso(),
                    message.get("provider"),
                    message.get("model"),
                    json.dumps(message.get("sources") or [], ensure_ascii=False),
                ),
            )
        connection.commit()
        _mirror_json(connection, path)


def list_sessions(path: Path) -> list[dict[str, Any]]:
    with _LOCK, _connect(path) as connection:
        rows = connection.execute(
            """SELECT s.*, COUNT(m.id) AS message_count
            FROM sessions s LEFT JOIN messages m ON m.session_id=s.session_id
            GROUP BY s.session_id ORDER BY s.updated_at DESC"""
        ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"] or "New chat",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]


def get_session(path: Path, session_id: str) -> dict[str, Any]:
    with _LOCK, _connect(path) as connection:
        row = connection.execute(
            "SELECT title FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
    return {
        "session_id": session_id,
        "title": row["title"] if row else "New chat",
        "messages": get_messages(path, session_id),
    }


def delete_session(path: Path, session_id: str) -> bool:
    with _LOCK, _connect(path) as connection:
        cursor = connection.execute(
            "DELETE FROM sessions WHERE session_id=?",
            (session_id,),
        )
        connection.commit()
        if cursor.rowcount:
            _mirror_json(connection, path)
        return bool(cursor.rowcount)
