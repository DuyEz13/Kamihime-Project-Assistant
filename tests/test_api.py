import pytest
from fastapi import HTTPException

from app import main


def test_object_update_endpoint_normalizes_object_type(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "start_update",
        lambda mode, object_type, provider: (
            calls.append((mode, object_type, provider)) or True
        ),
    )
    monkeypatch.setattr(
        main,
        "get_refresh_status",
        lambda: {"state": "starting"},
    )

    assert main.update_object_data("weapons", "latest") == {
        "state": "starting"
    }
    assert calls == [("latest", "weapon", "deepl")]


def test_object_update_endpoint_accepts_selected_translation_provider(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        main,
        "start_update",
        lambda mode, object_type, provider: (
            calls.append((mode, object_type, provider)) or True
        ),
    )
    monkeypatch.setattr(
        main,
        "get_refresh_status",
        lambda: {"state": "starting"},
    )

    main.update_object_data("eidolon", "database", "google")

    assert calls == [("database", "eidolon", "google")]


def test_object_translation_endpoint_rejects_unknown_type():
    with pytest.raises(HTTPException) as exc_info:
        main.translate_object_database("summon", "deepl")

    assert exc_info.value.status_code == 404
