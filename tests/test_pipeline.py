import pytest

from kami import pipeline


class _FakeThread:
    created = []

    def __init__(self, *, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        self.__class__.created.append(self)

    def start(self):
        self.started = True


def _reset_pipeline_status():
    with pipeline._lock:
        pipeline._status["state"] = "idle"


def test_start_update_tracks_selected_object_type(monkeypatch):
    _reset_pipeline_status()
    _FakeThread.created.clear()
    monkeypatch.setattr(pipeline.threading, "Thread", _FakeThread)

    assert pipeline.start_update("latest", "weapons") is True

    status = pipeline.get_refresh_status()
    thread = _FakeThread.created[-1]
    assert status["object_type"] == "weapon"
    assert status["provider"] == "deepl"
    assert thread.args == ("latest", "weapon", "deepl")
    assert thread.daemon is True
    assert thread.started is True
    _reset_pipeline_status()


@pytest.mark.parametrize("mode", ["latest", "database"])
@pytest.mark.parametrize("object_type", ["kamihime", "eidolon", "weapon"])
def test_update_defaults_to_deepl_for_all_catalogs(
    monkeypatch,
    mode,
    object_type,
):
    _reset_pipeline_status()
    _FakeThread.created.clear()
    monkeypatch.setattr(pipeline.threading, "Thread", _FakeThread)

    assert pipeline.start_update(mode, object_type) is True

    thread = _FakeThread.created[-1]
    assert thread.args == (mode, object_type, "deepl")
    assert pipeline.get_refresh_status()["provider"] == "deepl"
    _reset_pipeline_status()


def test_start_translation_rejects_unknown_object_type():
    _reset_pipeline_status()

    with pytest.raises(ValueError, match="Unknown object type"):
        pipeline.start_translation("deepl", "summon")
