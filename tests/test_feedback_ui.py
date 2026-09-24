"""Exercise the actual Gradio callbacks with per-client state and no model downloads."""

import importlib.util

import pytest
from PIL import Image

from adaptvlm.config import ModelConfig
from adaptvlm.feedback.store import FeedbackStore
from adaptvlm.inference.engine import InferenceEngine


@pytest.mark.skipif(importlib.util.find_spec("gradio") is None, reason="Install app extra")
def test_ui_session_initialization_failed_request_and_votes(tmp_path):
    from adaptvlm.dashboard.app import build_app

    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    engine = InferenceEngine(ModelConfig(backend="mock"))
    app = build_app(engine, store, engine)
    try:
        callbacks = {fn.api_name: fn.fn for fn in app.fns.values()}
        # API clients do not run page-load events. A None session must initialize here.
        first = callbacks["answer"](Image.new("RGB", (4, 4)), "Color?", "train", False, None)
        assert first[0] and first[-1]
        assert store.get_request(first[0])["session_id"] == first[-1]
        assert "saved" in callbacks["correct"](first[0], "Incorrect", "blue")
        assert "already submitted" in callbacks["correct"](first[0], "Correct", "")
        failed = callbacks["answer"](None, "Color?", "train", False, first[-1])
        assert failed[0] is None and failed[3] == ""
        assert "Unknown" in callbacks["correct"](failed[0], "Correct", "")
        second = callbacks["compare"](Image.new("RGB", (4, 4)), "Color?", "train", True, first[-1])
        assert second[-1] == first[-1]
        assert "Identical" in callbacks["prefer"](second[0], "a")
        assert "saved" in callbacks["prefer"](second[0], "neither")
        other_client = callbacks["answer"](Image.new("RGB", (4, 4)), "Color?", "train", False, None)
        assert other_client[-1] != first[-1]
        summary, recent = callbacks["review"]()
        assert summary["feedback"] == 2 and len(recent) == 2
    finally:
        app.close()
