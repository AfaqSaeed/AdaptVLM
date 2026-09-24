"""Offline tests exercise inference, leakage prevention, scoring and reports."""

import json
from pathlib import Path

import pytest
from PIL import Image

from adaptvlm.cli import main
from adaptvlm.config import ModelConfig, load_config
from adaptvlm.datasets.manifest import load_manifest
from adaptvlm.evaluation.runner import evaluate, normalize, write_report
from adaptvlm.inference.engine import InferenceEngine, Prediction

FIXTURE = Path(__file__).resolve().parents[1] / "examples/synthetic/manifest.json"


def test_config_rejects_invalid_values():
    for kwargs in (
        {"backend": "unknown"},
        {"max_new_tokens": 0},
        {"device": "cpu", "dtype": "float16"},
        {"family": "unknown"},
    ):
        with pytest.raises(ValueError):
            ModelConfig(**kwargs)
    assert load_config(FIXTURE.parents[2] / "configs/mock.yaml").backend == "mock"


def test_reuse_and_validation():
    class Backend:
        calls = 0

        def generate(self, image, question, config):
            self.calls += 1
            assert image.mode == "RGB"
            return "red"

    backend = Backend()
    engine = InferenceEngine(ModelConfig(backend="mock"), backend)
    for _ in range(2):
        assert engine.predict(Image.new("L", (2, 2)), "color?").answer == "red"
    assert backend.calls == 2
    with pytest.raises(ValueError):
        engine.predict(None, "question")
    with pytest.raises(ValueError):
        engine.predict(Image.new("RGB", (2, 2)), " ")


def test_scoring_and_reproducibility(tmp_path):
    engine = InferenceEngine(ModelConfig(backend="mock"))
    first, second = evaluate(engine, FIXTURE), evaluate(engine, FIXTURE)
    assert first["software_test_only"]
    assert first["metrics"]["normalized_match"] == 0.5
    assert first["metrics"]["count"] == 2
    assert first["dataset"] == second["dataset"]
    assert first["metrics"]["peak_gpu_memory_bytes"] is None
    write_report(first, tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["config"]["seed"] == 42
    assert normalize("  RED! ") == "red"
    assert normalize("A red shape") != "red"


def test_failures_count_as_incorrect():
    engine = InferenceEngine(ModelConfig(backend="mock"))
    engine.predict = lambda *args: Prediction("", 0.1, None, "simulated failure")
    result = evaluate(engine, FIXTURE)
    assert result["metrics"]["failures"] == 2
    assert result["metrics"]["normalized_match"] == 0


def test_leakage_and_traversal(tmp_path):
    Image.new("RGB", (2, 2), "red").save(tmp_path / "a.png")
    row = {
        "id": "1",
        "image": "a.png",
        "question": "Color?",
        "category": "color",
        "answers": ["red"],
        "split": "train",
    }
    data = {
        "schema_version": 1,
        "synthetic": True,
        "name": "bad",
        "examples": [row, row | {"id": "2", "split": "test"}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path)
    data["examples"] = [row | {"image": "../escape.png"}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(path)


def test_cli(tmp_path):
    root = FIXTURE.parents[2]
    assert (
        main(
            [
                "evaluate",
                "--config",
                str(root / "configs/mock.yaml"),
                "--evaluation-config",
                str(root / "configs/evaluation.yaml"),
                "--manifest",
                str(FIXTURE),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )


def test_unexpected_backend_error_propagates():
    class Broken:
        def generate(self, *args):
            raise RuntimeError("bad tensor")

    with pytest.raises(RuntimeError, match="bad tensor"):
        InferenceEngine(ModelConfig(backend="mock"), Broken()).predict(
            Image.new("RGB", (2, 2)), "question"
        )


def test_app_build(tmp_path):
    import importlib.util

    if importlib.util.find_spec("gradio") is None:
        pytest.skip("Install the app extra")
    from adaptvlm.dashboard.app import build_app

    from adaptvlm.feedback.store import FeedbackStore

    app = build_app(
        InferenceEngine(ModelConfig(backend="mock")), FeedbackStore(tmp_path / "feedback.sqlite3")
    )
    assert app is not None
    app.close()
