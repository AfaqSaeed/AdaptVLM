"""Feedback tests use fabricated candidates only; no research results are produced."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from adaptvlm.cli import main
from adaptvlm.config import ModelConfig
from adaptvlm.feedback.service import FeedbackService
from adaptvlm.feedback.store import FeedbackStore, image_digest
from adaptvlm.feedback.validation import validate_export
from adaptvlm.inference.engine import InferenceEngine, Prediction


def candidate(answer="red", backend="transformers"):
    """Fabricate provenance strictly for storage/eligibility tests."""
    config = ModelConfig(backend=backend, model_id="test-fixture")
    return {
        "answer": answer,
        "model_id": config.model_id,
        "resolved_revision": "fake-commit",
        "generation_parameters": config.to_dict(),
        "latency_seconds": 0.1,
        "peak_gpu_memory_bytes": None,
        "error": None,
    }


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(tmp_path / "feedback.sqlite3")


def request(store, candidates=None, split="train"):
    return store.create_request(
        Image.new("RGB", (8, 8), "red"), "Color?", candidates or [candidate()], split
    )


def test_persistence_correction_and_double_submit(store):
    saved = request(store)
    store.add_correction(saved["id"], False, "blue")
    reopened = FeedbackStore(store.database)
    rows = reopened.records()
    assert rows[0]["corrected_answer"] == "blue"
    assert rows[0]["candidates"][0]["answer"] == "red"
    assert rows[0]["training_eligible"]
    with pytest.raises(ValueError, match="already submitted"):
        reopened.add_correction(saved["id"], True)


@pytest.mark.parametrize("choice", ["a", "b", "both", "neither"])
@pytest.mark.parametrize("reverse", [False, True])
def test_preferences_preserve_display_order(store, monkeypatch, choice, reverse):
    monkeypatch.setattr(
        "secrets.SystemRandom.shuffle", lambda self, values: values.reverse() if reverse else None
    )
    saved = request(store, [candidate("red"), candidate("blue")])
    assert saved["display_order"] == ([1, 0] if reverse else [0, 1])
    store.add_preference(saved["id"], choice)
    row = store.records()[0]
    assert row["preference"] == choice
    if choice in {"a", "b"}:
        assert row["preferred_candidate_index"] == saved["display_order"][choice == "b"]
        assert row["training_eligible"]
    else:
        assert row["preferred_candidate_index"] is None
        assert not row["training_eligible"]


def test_identical_pairs_allow_ties_only(store):
    saved = request(store, [candidate(), candidate()])
    with pytest.raises(ValueError, match="Identical"):
        store.add_preference(saved["id"], "a")
    store.add_preference(saved["id"], "both")


def test_late_protection_blocks_previously_train_images(store, tmp_path):
    saved = request(store)
    store.add_correction(saved["id"], True)
    assert store.records()[0]["training_eligible"]
    Image.new("RGB", (8, 8), "red").save(tmp_path / "test.png")
    manifest = {
        "schema_version": 1,
        "synthetic": False,
        "name": "held-out",
        "examples": [
            {
                "id": "t",
                "image": "test.png",
                "question": "Other question?",
                "category": "color",
                "answers": ["red"],
                "split": "test",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    store.register_manifest(path)
    assert store.records()[0]["protected"]
    assert store.export(tmp_path / "eligible", True)["count"] == 0
    # Protection cannot be undone by a conflicting later train manifest.
    manifest["examples"][0]["split"] = "train"
    path.write_text(json.dumps(manifest))
    store.register_manifest(path)
    assert store.records()[0]["protected"]


def test_user_held_out_assignment_protects_all_feedback_on_image(store):
    first = request(store)
    store.add_correction(first["id"], True)
    request(store, split="validation")
    assert not store.records()[0]["training_eligible"]


@pytest.mark.parametrize(
    "split,backend,correct,replacement,eligible",
    [
        ("unassigned", "transformers", True, None, False),
        ("test", "transformers", True, None, False),
        ("train", "mock", True, None, False),
        ("train", "transformers", False, None, False),
        ("train", "transformers", False, "blue", True),
        ("train", "transformers", True, None, True),
    ],
)
def test_eligibility(store, split, backend, correct, replacement, eligible):
    saved = request(store, [candidate(backend=backend)], split)
    store.add_correction(saved["id"], correct, replacement)
    assert store.records()[0]["training_eligible"] == eligible


def test_export_round_trip_and_tamper_detection(store, tmp_path):
    saved = request(store)
    store.add_correction(saved["id"], False, "blue")
    output = tmp_path / "bundle"
    info = store.export(output)
    assert validate_export(output) == info
    row = json.loads((output / "feedback.jsonl").read_text())
    assert (output / row["image"]).is_file()
    Image.new("RGB", (8, 8), "blue").save(output / row["image"])
    with pytest.raises(ValueError, match="Image hash"):
        validate_export(output)
    with pytest.raises(ValueError, match="already exists"):
        store.export(output)


def test_invalid_export_mapping_rejected(store, tmp_path):
    saved = request(store, [candidate("red"), candidate("blue")])
    store.add_preference(saved["id"], "a")
    output = tmp_path / "bundle"
    store.export(output)
    path = output / "feedback.jsonl"
    row = json.loads(path.read_text())
    row["preferred_candidate_index"] = 1 - row["preferred_candidate_index"]
    path.write_text(json.dumps(row) + "\n")
    info = json.loads((output / "manifest.json").read_text())
    info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="mapping"):
        validate_export(output)


def test_export_rejects_missing_image_without_partial_output(store, tmp_path):
    saved = request(store)
    store.add_correction(saved["id"], True)
    (store.images / f"{saved['image_sha256']}.png").unlink()
    with pytest.raises(OSError):
        store.export(tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".feedback-export-*"))


def test_validation_and_failed_generation(store):
    with pytest.raises(ValueError):
        request(store, [candidate("")])
    saved = request(store)
    with pytest.raises(ValueError):
        store.add_correction(saved["id"], True, "replacement")
    with pytest.raises(ValueError):
        store.add_correction(saved["id"], "yes")
    with pytest.raises(ValueError):
        store.add_preference(saved["id"], "a")
    engine = InferenceEngine(ModelConfig(backend="mock"))
    engine.predict = lambda *args: Prediction("", 0.1, None, "CUDA out of memory")
    service = FeedbackService(store, engine)
    with pytest.raises(ValueError, match="out of memory"):
        service.generate(Image.new("RGB", (8, 8)), "Question?")
    assert store.summary()["requests"] == 1


def test_parallel_duplicate_submission_is_atomic(store):
    saved = request(store)

    def submit(_):
        try:
            store.add_correction(saved["id"], True)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(submit, range(2))) == 1
    assert store.summary()["feedback"] == 1


def test_service_snapshots_and_pair_provenance(store):
    class Backend:
        resolved_revision = "fake-commit"

        def generate(self, image, question, config):
            return config.prompt_suffix

    first = InferenceEngine(ModelConfig(backend="mock", prompt_suffix="red"), Backend())
    second = InferenceEngine(ModelConfig(backend="mock", prompt_suffix="blue"), Backend())
    saved = FeedbackService(store, first, second).generate(
        Image.new("RGB", (4, 4)), "Original?", pair=True
    )
    assert saved["question"] == "Original?"
    assert {c["answer"] for c in saved["candidates"]} == {"red", "blue"}
    assert saved["candidates"][0]["resolved_revision"] == "fake-commit"
    store.add_preference(saved["id"], "b")
    assert store.records()[0]["software_test_only"]
    with pytest.raises(ValueError, match="comparison-config"):
        FeedbackService(store, first).generate(Image.new("RGB", (4, 4)), "Q?", pair=True)


def test_cli_feedback_does_not_load_models(store, tmp_path, monkeypatch):
    saved = request(store)
    store.add_correction(saved["id"], True)
    monkeypatch.setattr("adaptvlm.cli.InferenceEngine", lambda *args: pytest.fail("loaded model"))
    config = Path(__file__).resolve().parents[1] / "configs/evaluation.yaml"
    output = str(tmp_path / "bundle")
    common = ["--database", str(store.database), "--evaluation-config", str(config)]
    assert main(["feedback-summary", *common]) == 0
    assert main(["feedback-export", *common, "--output", output, "--eligible-only"]) == 0
    assert main(["feedback-validate", "--output", output]) == 0


def test_synthetic_registration_excludes_train_fixture(store):
    root = Path(__file__).resolve().parents[1]
    store.register_manifest(root / "examples/synthetic/manifest.json")
    with Image.open(root / "examples/synthetic/0.png") as image:
        saved = store.create_request(image, "Color?", [candidate()], "train")
    store.add_correction(saved["id"], True)
    assert store.records()[0]["software_test_only"]
    assert not store.records()[0]["training_eligible"]


def test_image_hash_ignores_metadata():
    image = Image.new("RGB", (3, 3), "red")
    image.info["comment"] = "not persisted"
    assert image_digest(image) == image_digest(Image.new("RGB", (3, 3), "red"))


def test_persisted_images_strip_exif_and_bad_sessions_fail(store):
    image = Image.new("RGB", (4, 4), "red")
    exif = Image.Exif()
    exif[270] = "private description"
    image.info["exif"] = exif.tobytes()
    saved = store.create_request(image, "Color?", [candidate()])
    with Image.open(store.images / f"{saved['image_sha256']}.png") as persisted:
        assert not persisted.getexif()
    with pytest.raises(ValueError, match="UUID"):
        store.create_request(image, "Color?", [candidate()], session_id=lambda: "bad")
