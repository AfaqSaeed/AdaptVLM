"""Measured efficiency fixtures are software tests, never speedup evidence."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from adaptvlm.config import ModelConfig
from adaptvlm.benchmarking.runner import run_suite, run_variant, summarize
from adaptvlm.models.quantization import quantization_support

ROOT = Path(__file__).resolve().parents[1]


def test_latency_percentile_and_failure_denominators():
    samples = [
        {
            "error": None,
            "latency_seconds": n,
            "exact_match": 1,
            "normalized_match": 1,
            "peak_gpu_memory_bytes": None,
        }
        for n in [1, 2, 3, 4]
    ]
    samples.append(
        {
            "error": "OOM",
            "latency_seconds": 100,
            "exact_match": 0,
            "normalized_match": 0,
            "peak_gpu_memory_bytes": None,
        }
    )
    result = summarize(samples, 20, 5)
    assert result["median_latency_seconds"] == 2.5
    assert result["p95_latency_seconds"] == pytest.approx(3.85)
    assert result["normalized_match"] == 0.8
    assert result["successful_requests_per_second"] == 0.2
    assert result["peak_gpu_allocated_bytes"] is None
    assert result["failures"] == 1


def test_unsupported_quantization_and_training_gates(monkeypatch):
    import torch
    from adaptvlm.training.dpo import check_compatibility

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = ModelConfig(device="cuda", dtype="float16", quantization="bnb_nf4")
    assert not quantization_support(config)["supported"]
    assert not quantization_support(replace(config, quantization="dynamic_int8"))["supported"]
    assert not check_compatibility(ModelConfig(quantization="dynamic_int8"))["supported"]
    with pytest.raises(ValueError):
        ModelConfig(backend="mock", quantization="dynamic_int8")


def suite_file(tmp_path, variants=None):
    model = tmp_path / "mock.yaml"
    model.write_text(yaml.safe_dump(ModelConfig(backend="mock").to_dict()))
    settings = {
        "manifest": str(ROOT / "examples/synthetic/manifest.json"),
        "split": "test",
        "warmup_requests": 2,
        "repeats": 2,
        "order_seed": 42,
        "variants": variants
        or [
            {"name": "first", "model_config": "mock.yaml"},
            {"name": "second", "model_config": "mock.yaml"},
        ],
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_mock_suite_order_metadata_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from adaptvlm.benchmarking.reporting import export_visuals, read_benchmark

    report = run_suite(suite_file(tmp_path), tmp_path / "results", isolate=False)
    assert report["comparison_eligible"] and report["software_test_only"]
    assert report["results"][0]["example_order"] == report["results"][1]["example_order"]
    for result in report["results"]:
        assert result["metrics"]["attempts"] == 4  # Warm-up not counted.
        assert result["metrics"]["normalized_match"] == 0.5
        assert result["metrics"]["peak_gpu_allocated_bytes"] is None
        assert result["environment"]["torch_threads"] == 1
    paths = export_visuals(tmp_path / "results/benchmark.json")
    assert len(paths) == 3 and all(Path(path).stat().st_size > 0 for path in paths)
    assert read_benchmark(tmp_path / "results/benchmark.json")["kind"] == "efficiency-benchmark"
    with pytest.raises(ValueError, match="already exists"):
        run_suite(tmp_path / "benchmark.yaml", tmp_path / "results", isolate=False)


def test_suite_rejects_unfair_settings(tmp_path):
    path = suite_file(
        tmp_path,
        [
            {"name": "a", "model_config": "mock.yaml"},
            {"name": "b", "model_config": "mock.yaml", "overrides": {"seed": 13}},
        ],
    )
    with pytest.raises(ValueError, match="share base"):
        run_suite(path, tmp_path / "bad", isolate=False)
    assert not (tmp_path / "bad").exists()


def test_failed_and_unsupported_variants_have_no_metrics(tmp_path, monkeypatch):
    base = {
        "name": "failed",
        "manifest": str(ROOT / "examples/synthetic/manifest.json"),
        "split": "test",
        "warmup_requests": 1,
        "repeats": 1,
        "order_seed": 42,
        "config": ModelConfig(backend="mock").to_dict(),
    }

    def failed_load(*args):
        raise RuntimeError("load failure")

    monkeypatch.setattr("adaptvlm.benchmarking.runner.InferenceEngine", failed_load)
    result = run_variant(base)
    assert result["status"] == "failed" and result["metrics"] is None
    assert result["failure_stage"] == "load"
    base["config"] = ModelConfig(quantization="dynamic_int8", device="cuda").to_dict()
    result = run_variant(base)
    assert result["status"] == "unsupported" and result["metrics"] is None


@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="Install training extra")
def test_real_four_variant_tiny_suite(tmp_path):
    from test_training import tiny_checkpoint
    from adaptvlm.training.runner import attach_lora
    from adaptvlm.training.config import SFTConfig
    from adaptvlm.training.adapter import file_hash, local_base_hash

    base, adapter = tmp_path / "base", tmp_path / "adapter"
    original, _ = tiny_checkpoint(base)
    model = attach_lora(original, SFTConfig(lora_rank=2, lora_alpha=4))
    # Test-only adapter, no invented training run or quality claim.
    model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    config = ModelConfig(model_id=str(base), max_new_tokens=2, prompt_suffix="")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "software_test_only": True,
        "model_config": config.to_dict(),
        "resolved_base_revision": None,
        "local_base_sha256": local_base_hash(str(base)),
        "training_image_hashes": [],
        "dataset": {"name": "test-only"},
        "adapter_files": {
            name: file_hash(adapter / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
    }
    (adapter / "training_run.json").write_text(json.dumps(metadata))
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(config.to_dict()))
    variants = [
        {"name": "baseline", "model_config": "model.yaml"},
        {
            "name": "adapted",
            "model_config": "model.yaml",
            "overrides": {"adapter_path": str(adapter)},
        },
        {
            "name": "int8",
            "model_config": "model.yaml",
            "overrides": {"quantization": "dynamic_int8"},
        },
        {
            "name": "adapted-int8",
            "model_config": "model.yaml",
            "overrides": {"quantization": "dynamic_int8", "adapter_path": str(adapter)},
        },
    ]
    suite = {
        "manifest": str(ROOT / "examples/synthetic/manifest.json"),
        "warmup_requests": 1,
        "repeats": 2,
        "variants": variants,
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    report = run_suite(path, tmp_path / "results", isolate=False)
    assert all(r["status"] == "completed" for r in report["results"]), report
    assert report["comparison_eligible"] and report["software_test_only"]
    for result in report["results"]:
        assert result["serialized_state_bytes"] > 0
        assert result["metrics"]["attempts"] == 4
        if result["config"]["quantization"] == "dynamic_int8":
            assert result["quantization_details"]["converted_linear_layers"] > 0
    adapted = next(r for r in report["results"] if r["name"] == "adapted-int8")
    assert adapted["quantization_details"]["adapter_merged"]


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None or importlib.util.find_spec("matplotlib") is None,
    reason="Install app/benchmark extras",
)
def test_experiment_dashboard_callbacks(tmp_path):
    from adaptvlm.dashboard.experiments import build_experiment_app

    run_suite(suite_file(tmp_path), tmp_path / "suite", isolate=False)
    app = build_experiment_app(tmp_path)
    try:
        callbacks = {fn.api_name: fn.fn for fn in app.fns.values()}
        assert ("suite/benchmark.json", "suite/benchmark.json") in callbacks["refresh"]().choices
        outputs = callbacks["experiment"]("suite/benchmark.json")
        assert "Software-test" in outputs[0] and len(outputs[1]) == 2
        assert len(outputs) == 6
        with pytest.raises(Exception, match="within"):
            callbacks["experiment"]("../outside/benchmark.json")
    finally:
        app.close()
