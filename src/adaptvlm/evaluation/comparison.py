"""Matched baseline/adapter evaluation and descriptive regression analysis."""

import gc
from dataclasses import replace
from pathlib import Path
from statistics import mean

from adaptvlm.config import ModelConfig
from adaptvlm.evaluation.runner import evaluate, evaluation_image_hashes, write_report
from adaptvlm.inference.engine import InferenceEngine
from adaptvlm.training.adapter import adapter_revision, read_adapter


def compare_reports(baseline: dict, adapted: dict) -> dict:
    """Reject incomparable conditions, then report improvements and regressions by item/category."""
    if baseline["dataset"] != adapted["dataset"] or baseline["scoring"] != adapted["scoring"]:
        raise ValueError("Comparison requires identical dataset, split and scoring")
    fields = set(baseline["config"]) | set(adapted["config"])
    for field in fields - {"adapter_path"}:
        if baseline["config"].get(field) != adapted["config"].get(field):
            raise ValueError(f"Comparison configuration mismatch: {field}")
    if baseline["resolved_model_revision"] != adapted["resolved_model_revision"]:
        raise ValueError("Base checkpoint revisions differ")
    if baseline["environment"] != adapted["environment"]:
        raise ValueError("Comparison environment mismatch")
    before = {row["id"]: row for row in baseline["predictions"]}
    after = {row["id"]: row for row in adapted["predictions"]}
    if (
        not before
        or before.keys() != after.keys()
        or len(before) != len(baseline["predictions"])
        or len(after) != len(adapted["predictions"])
    ):
        raise ValueError("Prediction IDs differ or contain duplicates")
    changes = []
    for identity, a in before.items():
        b = after[identity]
        if any(a[key] != b[key] for key in ("question", "answers", "category", "image_sha256")):
            raise ValueError("Prediction examples differ")
        delta = b["normalized_match"] - a["normalized_match"]
        changes.append(
            {
                "id": identity,
                "question": a["question"],
                "category": a["category"],
                "image_sha256": a["image_sha256"],
                "references": a["answers"],
                "baseline_answer": a["answer"],
                "adapted_answer": b["answer"],
                "baseline_error": a["error"],
                "adapted_error": b["error"],
                "baseline_score": a["normalized_match"],
                "adapted_score": b["normalized_match"],
                "delta": delta,
            }
        )
    categories = {}
    for category in sorted({r["category"] for r in changes}):
        subset = [r for r in changes if r["category"] == category]
        categories[category] = {
            "count": len(subset),
            "baseline": mean(r["baseline_score"] for r in subset),
            "adapted": mean(r["adapted_score"] for r in subset),
            "delta": mean(r["delta"] for r in subset),
        }
    return {
        "schema_version": 1,
        "dataset": baseline["dataset"],
        "software_test_only": baseline["software_test_only"] or adapted["software_test_only"],
        "baseline_metrics": baseline["metrics"],
        "adapted_metrics": adapted["metrics"],
        "normalized_accuracy_delta": mean(r["delta"] for r in changes),
        "categories": categories,
        "improved": [r for r in changes if r["delta"] > 0],
        "degraded": [r for r in changes if r["delta"] < 0],
        "unchanged_count": sum(r["delta"] == 0 for r in changes),
        "interpretation": "Descriptive short-answer scores, not statistical significance or "
        "proof of catastrophic forgetting. No hyperparameter selection.",
    }


def compare_adapter(
    config: ModelConfig,
    adapter: str | Path,
    manifest: str | Path,
    output: str | Path,
    split: str = "test",
) -> dict:
    """Run both checkpoints sequentially with identical settings and a pinned base revision."""
    if (
        config.adapter_path
        or config.backend != "transformers"
        or split not in {"test", "validation"}
    ):
        raise ValueError("Use a real baseline without an adapter and a validation/test split")
    info = read_adapter(adapter)
    adapted_config = replace(config, adapter_path=str(Path(adapter).resolve()))
    revision, _ = adapter_revision(adapted_config)
    baseline_config = replace(config, revision=revision)
    adapted_config = replace(adapted_config, revision=revision)
    # Preflight overlap before loading either set of weights.
    from adaptvlm.datasets.manifest import load_manifest

    data, _ = load_manifest(manifest)
    if set(info["training_image_hashes"]) & evaluation_image_hashes(
        [r for r in data["examples"] if r["split"] == split]
    ):
        raise ValueError("Comparison split overlaps adapter training images")
    output = Path(output)
    if output.exists():
        raise ValueError("Comparison output already exists")
    baseline_engine = InferenceEngine(baseline_config)
    baseline = evaluate(baseline_engine, manifest, split)
    del baseline_engine
    gc.collect()
    if config.device == "cuda":
        import torch

        torch.cuda.empty_cache()
    adapted = evaluate(InferenceEngine(adapted_config), manifest, split)
    result = compare_reports(baseline, adapted)
    output.mkdir(parents=True)
    write_report(baseline, output / "baseline.json")
    write_report(adapted, output / "adapted.json")
    write_report(result, output / "comparison.json")
    return result
