"""Guarded comparisons of separately trained SFT and DPO adapters."""

from pathlib import Path

from adaptvlm.config import ModelConfig
from adaptvlm.evaluation.comparison import compare_adapter, compare_reports
from adaptvlm.evaluation.runner import write_report
from adaptvlm.training.adapter import read_adapter


def comparison_eligibility(sft: dict, dpo: dict) -> dict:
    """Require matched feedback sources, prompt coverage and training budget."""
    reasons = []
    if sft.get("method") != "sft" or dpo.get("method") != "dpo":
        reasons.append("Adapters must carry explicit SFT and DPO method metadata")
    for field in ("model_id", "family", "prompt_suffix", "dtype"):
        if sft["model_config"].get(field) != dpo["model_config"].get(field):
            reasons.append(f"Training model configuration differs: {field}")
    for field in ("resolved_base_revision", "local_base_sha256"):
        if sft.get(field) != dpo.get(field):
            reasons.append(f"Base checkpoint differs: {field}")
    keys = sft.get("training_example_keys")
    if not keys or set(keys) != set(dpo.get("training_example_keys", [])):
        reasons.append("Training image/question coverage differs or is missing")
    if sft["dataset"]["source_feedback_sha256"] != dpo["dataset"]["source_feedback_sha256"]:
        reasons.append("Adapters must derive from the same mixed feedback export")
    if sft["dataset"]["count"] != dpo["dataset"]["count"]:
        reasons.append("Training example counts differ")
    for field in (
        "seed",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "max_sequence_length",
    ):
        if sft["training_config"].get(field) != dpo["training_config"].get(field):
            reasons.append(f"Training budget/configuration differs: {field}")
    return {
        "comparable": not reasons,
        "limitations": reasons,
        "interpretation": "Matched-coverage observational comparison, not causal proof that "
        "one objective is superior. Corrections and preferences are different signals.",
    }


def compare_methods(
    config: ModelConfig,
    sft_adapter: str | Path,
    dpo_adapter: str | Path,
    manifest: str | Path,
    output: str | Path,
    split: str = "test",
) -> dict:
    """Write a limitation report or matched held-out evaluations and descriptive deltas."""
    import json

    output = Path(output)
    if output.exists():
        raise ValueError("Method-comparison output already exists")
    sft, dpo = read_adapter(sft_adapter), read_adapter(dpo_adapter)
    eligibility = comparison_eligibility(sft, dpo)
    if not eligibility["comparable"]:
        write_report(eligibility, output / "method-comparison.json")
        return eligibility
    compare_adapter(config, sft_adapter, manifest, output / "sft", split)
    compare_adapter(config, dpo_adapter, manifest, output / "dpo", split)
    reports = [
        json.loads((output / method / "adapted.json").read_text(encoding="utf-8"))
        for method in ("sft", "dpo")
    ]
    report = eligibility | {
        "sft_to_dpo": compare_reports(*reports),
        "training_configs": {"sft": sft["training_config"], "dpo": dpo["training_config"]},
    }
    write_report(report, output / "method-comparison.json")
    return report
