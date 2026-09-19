"""Reproducible short-answer evaluation, not free-form semantic grading."""

import importlib.metadata
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageOps

from adaptvlm.datasets.manifest import load_manifest
from adaptvlm.inference.engine import InferenceEngine
from adaptvlm.feedback.store import image_digest


def evaluation_image_hashes(rows: list[dict]) -> set[str]:
    """Include EXIF-oriented pixels when checking train/evaluation overlap."""
    hashes = {row["image_sha256"] for row in rows}
    for row in rows:
        with Image.open(row["image_path"]) as image:
            hashes.add(image_digest(ImageOps.exif_transpose(image)))
    return hashes


def normalize(answer: str) -> str:
    """Conservatively normalize case, whitespace and terminal punctuation."""
    return re.sub(r"\s+", " ", answer.lower().strip()).rstrip(".!?")


def evaluate(engine: InferenceEngine, manifest: str | Path, split: str = "test") -> dict:
    """Evaluate every selected row, retaining failures in score denominators."""
    data, fingerprint = load_manifest(manifest)
    rows = [r for r in data["examples"] if r["split"] == split]
    adapter = getattr(engine.backend, "adapter_metadata", None)
    if adapter and set(adapter["training_image_hashes"]) & evaluation_image_hashes(rows):
        raise ValueError("Evaluation images overlap the adapter training set")
    if not rows:
        raise ValueError(f"No examples for split {split}")
    predictions = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            result = engine.predict(image, row["question"])
        exact = not result.error and result.answer in row["answers"]
        normalized = not result.error and normalize(result.answer) in {
            normalize(a) for a in row["answers"]
        }
        predictions.append(
            {k: v for k, v in row.items() if k != "image_path"}
            | {
                **result.to_dict(),
                "exact_match": int(exact),
                "normalized_match": int(normalized),
            }
        )
    versions = {}
    for package in ("adaptvlm", "torch", "transformers", "peft", "Pillow", "gradio", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = None
    if engine.config.device == "cuda" and engine.config.backend != "mock":
        import torch

        gpu = {"name": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "software_test_only": data["synthetic"]
        or engine.config.backend == "mock"
        or bool(adapter and adapter.get("software_test_only")),
        "adapter": {"files": adapter["adapter_files"], "dataset": adapter["dataset"]}
        if adapter
        else None,
        "dataset": {
            "name": data["name"],
            "sha256": fingerprint,
            "split": split,
            "source": data.get("source"),
            "synthetic": data["synthetic"],
        },
        "config": engine.config.to_dict(),
        "resolved_model_revision": getattr(engine.backend, "resolved_revision", None),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "gpu": gpu,
        },
        "metrics": {
            "count": len(rows),
            "failures": sum(bool(p["error"]) for p in predictions),
            "exact_match": mean(p["exact_match"] for p in predictions),
            "normalized_match": mean(p["normalized_match"] for p in predictions),
            "median_latency_seconds": median(p["latency_seconds"] for p in predictions),
            "peak_gpu_memory_bytes": max(
                (p["peak_gpu_memory_bytes"] or 0 for p in predictions), default=0
            )
            or None,
            "model_load_seconds": engine.load_seconds,
        },
        "categories": {
            category: {
                "count": sum(p["category"] == category for p in predictions),
                "normalized_match": mean(
                    p["normalized_match"] for p in predictions if p["category"] == category
                ),
            }
            for category in sorted({p["category"] for p in predictions})
        },
        "scoring": "Short-answer exact/normalized reference match; NOT official VQA accuracy "
        "or semantic correctness. Latency includes preprocessing, no warm-up.",
        "predictions": predictions,
    }


def write_report(report: dict, output: str | Path) -> None:
    """Write a readable JSON baseline artifact."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
