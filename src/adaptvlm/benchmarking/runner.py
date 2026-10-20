"""Serial, optionally process-isolated inference experiment suites."""

import gc
import importlib.metadata
import io
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

import numpy as np
import yaml
from PIL import Image

from adaptvlm.config import ModelConfig, load_config
from adaptvlm.datasets.manifest import load_manifest
from adaptvlm.evaluation.runner import evaluation_image_hashes, normalize, write_report
from adaptvlm.inference.engine import InferenceEngine
from adaptvlm.models.quantization import quantization_support
from adaptvlm.training.adapter import local_base_hash


def environment() -> dict:
    """Capture software, CPU threading and available CUDA hardware."""
    import torch

    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "numpy", "Pillow", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "quantized_engine": torch.backends.quantized.engine,
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
        if torch.cuda.is_available()
        else None,
    }


def summarize(samples: list[dict], measured_seconds: float, unique_examples: int) -> dict:
    """Use successful-request latency and all-attempt accuracy; retain failures explicitly."""
    successes = [s for s in samples if not s["error"]]
    latencies = [s["latency_seconds"] for s in successes]
    if any(not math.isfinite(t) or t < 0 for t in latencies):
        raise ValueError("Invalid latency measurement")
    return {
        "unique_examples": unique_examples,
        "attempts": len(samples),
        "successful_requests": len(successes),
        "failures": len(samples) - len(successes),
        "exact_match": mean(s["exact_match"] for s in samples),
        "normalized_match": mean(s["normalized_match"] for s in samples),
        "median_latency_seconds": median(latencies) if latencies else None,
        "p95_latency_seconds": float(np.percentile(latencies, 95, method="linear"))
        if latencies
        else None,
        "successful_requests_per_second": len(successes) / measured_seconds
        if measured_seconds > 0
        else None,
        "measurement_seconds": measured_seconds,
        "peak_gpu_allocated_bytes": max(
            (s["peak_gpu_memory_bytes"] for s in samples if s["peak_gpu_memory_bytes"] is not None),
            default=None,
        ),
        "peak_gpu_reserved_bytes": max(
            (
                s["peak_gpu_reserved_bytes"]
                for s in samples
                if s.get("peak_gpu_reserved_bytes") is not None
            ),
            default=None,
        ),
    }


def run_variant(request: dict) -> dict:
    """Measure one configuration; unsupported/failed configurations produce no invented metrics."""
    config = ModelConfig(**request["config"])
    import torch

    torch.set_num_threads(request.get("torch_threads", 1))
    result = {
        "name": request["name"],
        "config": config.to_dict(),
        "status": "failed",
        "metrics": None,
        "failure_stage": "capability",
        "environment": environment(),
    }
    engine = None
    try:
        support = quantization_support(config)
        if not support["supported"]:
            return result | {"status": "unsupported", "reason": "; ".join(support["reasons"])}
        dataset, fingerprint = load_manifest(request["manifest"])
        rows = [row for row in dataset["examples"] if row["split"] == request["split"]]
        if not rows:
            raise ValueError("Benchmark split is empty")
        result["dataset"] = {
            "name": dataset["name"],
            "sha256": fingerprint,
            "split": request["split"],
            "synthetic": dataset["synthetic"],
        }
        result["failure_stage"] = "load"
        start = time.perf_counter()
        engine = InferenceEngine(config)
        gpu = config.device == "cuda" and config.backend != "mock"
        if gpu:
            import torch

            torch.cuda.synchronize()
        result["model_load_seconds"] = time.perf_counter() - start
        adapter = getattr(engine.backend, "adapter_metadata", None)
        if adapter and set(adapter["training_image_hashes"]) & evaluation_image_hashes(rows):
            raise ValueError("Benchmark images overlap adapter training data")
        result["software_test_only"] = (
            dataset["synthetic"]
            or config.backend == "mock"
            or bool(adapter and adapter.get("software_test_only"))
        )
        result["resolved_model_revision"] = getattr(engine.backend, "resolved_revision", None)
        result["local_base_sha256"] = (
            local_base_hash(config.model_id) if config.backend != "mock" else None
        )
        result["quantization_details"] = getattr(
            engine.backend, "quantization_details", {"mode": "none"}
        )
        processor = getattr(engine.backend, "processor", None)
        image_processor = getattr(processor, "image_processor", None)
        result["image_processor"] = (
            json.loads(json.dumps(image_processor.to_dict(), default=str))
            if image_processor
            else None
        )
        result["adapter_files"] = adapter["adapter_files"] if adapter else None
        statistics = adapter.get("statistics", {}) if adapter else {}
        result["training_seconds"] = statistics.get(
            "training_seconds", statistics.get("metrics", {}).get("train_runtime")
        )
        result["training_compute_cost"] = None
        images = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                images.append(image.copy())
        result["input_image_sizes"] = [list(image.size) for image in images]
        result["failure_stage"] = "warmup"
        warmup_start = time.perf_counter()
        for index in range(request["warmup_requests"]):
            i = index % len(rows)
            prediction = engine.predict(images[i], rows[i]["question"])
            if prediction.error:
                raise RuntimeError(f"Warm-up failed: {prediction.error}")
        result["warmup_seconds"] = time.perf_counter() - warmup_start
        result["failure_stage"] = "measurement"
        samples, order = [], []
        rng = random.Random(request["order_seed"])
        start = time.perf_counter()
        for repeat in range(request["repeats"]):
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            order.append([rows[i]["id"] for i in indices])
            for i in indices:
                row = rows[i]
                prediction = engine.predict(images[i], row["question"])
                exact = not prediction.error and prediction.answer in row["answers"]
                normalized = not prediction.error and normalize(prediction.answer) in {
                    normalize(a) for a in row["answers"]
                }
                samples.append(
                    prediction.to_dict()
                    | {
                        "id": row["id"],
                        "repeat": repeat,
                        "question": row["question"],
                        "category": row["category"],
                        "references": row["answers"],
                        "image_sha256": row["image_sha256"],
                        "exact_match": int(exact),
                        "normalized_match": int(normalized),
                        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()
                        if gpu
                        else None,
                    }
                )
        elapsed = time.perf_counter() - start
        result["metrics"] = summarize(samples, elapsed, len(rows))
        result["samples"], result["example_order"] = samples, order
        result["categories"] = {
            category: mean(s["normalized_match"] for s in samples if s["category"] == category)
            for category in sorted({s["category"] for s in samples})
        }
        result["repeat_answers_consistent"] = all(
            len({s["answer"] for s in samples if s["id"] == row["id"]}) == 1 for row in rows
        )
        # Quantized packed weights are not parameters: measure the serialized state, not
        # an incorrect sum of parameter bytes. This is NOT process RAM or GPU memory.
        model = getattr(engine.backend, "model", None)
        if model is not None:
            import torch

            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            result["serialized_state_bytes"] = buffer.tell()
        else:
            result["serialized_state_bytes"] = None
        result["status"] = "completed" if result["metrics"]["failures"] == 0 else "partial"
        result.pop("failure_stage")
        return result
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        return result | {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "metrics": None,
        }
    finally:
        del engine
        gc.collect()


def run_suite(configuration: str | Path, output: str | Path, isolate: bool = True) -> dict:
    """Run matched variants serially; persist raw samples and explicit capability failures."""
    path, output = Path(configuration).resolve(), Path(output).resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = (path.parent / settings["manifest"]).resolve()
    split = settings.get("split", "test")
    if split not in {"validation", "test"}:
        raise ValueError("Benchmark requires a validation or test split")
    common = {
        "manifest": str(manifest),
        "split": split,
        "warmup_requests": settings.get("warmup_requests", 1),
        "repeats": settings.get("repeats", 3),
        "order_seed": settings.get("order_seed", 42),
        "torch_threads": settings.get("torch_threads", 1),
    }
    if any(
        type(common[k]) is not int or common[k] < minimum
        for k, minimum in (
            ("warmup_requests", 0),
            ("repeats", 1),
            ("order_seed", 0),
            ("torch_threads", 1),
        )
    ):
        raise ValueError("Warm-up/seed must be nonnegative integers; repeats must be positive")
    variants, names = [], set()
    for entry in settings["variants"]:
        name = entry["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Variant names must be unique nonempty strings")
        names.add(name)
        config = load_config(path.parent / entry["model_config"])
        overrides = dict(entry.get("overrides", {}))
        if overrides.get("adapter_path"):
            overrides["adapter_path"] = str((path.parent / overrides["adapter_path"]).resolve())
        config = replace(config, **overrides)
        variants.append(common | {"name": name, "config": config.to_dict()})
    if not variants:
        raise ValueError("No benchmark variants")
    fair_fields = (
        "model_id",
        "family",
        "revision",
        "backend",
        "device",
        "seed",
        "max_new_tokens",
        "prompt_suffix",
    )
    if any(any(v["config"][k] != variants[0]["config"][k] for k in fair_fields) for v in variants):
        raise ValueError("Variants must share base, device, seed, prompt and generation settings")
    dataset, fingerprint = load_manifest(manifest)
    if not any(r["split"] == split for r in dataset["examples"]):
        raise ValueError("Benchmark split is empty")
    if output.exists():
        raise ValueError("Benchmark output already exists; choose a new directory")
    output.mkdir(parents=True)
    random.Random(common["order_seed"]).shuffle(variants)
    results = []
    for index, request in enumerate(variants):
        request_path, result_path = output / f"request-{index}.json", output / f"run-{index}.json"
        write_report(request, request_path)
        if isolate:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "adaptvlm.benchmarking.worker",
                    str(request_path),
                    str(result_path),
                ],
                capture_output=True,
                text=True,
            )
            (output / f"worker-{index}.log").write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            if process.returncode or not result_path.exists():
                result = {
                    "name": request["name"],
                    "config": request["config"],
                    "status": "failed",
                    "metrics": None,
                    "reason": f"Worker failed (exit {process.returncode}); see worker-{index}.log",
                }
                write_report(result, result_path)
            else:
                result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = run_variant(request)
            write_report(result, result_path)
        results.append(result)
    completed = [r for r in results if r["status"] == "completed"]
    reasons = []
    if len(completed) < 2:
        reasons.append("At least two completed variants are required for comparison")
    if completed and any(
        any(
            r.get(k) != completed[0].get(k)
            for k in (
                "dataset",
                "resolved_model_revision",
                "local_base_sha256",
                "environment",
                "image_processor",
                "example_order",
            )
        )
        for r in completed
    ):
        reasons.append("Resolved checkpoints, data, preprocessing, order or environments differ")
    report = {
        "schema_version": 1,
        "kind": "efficiency-benchmark",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": fingerprint,
        "settings": common,
        "batch_size": 1,
        "process_isolation": isolate,
        "variant_order": [v["name"] for v in variants],
        "software_test_only": dataset["synthetic"]
        or any(r.get("software_test_only", True) for r in completed),
        "comparison_eligible": not reasons,
        "comparison_scope": [r["name"] for r in completed],
        "omitted_variants": [r["name"] for r in results if r["status"] != "completed"],
        "comparison_limitations": reasons,
        "results": results,
        "measurement_notes": "Latency includes processor/generation/decode; excludes image-file reads and warm-up. "
        "Throughput includes in-memory image normalization and loop overhead. P95 uses linear interpolation. "
        "Accuracy includes failed attempts and repeated examples are not independent evidence. "
        "GPU bytes are allocator measurements, not total device use. Serialized state size is not runtime RAM. "
        "Load time may include downloads/cache effects. No speedup or statistical significance is asserted.",
    }
    write_report(report, output / "benchmark.json")
    return report
