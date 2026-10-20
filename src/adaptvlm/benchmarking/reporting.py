"""Portable plots and tables from measured benchmark artifacts only."""

import csv
import json
from pathlib import Path


def read_benchmark(path: str | Path) -> dict:
    """Load a benchmark artifact, not a training or baseline-evaluation report."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("kind") != "efficiency-benchmark":
        raise ValueError("Select an efficiency benchmark.json report")
    return report


def summary_rows(report: dict) -> list[dict]:
    """Keep unsupported and failed variants visible with blank measurements."""
    rows = []
    for result in report["results"]:
        metrics = result.get("metrics") or {}
        rows.append(
            {
                "name": result["name"],
                "status": result["status"],
                "quantization": result["config"].get("quantization", "none"),
                "adapter": result["config"].get("adapter_path"),
                "accuracy": metrics.get("normalized_match"),
                "median_seconds": metrics.get("median_latency_seconds"),
                "p95_seconds": metrics.get("p95_latency_seconds"),
                "requests_per_second": metrics.get("successful_requests_per_second"),
                "gpu_peak_allocated_bytes": metrics.get("peak_gpu_allocated_bytes"),
                "gpu_peak_reserved_bytes": metrics.get("peak_gpu_reserved_bytes"),
                "serialized_state_bytes": result.get("serialized_state_bytes"),
                "model_load_seconds": result.get("model_load_seconds"),
                "training_seconds": result.get("training_seconds"),
                "failures": metrics.get("failures"),
                "reason": result.get("reason"),
            }
        )
    return rows


def make_figures(report: dict) -> dict:
    """Plot quality versus latency, GPU allocation and serialized state size."""
    from matplotlib.figure import Figure

    results = [r for r in summary_rows(report) if r["status"] == "completed"]
    figures = {}
    for name, field, label, scale in (
        ("latency-quality", "median_seconds", "Median request latency (seconds)", 1),
        (
            "gpu-memory-quality",
            "gpu_peak_allocated_bytes",
            "Peak GPU allocated memory (MiB)",
            1024**2,
        ),
        (
            "state-size-quality",
            "serialized_state_bytes",
            "Serialized model state (MiB), NOT runtime RAM",
            1024**2,
        ),
    ):
        figure = Figure(figsize=(8, 5), layout="constrained")
        ax = figure.subplots()
        points = [r for r in results if r[field] is not None]
        for row in points:
            ax.scatter(row[field] / scale, row["accuracy"], s=65, label=row["name"])
        if points:
            ax.legend(loc="best", fontsize=8)
        if not points:
            ax.text(0.5, 0.5, "No measured values available", transform=ax.transAxes, ha="center")
        ax.set(xlabel=label, ylabel="Normalized answer accuracy", ylim=(-0.05, 1.1))
        note = (
            "SOFTWARE TEST ONLY"
            if report["software_test_only"]
            else "Measured observations; no speedup claim"
        )
        if not report["comparison_eligible"]:
            note += "\nComparison conditions not satisfied"
        ax.set_title(note)
        ax.grid(alpha=0.2)
        figures[name] = figure
    return figures


def export_visuals(report_path: str | Path) -> list[str]:
    """Write PNG plots and CSV alongside an existing benchmark report."""
    report_path = Path(report_path)
    report = read_benchmark(report_path)
    paths = []
    for name, figure in make_figures(report).items():
        target = report_path.parent / f"{name}.png"
        figure.savefig(target, dpi=150)
        paths.append(str(target.resolve()))
    rows = summary_rows(report)
    if rows:
        with (report_path.parent / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths
