"""Extract auditable software-validation evidence without generating research claims."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    """Fingerprint the exact local file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evidence(benchmark: Path, junit: Path, output: Path) -> dict:
    """Validate source counts and extract measured fields with provenance hashes."""
    root = Path(__file__).resolve().parents[1]
    raw = json.loads(benchmark.read_text(encoding="utf-8"))
    if raw.get("kind") != "efficiency-benchmark" or raw.get("software_test_only") is not True:
        raise ValueError("This evidence extractor expects a software-test benchmark")
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    counts = {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] == 0 or counts["failures"] or counts["errors"] or counts["skipped"]:
        raise ValueError(
            "Report validation requires a nonempty, passing suite with no skipped tests"
        )
    results = []
    for result in raw["results"]:
        metrics = result.get("metrics")
        if result["status"] == "completed":
            if len(result["samples"]) != metrics["attempts"]:
                raise ValueError("Benchmark sample count differs from metrics")
            if len({sample["id"] for sample in result["samples"]}) != metrics["unique_examples"]:
                raise ValueError("Benchmark unique-example count mismatch")
            if metrics["normalized_match"] != sum(
                s["normalized_match"] for s in result["samples"]
            ) / len(result["samples"]):
                raise ValueError("Reported accuracy disagrees with source samples")
        results.append(
            {
                "name": result["name"],
                "status": result["status"],
                "metrics": metrics,
                "model_load_seconds": result.get("model_load_seconds"),
                "serialized_state_bytes": result.get("serialized_state_bytes"),
                "quantization": result["config"].get("quantization"),
                "reason": result.get("reason"),
                "local_base_sha256": result.get("local_base_sha256"),
                "environment": result.get("environment"),
                "first_repeat_predictions": [
                    {
                        key: sample[key]
                        for key in ("id", "question", "answer", "references", "error")
                    }
                    for sample in result.get("samples", [])
                    if sample["repeat"] == 0
                ],
            }
        )
    sources = {}
    for folder in ("src", "tests", "scripts", "configs", "examples"):
        for path in sorted((root / folder).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".yaml", ".json", ".png"}
            ):
                sources[path.relative_to(root).as_posix()] = digest(path)
    for name in ("pyproject.toml", "requirements-tested.txt", ".github/workflows/tests.yml"):
        sources[name] = digest(root / name)
    source_fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    evidence = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "classification": "software-validation-only",
        "research_hypotheses_tested": False,
        "source_fingerprint": source_fingerprint,
        "source_files": sources,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_clean": not bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        ),
        "test_run": counts | {"passed": counts["tests"], "junit_sha256": digest(junit)},
        "benchmark": {
            "source_sha256": digest(benchmark),
            "created_at": raw["created_at"],
            "dataset_sha256": raw["dataset_sha256"],
            "batch_size": raw["batch_size"],
            "process_isolation": raw["process_isolation"],
            "settings": {k: v for k, v in raw["settings"].items() if k != "manifest"},
            "results": results,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("reports/phase5-tiny-smoke/benchmark.json")
    )
    parser.add_argument("--junit", type=Path, default=Path("reports/phase6-validation.xml"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/evidence/phase6-evidence.json")
    )
    args = parser.parse_args()
    try:
        result = build_evidence(args.benchmark, args.junit, args.output)
        print(f"Evidence: {args.output}; {result['test_run']['passed']} tests passed")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
