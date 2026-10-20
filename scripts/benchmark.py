"""Run a controlled efficiency suite and export plots from actual measurements."""

import argparse
from adaptvlm.benchmarking.runner import run_suite
from adaptvlm.benchmarking.reporting import export_visuals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = run_suite(args.config, args.output)
        export_visuals(f"{args.output}/benchmark.json")
        for result in report["results"]:
            print(f"{result['name']}: {result['status']} {result.get('reason', '')}")
        print(f"Report: {args.output}/benchmark.json")
        raise SystemExit(0 if all(r["status"] == "completed" for r in report["results"]) else 2)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
