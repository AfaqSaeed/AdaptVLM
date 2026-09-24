"""Command line entry points for inference, evaluation and feedback workflows."""

import argparse
import json
import sqlite3
from pathlib import Path

import yaml
from PIL import Image

from adaptvlm.config import load_config
from adaptvlm.evaluation.runner import evaluate, write_report
from adaptvlm.inference.engine import InferenceEngine


def main(argv: list[str] | None = None) -> int:
    """Run inference, evaluation, or the local app."""
    parser = argparse.ArgumentParser(description="AdaptVLM inference and feedback")
    parser.add_argument(
        "command",
        choices=[
            "infer",
            "evaluate",
            "app",
            "feedback-export",
            "feedback-summary",
            "feedback-validate",
        ],
    )
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--image")
    parser.add_argument("--question")
    parser.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--output")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--database", default="data/feedback.sqlite3")
    parser.add_argument("--comparison-config")
    parser.add_argument(
        "--protected-manifest",
        action="append",
        default=[],
        help="Register dataset splits; repeat for all known datasets",
    )
    parser.add_argument("--eligible-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "infer" and (not args.image or not args.question):
        parser.error("infer requires --image and --question")
    try:
        if args.command == "feedback-validate":
            from adaptvlm.feedback.validation import validate_export

            if not args.output:
                parser.error("feedback-validate requires --output BUNDLE_DIRECTORY")
            print(json.dumps(validate_export(args.output), indent=2))
            return 0
        store = None
        if args.command in {"app", "feedback-export", "feedback-summary"}:
            from adaptvlm.feedback.store import FeedbackStore

            store = FeedbackStore(args.database)
            settings = yaml.safe_load(Path(args.evaluation_config).read_text(encoding="utf-8"))
            for manifest in [settings["manifest"], *args.protected_manifest]:
                store.register_manifest(manifest)
            if args.command == "feedback-summary":
                print(json.dumps(store.summary(), indent=2))
                return 0
            if args.command == "feedback-export":
                if not args.output:
                    parser.error("feedback-export requires --output NEW_BUNDLE_DIRECTORY")
                print(json.dumps(store.export(args.output, args.eligible_only), indent=2))
                return 0
        engine = InferenceEngine(load_config(args.config))
        if args.command == "infer":
            with Image.open(args.image) as image:
                result = engine.predict(image, args.question)
            print(json.dumps(result.to_dict(), indent=2))
            return 1 if result.error else 0
        if args.command == "evaluate":
            settings = yaml.safe_load(Path(args.evaluation_config).read_text(encoding="utf-8"))
            report = evaluate(
                engine, args.manifest or settings["manifest"], args.split or settings["split"]
            )
            output = args.output or settings["output"]
            write_report(report, output)
            print(f"Report: {output}; software_test_only={report['software_test_only']}")
            return 1 if report["metrics"]["failures"] else 0
        from adaptvlm.dashboard.app import build_app

        comparison = None
        if args.comparison_config:
            comparison_config = load_config(args.comparison_config)
            # Reuse weights and the synchronization lock for prompt-only comparisons.
            loading_fields = (
                "backend",
                "model_id",
                "revision",
                "family",
                "device",
                "dtype",
                "adapter_path",
            )
            if all(
                getattr(engine.config, f) == getattr(comparison_config, f) for f in loading_fields
            ):
                comparison = InferenceEngine(comparison_config, engine.backend)
                comparison.lock = engine.lock
            else:
                comparison = InferenceEngine(comparison_config)
        build_app(engine, store, comparison).queue().launch(
            server_name="127.0.0.1", server_port=args.port, share=False
        )
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, sqlite3.Error) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
