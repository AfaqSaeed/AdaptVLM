"""Evaluate baseline and adapter on the same held-out examples and compare predictions."""

import argparse
import json

from adaptvlm.config import load_config
from adaptvlm.evaluation.comparison import compare_adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    args = parser.parse_args()
    try:
        result = compare_adapter(
            load_config(args.config), args.adapter, args.manifest, args.output, args.split
        )
        print(
            json.dumps(
                {
                    "accuracy_delta": result["normalized_accuracy_delta"],
                    "improved": len(result["improved"]),
                    "degraded": len(result["degraded"]),
                },
                indent=2,
            )
        )
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
