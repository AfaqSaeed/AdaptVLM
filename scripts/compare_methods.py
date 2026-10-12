"""Compare SFT and DPO only when recorded data and budgets support comparison."""

import argparse
import json
from adaptvlm.config import load_config
from adaptvlm.evaluation.methods import compare_methods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dpo-adapter", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    args = parser.parse_args()
    try:
        result = compare_methods(
            load_config(args.config),
            args.sft_adapter,
            args.dpo_adapter,
            args.manifest,
            args.output,
            args.split,
        )
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["comparable"] else 2)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
