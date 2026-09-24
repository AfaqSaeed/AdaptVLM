"""Train a new SmolVLM LoRA adapter on validated human corrections."""

import argparse
import json

from adaptvlm.config import load_config
from adaptvlm.training.config import load_sft_config
from adaptvlm.training.runner import train_sft

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/sft.yaml")
    parser.add_argument("--protected-manifest", action="append", required=True)
    args = parser.parse_args()
    try:
        result = train_sft(
            args.data,
            args.output,
            load_config(args.model_config),
            load_sft_config(args.training_config),
            args.protected_manifest,
        )
        print(json.dumps(result["statistics"], indent=2))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
