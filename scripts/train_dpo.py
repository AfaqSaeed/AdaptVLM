"""Check compatibility or run experimental multimodal TRL DPO."""

import argparse
import json
from pathlib import Path
import yaml
from adaptvlm.config import load_config
from adaptvlm.training.dpo import PreferenceConfig, check_compatibility, train_dpo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/dpo.yaml")
    parser.add_argument("--protected-manifest", action="append", default=[])
    args = parser.parse_args()
    try:
        model = load_config(args.model_config)
        if args.check_only:
            report = check_compatibility(model)
            if args.output:
                path = Path(args.output)
                if path.exists():
                    raise ValueError("Compatibility report already exists")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            raise SystemExit(0 if report["supported"] else 2)
        if not args.data or not args.output or not args.protected_manifest:
            parser.error("Training requires --data, --output and --protected-manifest")
        config = PreferenceConfig(
            **yaml.safe_load(Path(args.training_config).read_text(encoding="utf-8"))
        )
        result = train_dpo(args.data, args.output, model, config, args.protected_manifest)
        print(json.dumps(result["statistics"], indent=2))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
