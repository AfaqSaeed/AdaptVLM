"""Convert an eligible feedback bundle into validated correction-only SFT data."""

import argparse
import json

from adaptvlm.training.data import prepare_sft

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protected-manifest", action="append", required=True)
    args = parser.parse_args()
    try:
        print(
            json.dumps(prepare_sft(args.feedback, args.output, args.protected_manifest), indent=2)
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"AdaptVLM: {exc}\n")
