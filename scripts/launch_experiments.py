"""Open a local benchmark comparison dashboard without loading model weights."""

import argparse
from adaptvlm.dashboard.experiments import build_experiment_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    build_experiment_app(args.reports).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )
