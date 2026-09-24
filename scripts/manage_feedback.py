"""Run model-free feedback-summary, feedback-export or feedback-validate commands."""

import sys

from adaptvlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
