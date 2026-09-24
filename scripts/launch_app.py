"""Run the installed AdaptVLM package from the repository root."""

import sys

from adaptvlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["app", *sys.argv[1:]]))
