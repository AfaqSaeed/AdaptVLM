"""One benchmark configuration per process to avoid cross-model allocator state."""

import json
import sys
from pathlib import Path
from adaptvlm.benchmarking.runner import run_variant
from adaptvlm.evaluation.runner import write_report

if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    write_report(run_variant(request), sys.argv[2])
