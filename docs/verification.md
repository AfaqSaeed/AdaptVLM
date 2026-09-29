# Phase 1 verification â€” 2026-11-04

Environment: Windows, Python 3.11.5, NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.
The isolated environment has PyTorch 2.7.1, Transformers 4.51.3 and Gradio 5.50.0.
Exact installed packages are recorded in `requirements-tested.txt` (Windows snapshot).

Verified:
- `python -m pytest -q`: **11 passed** in 6.95 seconds.
- `ruff check .`: passed.
- Mock CLI evaluation wrote `reports/smoke.json`, with dataset/image fingerprints,
  predictions, categories, measurements and `software_test_only: true`.
- Gradio app creation and localhost launch succeeded; HTTP GET returned 200 and the
  server shut down cleanly. Browser rendering and manual image upload were not inspected.
- Tests cover config errors, model reuse, image conversion, empty inputs, scoring,
  failure denominators, deterministic reports, split leakage, path traversal,
  deterministic VQAv2 import, image chat template, prompt-token trimming and CUDA OOM handling.
- CUDA OOM test uses a simulated exception; it does not prove GPU execution.

Not verified:
- Actual SmolVLM generation: processor/tokenizer download succeeded, but the model-weight
  download stalled with a zero-byte incomplete cache file. No real-model report was produced.
- Qwen inference, GPU inference/memory, real VQAv2 accuracy, or any adaptation/training.
- The VQAv2 importer was tested with locally generated input files, not the full official corpus.

The default synthetic dataset only validates plumbing. Its metrics, and all mock outputs,
must not be cited as research evidence. A real baseline requires the documented checkpoint
and VQAv2 preparation commands to complete on a network/hardware environment that supports them.

The initial global environment had incompatible Transformers/Hub versions, so an isolated
`.venv` was created. A lint invocation initially traversed the dependency cache; dependencies
were reinstalled from fresh downloads and the cache excluded before successful verification.
Windows sandbox socket restrictions stalled Gradio construction; the successful suite and
HTTP check were run outside that restriction. Stalled test processes were stopped.

Phase 2 was subsequently implemented after user approval; see [Phase 2 verification](phase2-verification.md).
