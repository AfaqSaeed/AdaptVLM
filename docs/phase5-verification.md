# Phase 5 verification — 2026-11-04

Implemented controlled inference suites, CPU dynamic-int8 and gated CUDA bitsandbytes options,
quality/resource plots, raw measurements, CSV export and a model-free experiment dashboard.
Phase 6's final research report is not included.

## Executed evidence

- Full suite: **65 passed** in 20.64 seconds, retaining all earlier phase coverage.
- Ruff lint/formatting and Git whitespace checks passed.
- Real tiny Idefics3 inference passed for float baseline, float adapter, CPU dynamic-int8
  baseline and merged-adapter dynamic-int8. The suite recorded actual latency, throughput,
  loading time and serialized state bytes for each. No expected speedup or accuracy gain was
  asserted. The adapter in this efficiency test is explicitly test-only, not a trained result.
- A separate CLI smoke suite ran float32 and dynamic-int8 in fresh worker processes. Both
  completed; an NF4-on-CPU configuration was recorded as unsupported with no numerical metrics.
  The resulting benchmark/variant JSON files, worker logs, CSV and three PNGs are stored locally
  under `reports/phase5-tiny-smoke` and ignored by Git.
- The latency plot was visually inspected: labels, units, legend and SOFTWARE TEST ONLY title
  rendered correctly. GPU memory is null on CPU and the GPU plot reports no measured values.
- The dashboard was started briefly on localhost:7863. Its HTTP API returned the measured
  three-variant table, all three Matplotlib plots, metadata and the unsupported variant status,
  then the server shut down. No full browser layout/manual interaction was inspected.
- The HTTP check found a Windows dropdown path separator issue; choices now use portable
  slash-separated paths. The benchmark/dashboard tests were rerun successfully after the fix.
- Unit checks verify median/P95 calculations, failure-inclusive scoring, successful-request
  throughput, warm-up exclusion, reproducible example ordering, capability gates, shared-setting
  enforcement, image/model provenance, adapter merge flags, absent CPU GPU-memory measurements,
  model-load failure reports and dashboard path confinement.

The seven new benchmark tests passed separately after the dashboard correction. Existing Gradio
5 private-event deprecation warnings remain in the full suite (24 warnings). Socket-dependent
checks were run outside the Windows sandbox restriction documented in earlier phases.

## Environment and limits

Windows, Python 3.11.5, CPU PyTorch 2.7.1 and Matplotlib 3.10.9 were used. The tested dependency
snapshot was updated. bitsandbytes was not installed because this environment's PyTorch has no
CUDA support; the optional GPU path is capability-gated rather than silently emulated.

The cached full SmolVLM processor and the small random Idefics3 tests do not establish full-model
quantization quality, throughput or memory use. All measured inference in this phase used
synthetic images and a randomly initialized miniature model. These measurements must not be
presented as research evidence. CUDA int8/NF4 execution, quantized CUDA adapters, real public-data
accuracy, full checkpoint memory use and compute cost remain unverified. No speedup, quality
retention or adaptation benefit is claimed.

The runner now provides the reproducible process and recorded conditions needed to gather real
comparisons once full model weights, real held-out data and appropriate hardware are available.
