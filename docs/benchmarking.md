# Phase 5: controlled efficiency benchmarking

## Scope and configuration

`benchmark.py` runs one sequential inference configuration per fresh subprocess. It supports a
baseline, a separate LoRA adapter, quantized baseline and quantized adapted model. The benchmark
batch size is explicitly 1; this is request throughput, not batched serving throughput. A suite
must share model ID, family, configured revision, backend, device, seed, max generation tokens
and prompt suffix. Precision, quantization and adapter may vary. Use separate suites for device
comparisons; this version does not treat CPU and GPU timings as controlled alternatives.

Manifest and model-config file paths are relative to the suite YAML. Adapter overrides are also
relative to that YAML. Model YAML fields otherwise retain their existing inference semantics.
The suite accepts a test/validation split, warm-up request count, measurement repeats, order seed,
CPU `torch_threads` (default 1), and uniquely named variants. Configurations and per-repeat example
order are shuffled reproducibly. There is no automatic tuning using test results. Select settings
on validation and freeze the suite before a held-out experiment.

The runner validates dataset hashes and adapter train/evaluation image disjointness. Results
record resolved Hub revision or local base fingerprint, adapter hashes, source image sizes,
processor image settings, model precision, quantization details, software versions, CPU/platform,
PyTorch thread counts, CUDA runtime and GPU identity/capability/total memory where available.
The comparison gate rejects unequal resolved checkpoints, preprocessing, environments, example
order or datasets. Only completed variants enter comparison_scope; omitted variants retain their
reason and status. A comparison with fewer than two completed variants is ineligible.

## Quantization paths

CPU `dynamic_int8` requires real SmolVLM/Idefics3, float32 and an active PyTorch quantized engine.
It converts only language-model Linear layers to dynamic qint8. Vision, connector, embeddings
and output head remain floating point. For an adapted model, PEFT is safely merged first, then
the language layers are quantized. This is an inference-only merged model; no merged base weights
are saved. Merge and conversion costs are included in loading time. The number of converted
layers and the merge choice are recorded. Baseline and adapted tiny-model inference were tested.

CUDA `bnb_int8` and `bnb_nf4` use the pinned bitsandbytes 0.45.5 with Transformers
[BitsAndBytesConfig](https://huggingface.co/docs/transformers/v4.51.3/en/quantization/bitsandbytes).
The gate requires CUDA PyTorch, a GPU with sufficient capability, the expected package version,
and supported bfloat16 hardware when requested. The device map is explicitly GPU 0; there is no
silent CPU offload. Vision/connector/output-head conversion is skipped in loading settings.
NF4 uses double quantization and the configured compute dtype. An adapter can be attached after
loading the quantized base. These GPU paths are implemented but unverified on this CPU-only
PyTorch installation; kernel/model incompatibilities are reported as failures, not replaced by
another backend. Quantized SFT/DPO remain explicitly unsupported.

## Measurement definitions

Images are read into memory before warm-up. The same configured warm-up count runs for every
variant and is excluded from measured samples. A warm-up error invalidates the variant.

Each measured request uses the existing serialized inference engine. GPU synchronization occurs
around measured generation, and peak allocator stats are reset per request. Latency includes
processor/tokenization, generation and decoding; it excludes file I/O and the engine's initial
image orientation/RGB normalization. Successful-request median and P95 use actual latency
samples, with NumPy linear interpolation for P95. Failed attempts do not become attractive low
latencies: they stay in failure counts and are incorrect in the accuracy denominator. A run with
recoverable request failures is partial and excluded from comparison.

Successful requests per second equals successes divided by the whole measurement-loop wall
time, including image normalization, scoring and loop overhead, but excluding warm-up/loading.
This is distinct from inverse median latency. Accuracy is averaged over all repeated attempts;
repetition does not increase the number of independent evaluation examples. Raw answers,
references, category, repeat index, errors and timing samples are retained. Category scores and
repeated-answer consistency are recorded.

Peak GPU allocated/reserved bytes are PyTorch allocator values including resident weights, not
whole-device usage; CPU values are null, never zero. A separate serialized model-state byte
measurement includes packed quantized weights and is made after timing. It is NOT peak process
RAM or GPU memory and has its own plot/label. State serialization itself uses extra temporary
memory and is not part of the measured inference loop.

Loading time includes weights, adapter loading/merging and quantization, and can include network
or filesystem-cache effects. It is one observation, not a cold-cache claim. Adapter training time
is taken from saved SFT/DPO metadata when present; otherwise null. Compute cost is null because
no hardware price or energy measurement is supplied. No estimated costs are invented.

## Reports and visualization

Every suite publishes `benchmark.json`, raw `run-N.json` results, replayable `request-N.json`
inputs and worker logs. Failed/unsupported variants retain no numerical metrics. Successful
reports include explicit measurement assumptions, comparison conditions, run order and
software-test-only flags. Existing output directories are rejected. A failed worker does not
prevent the next variant from being recorded. Exit code 2 signals incomplete coverage.

Matplotlib produces latency-versus-quality, GPU-allocation-versus-quality and serialized-state-
size-versus-quality PNGs, plus `summary.csv`. Missing GPU data stays missing, including an empty
GPU plot on CPU. Figures are labeled software-test-only if synthetic/mock artifacts are involved.
The model-free Gradio dashboard reads suites from a chosen report root, blocks paths outside it,
and exposes results, plots, category performance, settings and hardware metadata. It does not
load weights, train models, or make automatic speedup/significance claims.

## Limitations

Measurements on real public data/full checkpoint/GPU remain to be collected. Tiny synthetic
runs prove the software path only. Sequential process isolation prevents cross-model allocator
residue, but does not control background applications, thermal state, filesystem cache or clock
frequency. Thread count, random ordering and repeated requests improve repeatability, not full
experimental control. No confidence intervals, energy measurements, multi-GPU serving or
statistical speedup claims are implemented. Near-duplicate dataset leakage remains the previously
documented limitation. A final scientific report belongs to Phase 6.
