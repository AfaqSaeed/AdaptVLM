# AdaptVLM: Feedback-Driven, Compute-Efficient Vision-Language Adaptation

**Technical research report — 4 November 2026**  
**Evidence status: implemented platform and software validation; research hypotheses remain untested.**

## Abstract

AdaptVLM implements a local experimental pipeline for image-question inference, human corrections
and preferences, supervised LoRA adaptation, experimental multimodal DPO, and controlled efficiency
measurement. It preserves feedback provenance, protects registered held-out images, and compares
checkpoints with shared evaluation settings. A fresh validation run passed 65 automated tests,
including actual optimization and adapter reload on tiny randomly initialized Idefics3 models.

No full pretrained checkpoint was fine-tuned, no genuine human-feedback study was conducted,
and no public-dataset accuracy or GPU efficiency results are available. The retained benchmark
contains only two synthetic test images and a miniature random model. Its observations establish
that measurement and reporting work; they do not establish feedback efficiency, generalization,
quality retention or superiority of DPO. This report records the methods, available evidence,
limitations and the protocol needed to investigate those questions without inventing results.

## 1. Research questions

| Hypothesis | Required experiment | Current conclusion |
| --- | --- | --- |
| H1: Carefully selected corrections improve feedback efficiency over random examples | Match feedback/compute budgets and vary example selection | Not tested; no real correction-learning curve |
| H2: Targeted adaptation preserves unrelated capabilities | Compare targeted and disjoint category scores before/after adaptation | Not tested; no broad public-data regression study |
| H3: Quantization preserves useful quality while reducing resource requirements | Evaluate the same trained checkpoint and data at different precisions | Not tested for a useful pretrained VLM; tiny software measurements only |
| H4: Preference optimization adds value beyond supervised adaptation | Match training coverage, annotation effort and compute; compare held-out results | Not tested scientifically; matched tiny-model pipeline exercised |

The platform makes these experiments possible, but successful execution is not evidence that
any hypothesis is true. In particular, a falling training loss or increasing DPO reward margin
would not by itself demonstrate better visual-question-answering performance.

## 2. Models, data and baselines

### Model choices

The practical initial checkpoint is
[HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct),
whose model card identifies the Idefics3 architecture and Apache-2.0 license. It was selected
for a low-memory starting point rather than an assumed accuracy advantage. The workstation has
an RTX 3050 Laptop GPU with 4 GB VRAM, but the installed PyTorch build is CPU-only. Full model
weight downloading previously stalled; cached processor/tokenizer files were sufficient to check
prompt masking, not to run the pretrained VLM.

Qwen2.5-VL-3B is an optional inference configuration, not a tested adaptation baseline here.
Its checkpoint license must be checked at the exact revision before use. Training support is
intentionally restricted to the SmolVLM/Idefics3 path.

Executed optimization tests instead instantiate a random miniature Idefics3 model: a one-layer
32-dimensional language model, a one-layer 16-dimensional vision encoder, 16-by-16 processed
images and a small test tokenizer. These are deliberately incapable of serving as substitutes
for pretrained-model quality measurements. The mock backend's constant answer is a separate
plumbing test and is never a real model baseline.

### Available and intended datasets

The checked-in [synthetic manifest](../examples/synthetic/manifest.json) has four 128-by-128 shape
images with color questions: one train, one validation, two test examples, all in the attribute
category. It tests storage, partitioning, inference and scoring. It does not cover counting,
spatial relations, object recognition or scene understanding sufficiently for research.

The intended public starting dataset is VQAv2. The local importer shuffles annotations with a
fixed seed, selects one question per unique image, retains all annotated answers, and assigns
project train/validation/test partitions of roughly 60/20/20. These are project splits, not the
official VQA test split. Source question types become categories; a broader research taxonomy
would need explicit mapping and review. No real VQAv2 subset was evaluated in the present work.
VQA annotations are CC BY 4.0 under the [official terms](https://visualqa.org/terms.html);
COCO image conditions are separate. No downloaded images or private feedback are redistributed.

### Baselines and comparators

The implemented comparison set is the instructed base, an SFT LoRA adapter, a DPO LoRA adapter,
and inference-quantized base/adapted variants where supported. Full-model results for every
member remain **not measured**. Tiny tests cover float32, CPU dynamic int8, adapter inference,
merged-adapter int8, and matched SFT/DPO evaluation. GPU int8/NF4 configurations produce explicit
unsupported/failed statuses on unsuitable environments rather than fallback measurements.

## 3. Methodology

### Feedback and leakage control

SQLite stores immutable image/question/response snapshots, model and generation provenance,
UTC timestamps and optional random session UUIDs. Correctness feedback may include a replacement.
Pairwise feedback is a genuine vote over two displayed responses, with randomized order saved
alongside the vote. Both/neither remain distinct from directional preferences. Duplicate
submission to one request is rejected; conflicting labels/pair votes fail dataset conversion.

Uploads default to unassigned. Training exports require explicit train assignment and exclude
mock/synthetic and registered validation/test images. Current protected manifests are checked
again during conversion and training. Decoded RGB and EXIF-oriented pixel hashes detect exact
image overlap, including later registration, but not resized/recompressed near-duplicates or
unknown held-out sets. Exports are snapshots: regenerate them when protection knowledge changes.

### Supervised adaptation

The SFT converter selects incorrect answers with human replacements, not preference winners or
merely confirmed model answers. [LoRA](https://arxiv.org/abs/2106.09685) provides trainable low-rank
updates while original parameters remain frozen. The implementation targets language-model
projection layers and freezes vision, embeddings and original base weights.

Both prompt-only and full conversations pass through the same processor. The prompt token IDs
must be an exact prefix of the full sequence; otherwise training fails. User/header/image and
padding tokens are masked from loss, while assistant and turn-ending tokens are supervised.
Overlength examples are rejected rather than silently truncating multimodal sequences. Gradient
accumulation is normalized by supervised next-token count, including the final partial window.

### Preference adaptation

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) uses chosen/rejected completion
likelihoods relative to a frozen reference. This platform uses the actual sigmoid DPO objective
in pinned TRL 0.16.1, with beta 0.1 by default. Image tensors and exact completion boundaries are
preserved through a narrow, version-gated preprocessing hook; TRL handles the multimodal forward,
reference likelihoods, loss and optimization. There is no text-only or SFT fallback.

The DPO policy starts a fresh LoRA adapter on the instructed base; disabling its adapters yields
the frozen reference. It does not continue an SFT adapter. SFT-versus-DPO comparisons therefore
compare two separately adapted policies from the same base, not sequential SFT-then-DPO training.
The guarded comparison requires matching feedback export, image/question coverage, example
counts, base identity, seeds and training budgets. Different supervision signals still prevent
a simple causal claim that one objective is better.

| Default setting | SFT | DPO |
| --- | --- | --- |
| Learning rate | 0.0001 | 0.00005 |
| Batch / accumulation | 1 / 4 | 1 / 1 |
| Epochs / seed | 1 / 42 | 1 / 42 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |
| Targets / maximum sequence length | q_proj, v_proj / 2048 | q_proj, v_proj / 2048 |

These defaults are not a controlled method-comparison setting because accumulation differs.
Align budgets deliberately, select hyperparameters on validation, and freeze them before test
analysis. No early stopping or hyperparameter selection uses held-out test answers.

### Evaluation and efficiency measurement

Quality metrics are exact reference match and conservative normalized short-answer match
(lowercase, whitespace normalization, terminal punctuation removal), with category scores and
per-example improvement/regression analysis. These are **not official VQA consensus scores**
or semantic judgments of unrestricted prose. Failed predictions count as incorrect.

Efficiency suites use batch size 1, shared data/generation settings, recorded thread count,
seeded order, separate warm-up and measurement, and sequential fresh worker processes. Successful
request latency includes processor, generation and decoding; wall-clock throughput also includes
in-memory normalization/scoring overhead. P95 uses linear interpolation. GPU memory is allocator
allocated/reserved memory, not total device consumption. Serialized state bytes are a separate
measurement, not RAM usage. Loading may include cache/network effects. Training time is taken
from adapter metadata; compute price/energy cost is unknown and remains null.

Adapters are saved separately with data/checkpoint hashes and training metadata. Baseline and
adapted evaluations must share the resolved base, data, settings and environment, and must not
contain training-image overlap. [Detailed SFT](../docs/sft.md), [DPO](../docs/dpo.md), and
[benchmark](../docs/benchmarking.md) specifications define the implemented checks and limits.

## 4. Available results — software validation only

A fresh Phase 6 run completed **65 tests with zero failures, errors or skips**. The suite exercises
real tiny-model SFT and DPO optimization, unchanged base weights, adapter save/reload, frozen DPO
reference likelihoods, matched evaluation, quantized inference and leakage/integrity checks.
Twenty-four existing Gradio deprecation warnings remain. These are engineering acceptance
results, not research performance results.

The retained process-isolated benchmark used two synthetic test images, three repetitions,
one warm-up request, one CPU inference thread, seed 42 and at most two generated tokens. The
machine reported an AMD64 Family 25 Model 80 CPU, Windows, Python 3.11.5, PyTorch 2.7.1 CPU and
quantized engine x86. Each completed variant has six attempts but only **two unique examples**.

| Tiny random model variant | Status | Normalized match | Median / P95 latency (ms) | Successful requests/s | Serialized state (bytes) | Loading (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Float32 | Completed | 0.000 | 8.019 / 8.881 | 124.589 | 76,797 | 5.269 |
| Dynamic int8 | Completed | 0.000 | 8.349 / 9.024 | 120.516 | 54,931 | 5.216 |
| NF4 on CPU | Unsupported | Not measured | Not measured | Not measured | Not measured | Not measured |

**This table is a software instrumentation check, not evidence about SmolVLM quality or efficiency.**
GPU memory is unmeasured, not zero. The int8 serialized state is smaller in this observation,
but its measured median latency is not lower. A single tiny run with uncontrolled machine state
and no repeated independent sessions cannot establish a speed advantage or disadvantage.
Neither model answered the two color questions correctly, so this cannot demonstrate meaningful
quality retention under quantization.

Exact values, package/hardware details, test counts and source fingerprints are preserved in
[the evidence snapshot](evidence/phase6-evidence.json). The original local raw benchmark and
JUnit files are ignored generated artifacts; their SHA-256 values are recorded. The source
snapshot explicitly records an uncommitted working tree: the Git HEAD alone does not identify
all Phase 4–6 work. Dates in Git history are not used as experimental evidence.

## 5. Ablations and failure analysis

### Executed engineering controls

- **Masking and accumulation:** tests verify assistant-only targets, reject unstable boundaries,
  and match accumulated updates to full-batch updates with unequal answer lengths.
- **Reference freezing:** perturbing LoRA changes DPO policy likelihoods while reference
  likelihoods remain unchanged.
- **Data protection:** stale exports, exact-image overlap, conflicting feedback, corrupted
  artifacts and mismatched comparison conditions are rejected in tests.
- **Quantization:** the retained tiny float/int8 run checks measurement and loading paths;
  four-variant tests additionally cover merged-adapter int8. These are not research ablations.

### Research ablations — proposed, not run

| Factor | Proposed controlled contrast | Required readout |
| --- | --- | --- |
| Feedback budget | 16, 64, 256 corrections, where enough eligible data exist | Held-out quality versus unique feedback count and annotation effort |
| Selection strategy | Difficult-example selection versus random selection at equal budgets | H1 difference with repeated seeds and matched compute |
| Adaptation capacity | LoRA ranks 4, 8, 16 with recorded trainable parameters | Quality, training time and unrelated-category regressions |
| Objective | SFT versus DPO on matched image/questions and explicitly recorded annotation effort | H4 differences with the different supervision signals acknowledged |
| Precision | Float baseline/adapted versus supported int8/NF4 counterparts | Quality, latency distribution, GPU memory and loading cost |
| Generalization | Targeted categories versus disjoint held-out categories | Improved and degraded cases, not just aggregate accuracy |

A future protocol should predeclare splits and primary outcomes, use several seeds (for example
17, 42 and 101), and report uncertainty using image-level resampling rather than treating repeated
inference attempts as independent accuracy examples. These analyses are **not implemented or
executed results**. Dataset size and annotation budget must be established before choosing the
final design.

### Observed failures and practical implications

In the retained tiny benchmark, the red-shape question produced `user` and the yellow-shape
question produced an empty decoded answer under both float and int8 configurations. Neither
was a runtime exception; both were scored incorrect. This illustrates why successful execution
is separate from answer quality. The evidence snapshot retains these first-repeat predictions.

Full-weight downloading stalled in Phase 1; no pretrained-model result was substituted. NF4 on
CPU is explicitly unsupported here, not a failed quality score. Tests caught and corrected
unsupported segment IDs forwarded to Idefics3, API session initialization, and Windows report-
path separators. OOM handling, stale-feedback prevention and integrity failures were exercised
as software conditions; they are not a measured full-model failure-rate study.

## 6. Limitations and interpretation

The central research question remains unanswered. There is no genuine feedback corpus, no
trained full-model adapter, no public-data evaluation, no GPU benchmark, no learning curve and
no broad forgetting study. Tiny models, four synthetic images and two-token generations cannot
support claims of practical VLM quality, annotation efficiency or retained capability.

Exact lexical scoring may undercount valid paraphrases and is unsuitable for open-ended answers.
Exact-image protection misses near-duplicates and pretraining contamination. Shared image/question
coverage does not equal matched human annotation effort. CPU thread control and process isolation
do not control thermal state, background load or cache effects. Serialized size does not measure
runtime memory, and no energy/cost estimate is available. CUDA quantization is implemented but
unverified; training quantization, adapter resume and SFT-to-DPO continuation remain unsupported.

The supported conclusion is narrower: AdaptVLM has a modular, tested experimental workflow that
can collect provenance-preserving feedback and execute real multimodal adaptation primitives on
small fixtures. The research claims require new data and experiments, not reinterpretation of
these software checks.

## 7. Reproducibility and next experiment

From the repository root on Python 3.11+, create an isolated environment and install all tested
feature groups. The [Windows dependency snapshot](../requirements-tested.txt) records exact
versions; use a hardware-appropriate PyTorch build for separate GPU experiments.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,app,model,training,dpo,benchmark]"
python -m pytest -q --junitxml=reports/phase6-validation.xml
ruff check .
python scripts/evaluate.py --config configs/mock.yaml --output reports/smoke.json
python scripts/train_dpo.py --check-only
```

The tests create random miniature checkpoints locally; they do not need pretrained weight
downloads. To exercise the retained benchmark procedure from a clean checkout, create a new
local tiny checkpoint and suite, then run the normal benchmark command:

```powershell
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'tests'); from test_training import tiny_checkpoint; tiny_checkpoint(Path('outputs/report-tiny-base'))"
```

Create `outputs/report-tiny-model.yaml`:

```yaml
model_id: outputs/report-tiny-base
device: cpu
dtype: float32
max_new_tokens: 2
prompt_suffix: ''
```

Create `outputs/report-tiny-suite.yaml` (run commands from the repository root):

```yaml
manifest: ../examples/synthetic/manifest.json
split: test
warmup_requests: 1
repeats: 3
order_seed: 42
torch_threads: 1
variants:
  - name: tiny-fp32
    model_config: report-tiny-model.yaml
  - name: tiny-dynamic-int8
    model_config: report-tiny-model.yaml
    overrides:
      quantization: dynamic_int8
```

Run `python scripts/benchmark.py --config outputs/report-tiny-suite.yaml --output reports/new-tiny-run`.
Timings will differ; do not relabel regenerated data as the original observation.

Regenerate the evidence snapshot from an explicitly chosen benchmark and JUnit result:

```powershell
python scripts/build_report_evidence.py --benchmark reports/new-tiny-run/benchmark.json --junit reports/phase6-validation.xml --output reports/evidence/new-evidence.json
```

The extractor checks sample counts and accuracy against source predictions, requires a passing
non-skipped test suite, records input hashes, and hashes implementation/configuration/fixture
files. It does not invent research results or automatically rewrite this narrative.

For a real study: obtain licensed VQAv2 inputs; prepare and freeze disjoint splits; download and
pin the base checkpoint; establish validation quality; collect genuine train-only corrections
and preferences; export and validate them; train with fixed recorded budgets; evaluate frozen
checkpoints on test once the protocol is fixed; then run repeatable efficiency suites. Exact
commands appear in the [README](../README.md). Archive configs, hashes, adapters and raw reports,
and inspect both improved and degraded examples. Publish a revised research report only when
those data support it.
