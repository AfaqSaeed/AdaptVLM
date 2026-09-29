# AdaptVLM — Phases 1–3

A local, configurable VLM inference app and reproducible short-answer baseline evaluator.
Phases 1–3 provide baseline inference/evaluation, human feedback collection, and correction-driven
LoRA adaptation with matched before/after reports. DPO and efficiency research remain deferred.

## Install (Python 3.11+)

From the repository root, create an isolated environment; do not reuse a global ML installation.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,model,app]"
.venv\Scripts\Activate.ps1
```

On Linux/macOS use `source .venv/bin/activate`. Install an appropriate PyTorch build for your
hardware before the project extras when necessary. For offline plumbing only install `.[dev]`.
Dependencies are constrained in `pyproject.toml`; each report records installed versions.
`requirements-tested.txt` records the tested Windows environment; it is not a cross-platform lock.

## Run

```powershell
python scripts/run_inference.py --image examples/synthetic/2.png --question "What color is the shape?"
python scripts/evaluate.py
python scripts/launch_app.py
```

The default downloads SmolVLM-256M once into the Hugging Face cache and runs float32 CPU
inference. Set `HF_HOME` to `.cache/huggingface` to keep downloads within this workspace.
The app binds to localhost:7860 with public sharing disabled and reuses one model instance.
For GPU use set `device: cuda` and `dtype: float16` in a copy of `configs/model.yaml`.
Use `--config configs/qwen.yaml` only on hardware with sufficient memory.
Model load failures are explicit; inference CUDA OOM becomes a failed prediction with guidance.

Offline smoke run (no model downloads, constant answer):

```powershell
python scripts/run_inference.py --config configs/mock.yaml --image examples/synthetic/2.png --question "Color?"
python scripts/evaluate.py --config configs/mock.yaml --output reports/smoke.json
python scripts/launch_app.py --config configs/mock.yaml
python -m pytest
ruff check .
```

## Dataset and scoring

The four bundled shape images are **synthetic software fixtures, not research evidence**.
The default report is therefore marked `software_test_only`, even with real model weights.
Mock runs are always marked the same way. Nothing reports mock accuracy as model quality.

For real evaluation, obtain official VQAv2 questions, annotations and COCO images from
[VisualQA](https://visualqa.org/download.html), respecting its
[annotation terms](https://visualqa.org/terms.html) (CC BY 4.0) and the separate
[COCO image terms](https://cocodataset.org/#termsofuse). Images are not redistributed here.
Use the local importer (avoids executing remote dataset scripts):

```powershell
python scripts/prepare_vqav2.py --questions data/v2_OpenEnded_mscoco_val2014_questions.json --annotations data/v2_mscoco_val2014_annotations.json --images data/val2014 --output data/vqa-small --limit 100 --seed 42
python scripts/evaluate.py --manifest data/vqa-small/manifest.json --split test --output reports/vqa-baseline.json
```

The importer selects one question per image deterministically and assigns project train,
validation and held-out test partitions (roughly 60/20/20); these are not official VQA splits.
Use validation to select settings, then freeze them before evaluating test. Never train on test.
The loader rejects decoded-image duplicates across partitions. Near-duplicates and pretraining
contamination are not detected. Small sample results do not establish generalization.

Reports contain dataset/image hashes, model configuration and resolved model commit, seed,
software/hardware information, per-example answers/failures, category scores, load time,
latency and peak allocated CUDA memory (null on CPU). Categories preserve source question types.
Scoring uses exact and conservative normalized short-answer reference matching, **not official
VQA consensus accuracy**. It is unsuitable for judging unrestricted prose; inspect predictions
and failure cases. The default prompt requests short answers. There is no LLM judge.
Latency includes processing and decoding, excludes loading, and includes the cold first request.
This is baseline instrumentation, not a speedup benchmark. CUDA memory is allocated peak, not
whole-system usage. Seeds cannot guarantee bit-identical output across hardware/software.
Pin `revision` to the resolved commit from a report for repeat runs; `main` is a convenient
first-run default. Preserve the dataset manifest, image files and dependency versions too.

## Model choices and limits

[SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
is Apache-2.0 and is the practical low-memory starting point; capability must be measured.
[Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) is an optional larger baseline;
check the license at the exact checkpoint revision before use or redistribution. Its unquantized
weights plus image activations are unsuitable for the detected 4 GB GPU. Quantization remains unsupported; Phase 3 adds adapter loading and SmolVLM LoRA training. Arbitrary VLM architectures are deliberately rejected;
processor compatibility must be implemented and tested explicitly. Phase 3 tests assistant-only
masking for its supported SFT path; future multimodal DPO needs separate compatibility checks.

See [verification notes](docs/verification.md). Model weights, datasets, private data and reports
are ignored by Git. No fabricated research results are included.


## Phase 2: collect and export feedback

```powershell
python scripts/launch_app.py --comparison-config configs/comparison.yaml --protected-manifest data/vqa-small/manifest.json
```

Omit `--protected-manifest` if you have not prepared that dataset. The evaluation configuration's
manifest is always registered automatically. Repeat `--protected-manifest` for every additional
known dataset. The alternate prompt config reuses the primary model's loaded weights and lock;
a different checkpoint loads a second model and needs additional memory.

The app now has three tabs:
- **Answer and correct:** generate an answer, mark it correct/incorrect, optionally supply a
  corrected answer. Uploaded image/question changes invalidate the current review.
- **Compare responses:** generate two candidates, shown in randomized order; choose A, B,
  both acceptable, or neither. Identical answers permit only both/neither. No preference is
  invented or converted from correctness labels.
- **Review and export:** inspect collection counts and recent saved reviews, then download a
  validated ZIP containing JSONL records, a manifest and image files.

For an offline UI smoke run:

```powershell
python scripts/launch_app.py --config configs/mock.yaml --comparison-config configs/mock.yaml
```

Mock candidates are identical, so use both/neither in this smoke mode. All mock feedback is
excluded from training-eligible exports. The standalone database commands require no model:

```powershell
python scripts/manage_feedback.py feedback-summary
python scripts/manage_feedback.py feedback-export --output data/exports/audit-001
python scripts/manage_feedback.py feedback-export --eligible-only --output data/exports/eligible-001
python scripts/manage_feedback.py feedback-validate --output data/exports/eligible-001
```

Use a new directory for each export. Audit exports contain held-out and otherwise ineligible
records and **must not be passed to training**. Eligible exports contain validated raw feedback,
not SFT/DPO training examples; Phase 3 converts human corrections into SFT data; preference conversion remains deferred.

See [feedback design and data rules](docs/feedback.md) and
[Phase 2 verification](docs/phase2-verification.md).


## Phase 3: supervised adaptation

Install the training extra into the same isolated environment:

```powershell
python -m pip install -e ".[dev,app,model,training]"
```

Collect actual incorrect-answer corrections on train-assigned images. Mock/synthetic feedback,
preferences, and merely confirmed-correct answers are not used by this SFT converter. Export a
fresh eligible bundle and supply every known protected dataset at conversion and training:

```powershell
python scripts/manage_feedback.py feedback-export --eligible-only --protected-manifest data/vqa-small/manifest.json --output data/exports/eligible-002
python scripts/prepare_sft.py --feedback data/exports/eligible-002 --protected-manifest data/vqa-small/manifest.json --output data/sft-001
python scripts/train_sft.py --data data/sft-001 --model-config configs/model.yaml --training-config configs/sft.yaml --protected-manifest data/vqa-small/manifest.json --output outputs/sft-run-001
python scripts/compare_adapter.py --adapter outputs/sft-run-001 --config configs/model.yaml --manifest data/vqa-small/manifest.json --split test --output reports/comparison-001
```

These are real-model commands, not mock demonstrations. They require downloaded base weights,
genuine eligible corrections and adequate compute. The default CPU float32 config works with
the code path but may be slow; CUDA requires a CUDA-enabled PyTorch installation. Use float32
or supported CUDA bfloat16 for training. FP16 training, QLoRA, Qwen training, adapter resuming,
and DPO are explicitly unsupported. Freeze hyperparameters before running the test split.

The comparison writes `baseline.json`, `adapted.json` and `comparison.json`, including category
changes and improved/degraded predictions. It runs models sequentially, uses identical settings,
pins the base checkpoint, and rejects train/evaluation image overlap.

For adapted single-image inference or the app, use `configs/adapted.example.yaml` after setting
its `adapter_path` to your completed run. It loads the original base plus the separate adapter;
base weights are never copied into the adapter output. See [SFT design and limits](docs/sft.md)
and [Phase 3 verification](docs/phase3-verification.md).
