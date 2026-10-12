# Phase 3 verification â€” 2026-11-04

Implemented correction-to-SFT conversion, language-projection LoRA training, separate adapter
save/load with provenance validation, matched baseline/adapted evaluation and regression reports.
Phase 4 was subsequently implemented after approval; see [Phase 4 verification](phase4-verification.md).

## Executed checks

- Full suite: **50 passed** in 14.79 seconds, including existing Phase 1/2 checks.
- Ruff lint and formatting checks passed.
- `prepare_sft.py`, `train_sft.py` and `compare_adapter.py` command help passed.
- Existing mock evaluation still produced a software-test-only report.
- Ten new tests cover SFT eligibility, fresh protected-manifest checks, duplicate/conflicting
  corrections, exclusion of preference winners and correct-only labels, invalid settings,
  overlength rejection, assistant-boundary rejection, dataset integrity and missing manifests.
- A real tiny Idefics3 model and real Idefics3 processor/tokenizer were initialized locally with
  random weights. The production training function performed three AdamW updates with PEFT
  LoRA, saved adapter-only weights, and reloaded them through the production inference backend.
  Assertions verified nonzero LoRA updates, unchanged original parameters, frozen reloaded
  parameters, assistant/end-token supervision, image inputs, and padding masks.
- That tiny adapter completed baseline and adapted inference/evaluation on the same disjoint
  image, producing the three comparison reports in a temporary test directory. The comparison
  was marked software-test-only. Adapter corruption and wrong-base configuration were rejected.
- Accumulated updates with unequal target lengths matched a full-batch update within numeric
  tolerance, including a partial accumulation window.
- The cached actual SmolVLM processor was also checked offline against the synthetic shape
  image. It produced 1,147 input tokens; the only supervised IDs were the assistant answer,
  turn-ending token and trailing newline. Full pretrained model weights were not needed for
  this processor/masking check.

The tiny checkpoint test initially exposed unsupported tokenizer segment IDs being forwarded
to Idefics3; inference and training now remove those IDs. The corrected full pipeline passed.
The existing Gradio 5 private-event deprecation warnings remain (24 warnings); Gradio is pinned
below version 6. Full-suite socket-dependent UI checks ran outside the Windows sandbox.

## Environment and limits

Python 3.11.5, PyTorch 2.7.1+cpu, Transformers 4.51.3 and PEFT 0.15.2 were used. PEFT was installed
in the isolated project environment; CI now installs the training extra as well. The configured
CPU PyTorch cannot exercise the machine's NVIDIA GPU.

These are software tests with synthetic images, fabricated feedback provenance and a miniature
randomly initialized VLM, not an adaptation study. No public-data accuracy, feedback efficiency,
speedup, generalization or forgetting claim follows from them. No test-generated model weights
or training feedback are committed. The real SmolVLM checkpoint was not fine-tuned; the earlier
weight-download limitation and absence of genuine feedback prevent a credible research run.
CUDA float32/bfloat16 training and memory behavior are unverified. Qwen training, FP16 training,
QLoRA, adapter resume and multimodal DPO are explicitly unsupported in this phase.

Use the documented fresh-feedback export, SFT preparation, training and comparison commands
with real data and available weights for a research baseline. Choose settings on validation
experiments and freeze them before using the held-out test split.
