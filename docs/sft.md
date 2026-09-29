# Phase 3: correction-driven LoRA

## Supported architecture

Training supports the Idefics3 architecture used by the selected
[SmolVLM-256M-Instruct checkpoint](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct),
with Transformers 4.51.3 and PEFT 0.15.2. The implementation uses
[PEFT LoRA](https://huggingface.co/docs/peft/v0.15.0/package_reference/lora) and the model's
own processor/chat template. Other model families fail explicitly before training. No claim
of arbitrary-VLM trainer compatibility is made.

`configs/sft.yaml` controls learning rate, batch size, gradient accumulation, epochs, LoRA rank
and alpha, target projection names, maximum sequence length, random seed and gradient clipping.
Target names are expanded only within the language model; vision encoder and all original
weights are frozen. No biases or embedding layers are trained. Quantization, optimizer resume,
SFT from preference winners, preference optimization and automatic hyperparameter search are
not implemented. CPU float32 is the reproducible local path; CUDA float32/bfloat16 support is
implemented but not hardware-tested here. FP16 training is rejected because this loop does not
implement gradient scaling. The detected 4 GB GPU and current CPU-only PyTorch are not evidence
that a real checkpoint can be fine-tuned on that GPU.

## Data conversion and protection

`prepare_sft.py` accepts only an independently validated eligible-only Phase 2 bundle. It
selects incorrect-answer records with a nonblank human replacement. It never uses confirmed
model answers or pairwise winners as SFT targets. Duplicate image/question/correction triples
are consolidated while retaining source feedback UUIDs. Conflicting corrections fail conversion
and require human resolution upstream. No eligible corrections is an error, not an empty run.

The output is a new directory with `examples.jsonl`, images and a manifest containing content
hashes, source feedback fingerprint, protected-manifest fingerprints and counts. Conversion is
staged, validated and published without overwriting existing data. Training independently
validates the bundle, image bytes, train-only labels, duplicates, counts and fingerprints.

Both commands require repeatable `--protected-manifest` arguments. Include every known dataset;
training must receive every manifest used for conversion, plus any newer held-out datasets.
Validation/test and synthetic images are excluded using decoded RGB and EXIF-oriented hashes.
A changed registered manifest requires reconversion. Export fresh feedback before conversion;
old artifacts cannot automatically learn about later database edits. Exact-pixel matching does
not detect resized/recompressed near-duplicates or unknown held-out datasets.

Validation and test images are read only for exclusion checks. Their answers do not enter the
optimizer, early stopping, or configuration selection. Training uses fixed epochs; select
settings with separate validation experiments and freeze them before held-out test evaluation.

## Multimodal labels and optimization

The collator processes both the user-only generation prompt and the full user/assistant
conversation with the same images. It requires the prompt IDs to be an exact prefix of full
conversation IDs. Unexpected template/tokenization boundaries fail instead of guessing a mask.
User/header/image tokens and batch padding receive label -100. Assistant response and turn-ending
tokens are supervised. Image tensors and pixel attention masks pass through to the model.
Overlength examples fail before optimization; truncation never silently removes image tokens or
assistant targets. Increase the limit only within your model/compute budget or shorten examples.

Every example is preflighted before any optimizer step. Gradient accumulation weights each
microbatch loss by its number of supervised next-token targets, including an incomplete final
window. This avoids over-weighting short replies. Nonfinite loss/gradients fail the run. The loop
uses AdamW with no weight decay and a constant learning rate, deterministic seeded shuffling,
and gradient clipping. Cross-hardware bitwise reproducibility is not promised.

## Adapter artifacts and inference

A completed run contains only adapter weights/configuration plus `training_run.json`. The
metadata is written last and records the resolved base Hub commit (or a local checkpoint content
fingerprint), model/training configurations, data fingerprint, training image hashes, package
versions, hardware, elapsed time, loss history and adapter file hashes. The loader requires
completed metadata, validates adapter file integrity and rejects mismatched base identity or
revision. Local bases must remain at the recorded location with the same contents.

Set `adapter_path` in a model YAML to use the adapter in inference, evaluation or the feedback
app. A `main` revision with an adapter resolves to the recorded training commit rather than a
newer Hub checkpoint. Adapter file hashes are included in adapted evaluation and newly collected
feedback provenance. Standard third-party adapters lacking AdaptVLM metadata are deliberately
not accepted by this path. Hashes detect accidental modification, not malicious rewriting of
both weights and metadata.

Training saves no base weights, embeddings or optimizer state. A failure during save may leave
an incomplete output directory; it is not loadable as a completed adapter. Choose a new output
path on retries. There is no automatic checkpoint resume.

## Matched evaluation and regressions

`compare_adapter.py` preflights the held-out split against training image hashes, then evaluates
the pinned base and its adapter sequentially with identical generation settings. It releases
the baseline before loading the adapted model to reduce peak resident memory. Reports preserve
all per-example predictions and failures. Comparison rejects mismatched datasets, item IDs,
questions, references, categories, model settings, revisions and environments.

The comparison includes normalized-accuracy delta, per-category before/after scores, improved
and degraded predictions, and unchanged counts. Existing exact/normalized short-answer scoring
limitations still apply. Failures remain incorrect in the denominator. These are descriptive
regression checks, not statistical significance tests or proof of catastrophic forgetting.
Use sufficiently broad, disjoint evaluation categories to investigate generalization. No
improvement or speedup is assumed. Tiny synthetic CPU tests are software checks only.
