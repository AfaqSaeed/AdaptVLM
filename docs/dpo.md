# Phase 4: experimental multimodal DPO

## Compatibility contract

This path pins TRL 0.16.1, Transformers 4.51.3, PEFT 0.15.2 and Datasets 3.5.1.
The [versioned TRL documentation](https://huggingface.co/docs/trl/v0.16.1/en/dpo_trainer)
includes a VLM path using a processor. We inspected the installed trainer and verified its
image-conditioned policy/reference forwards with a real tiny Idefics3 model.

`train_dpo.py --check-only` checks versions, imports, architecture mapping and required trainer
APIs without downloading model weights. It returns 0 for static support or 2 for unsupported
configurations, optionally writing a JSON limitation report via `--output`. Other architectures,
unrecognized dependency versions, FP16 training and existing-adapter continuation are rejected.
Training additionally checks the actual loaded architecture, both reference/policy likelihoods,
and the presence of image tensors before any update. There is no text-only or SFT fallback.

The production path supports SmolVLM's Idefics3 architecture and fresh language-projection LoRA
on the instructed base. The reference is that same frozen base with adapters disabled, not a
second trainable policy. The implementation does not claim SFT-initialized DPO: continuing a
prior SFT adapter would need an independently preserved SFT reference and artifact semantics.

## Preference dataset

Use a fresh eligible-only Phase 2 export. `prepare_preferences.py` resolves A/B through the
persisted randomized display order and retains the actual chosen/rejected answers, original
candidate provenance, source feedback UUIDs and vote mapping. It never derives preferences from
correctness flags, rejected-only answers, both/neither, or correction labels. Identical responses
and opposing votes on the same response pair are errors. Repeated consistent votes are grouped,
not used as an unrecorded weighting multiplier. No eligible pairs is an error.

Conversion writes `pairs.jsonl`, copied images and a hashed manifest into a new directory.
Training revalidates image bytes, pair direction, candidate provenance, duplicate IDs/pairs,
train-only assignments, mock exclusion and current protected image hashes. Both conversion
and training require all known protected manifests, including the original conversion manifests.
Protection uses the same exact RGB/EXIF-oriented checks and limitations as SFT. Re-export and
reconvert when new protected datasets become known; unknown or near-duplicate images cannot
be reliably excluded automatically. All outputs default to Git-ignored data/outputs paths.

## Correct image-conditioned objective

The standard sigmoid DPO objective uses the chosen/rejected completion log-probability sums
under the policy and reference, scaled by positive beta. There is no SFT mixture. LoRA trains
only language-model projections; the vision encoder and original base weights remain frozen.

`encode_pairs` uses the verified assistant-only SFT collator separately for each answer. It
requires identical image tensors and prompt token IDs, preserves the template's exact assistant
and end-of-turn tokens, and rejects pairs that tokenize identically. Length limits fail early
rather than truncate image placeholders or remove answer targets.

A narrow pinned-version DPOTrainer subclass bypasses its dataset-preparation hook to consume
these validated token sequences. The actual TRL preference collator, concatenated multimodal
forward, disabled-adapter reference calculation, sigmoid loss and Trainer optimizer are used.
This avoids appending a second EOS after the template has already ended the turn. It is not a
replacement objective. `max_length`, `max_prompt_length` and `max_completion_length` are all
None inside TRL; the validated external length limit remains enforced. `remove_unused_columns`
is False so the trainer cannot drop pixel tensors. Padding-free/text-only paths are not enabled.

Each loss is checked for finiteness, and final trainable weights and aggregate training loss
must be finite. Failure leaves no completed training metadata. A partial `trainer` directory
may remain for diagnosis; choose a new output path on retry. Successful output contains separate
adapter weights/configuration and the same reloadable provenance format as SFT, with DPO method,
reference policy, dependency gate results, training coverage, loss/reward logs and data hashes.
No model-quality conclusion follows from training reward or loss alone.

## Evaluation and comparison with SFT

DPO adapters use the existing `compare_adapter.py` path: identical held-out data, pinned base,
generation settings, no training-image overlap, category scores and improved/degraded examples.

`compare_methods.py` additionally requires explicit SFT/DPO metadata, the same base identity,
training precision/prompt, source mixed-feedback export, image/question coverage, example counts,
seed, epochs, batch/accumulation settings, LoRA capacity and sequence limit. Missing or unequal
conditions generate a limitation report before inference, with no misleading winner ranking.
Defaults are not automatically comparable: align experimental budgets explicitly. Learning rates
may differ and are retained in the report; choose them on validation, never held-out test scores.

When eligible, the script runs both baseline/adapter comparisons on the same split and summarizes
the change from SFT predictions to DPO predictions. The nested comparison's baseline labels mean
SFT in this context. This is a matched-coverage observational comparison; human corrections and
pairwise votes are distinct information, so it cannot alone prove superiority of an objective.
No metrics are fabricated when real data or weights are unavailable.

## Known limits

Tiny CPU multimodal optimization, reference freezing, save/reload and matched comparison are
exercised by tests. Full SmolVLM weights, genuine preference training, CUDA/bfloat16 memory use,
large datasets, distributed training, quantization and DPO continuation from SFT are unverified
or unsupported. Preprocessed images are materialized in memory; start with small feedback sets.
The pinned internal hook deliberately fails the version gate after dependency changes until
compatibility is rechecked. Phase 5 efficiency research is not included.
