# Phase 4 verification â€” 2026-11-04

Implemented genuine preference-data conversion, pinned multimodal DPO compatibility checks,
experimental TRL LoRA training, adapter reload/evaluation, and guarded SFT-versus-DPO comparison.
Phase 5 was subsequently implemented after approval; see [Phase 5 verification](phase5-verification.md).

## Executed evidence

- Full pytest suite: **58 passed** in 18.71 seconds; all previous phases still pass.
- Ruff lint and formatting passed, and `git diff --check` found no whitespace errors.
- Static compatibility CLI returned supported for TRL 0.16.1, Transformers 4.51.3,
  PEFT 0.15.2 and Datasets 3.5.1, saving `reports/dpo-compatibility.json` locally.
- Preference-preparation and method-comparison CLI help passed; the existing mock baseline
  evaluation still produced a software-test-only report.
- Actual TRL DPO performed two optimization steps on a tiny randomly initialized Idefics3
  model with image tensors. The adapter was saved, reloaded through the production backend,
  and evaluated on a disjoint image against its base checkpoint.
- Policy and reference likelihoods initially matched. Manually perturbing LoRA weights changed
  policy likelihoods while reference likelihoods remained identical, verifying the disabled-
  adapter frozen-base reference path.
- A second tiny experiment trained SFT and DPO from one mixed feedback export with matched
  image/question coverage and budgets. The method-comparison runner generated held-out
  comparisons. Altering coverage produced a limitation report without running inference.
- Additional tests cover randomized A/B mapping, conflicting votes, correction/audit rejection,
  current protected-image checks, pair-integrity validation, dependency-version gating,
  sequence-length rejection without truncation, and identical-token response rejection.

The tests use the real pinned TRL preference loss and multimodal forwards, not a text-only
substitute. Preprocessing is deliberately customized through a narrow version-pinned trainer
hook to retain template-defined completion boundaries and image tensors. The tests also exercise
that hook and the normal TRL data collator, optimizer and reference-policy implementation.

## Environment and boundaries

Python 3.11.5 and CPU PyTorch 2.7.1 were used. TRL/Datasets installation added their dependencies
and selected fsspec 2025.3.0; `requirements-tested.txt` records the resulting environment.
CI installs the DPO extra. The 24 existing Gradio 5 private-event deprecation warnings remain;
the full suite ran outside the Windows sandbox restriction for local event-loop sockets.

These successful experiments validate software, not preference-learning benefits. Images,
feedback provenance and tiny random model weights in tests are fabricated fixtures, held only
in temporary directories. Evaluation reports are marked software-test-only through the synthetic
held-out manifest. No research dataset, model weights or invented results are committed.

The full SmolVLM checkpoint was not downloaded or trained in this phase; genuine human preference
training, GPU execution, peak memory, large-scale training and objective superiority remain
unverified. A static supported report does not guarantee a full checkpoint fits the machine.
Qwen/other architectures, quantization, SFT-adapter continuation, FP16 training and unknown
library versions are rejected. No silent fallback is provided.

Phase 4's software acceptance is met by successful real multimodal TRL optimization and
reload/evaluation on the tiny test model. A credible research result still needs actual
preferences, available full-model weights and the documented controlled evaluation procedure.
