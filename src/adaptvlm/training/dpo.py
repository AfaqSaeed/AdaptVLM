"""Experimental pinned TRL multimodal DPO, with no text-only fallback."""

import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from adaptvlm.config import ModelConfig
from adaptvlm.models.backend import TransformersBackend
from adaptvlm.training.adapter import file_hash, local_base_hash
from adaptvlm.training.collator import SFTCollator
from adaptvlm.training.config import SFTConfig
from adaptvlm.training.preferences import load_preferences
from adaptvlm.training.runner import attach_lora

PINNED = {"trl": "0.16.1", "transformers": "4.51.3", "peft": "0.15.2", "datasets": "3.5.1"}


@dataclass(frozen=True)
class PreferenceConfig(SFTConfig):
    """Shared LoRA settings plus standard sigmoid DPO beta."""

    beta: float = 0.1
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("DPO beta must be finite and positive")


def check_compatibility(config: ModelConfig) -> dict:
    """Check installed versions and trainer APIs without downloading weights."""
    versions, reasons = {}, []
    if config.quantization != "none":
        reasons.append("DPO quantized training is unsupported; quantization is inference-only")
    for package, expected in PINNED.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        if versions[package] != expected:
            reasons.append(f"Requires {package}=={expected}; found {versions[package]}")
    if config.backend != "transformers" or config.family != "smolvlm":
        reasons.append("Only the real SmolVLM/Idefics3 path is supported")
    if config.adapter_path:
        reasons.append(
            "DPO initializes a fresh LoRA policy on the instructed base; SFT adapter continuation is unsupported"
        )
    if config.dtype == "float16":
        reasons.append("FP16 training is unsupported; use float32 or CUDA bfloat16")
    if not reasons:
        try:
            from trl import DPOTrainer
            from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

            if "idefics3" not in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES:
                reasons.append("Installed Transformers does not classify Idefics3 as a VLM")
            for name in ("concatenated_forward", "compute_ref_log_probs", "_prepare_dataset"):
                if not hasattr(DPOTrainer, name):
                    reasons.append(f"TRL is missing {name}")
        except (ImportError, RuntimeError) as exc:
            reasons.append(f"Trainer import failed: {exc}")
    return {
        "schema_version": 1,
        "experimental": True,
        "supported": not reasons,
        "versions": versions,
        "limitations": reasons,
        "scope": "Static compatibility only; each training run preflights real image-conditioned policy/reference forwards",
    }


def encode_pairs(
    processor, model, rows: list[dict], config: PreferenceConfig, model_config: ModelConfig
) -> list[dict]:
    """Preserve exact chat-template completion tokens and multimodal prompt tensors."""
    import torch

    collate = SFTCollator(
        processor,
        config.max_sequence_length,
        model.config.image_token_id,
        model_config.prompt_suffix,
    )
    encoded = []
    for row in rows:
        sides = [collate([row | {"answer": row[key]}]) for key in ("chosen", "rejected")]
        result = {}
        for key, batch in zip(("chosen", "rejected"), sides, strict=True):
            labels = batch["labels"][0]
            start = int((labels != -100).nonzero()[0])
            if (labels[start:] == -100).any():
                raise ValueError(
                    "Completion contains masked image/padding tokens; cannot form DPO likelihood"
                )
            prompt = batch["input_ids"][0, :start].tolist()
            if key == "rejected" and prompt != result["prompt_input_ids"]:
                raise ValueError("Chosen/rejected multimodal prompts differ")
            result["prompt_input_ids"] = prompt
            result[f"{key}_input_ids"] = batch["input_ids"][0, start:].tolist()
        for field in ("pixel_values", "pixel_attention_mask"):
            if field in sides[0]:
                if not torch.equal(sides[0][field], sides[1][field]):
                    raise ValueError("Pair image preprocessing differs")
                result[field] = sides[0][field][0].tolist()
        if "pixel_values" not in result:
            raise ValueError("Processor discarded image tensors; text-only DPO is forbidden")
        if result["chosen_input_ids"] == result["rejected_input_ids"]:
            raise ValueError("Pair responses tokenize identically")
        encoded.append(result)
    return encoded


def make_trainer(
    model,
    processor,
    encoded: list[dict],
    output: Path,
    config: PreferenceConfig,
    model_config: ModelConfig,
):
    """Use TRL's DPO forward/reference/loss with validated pretokenized VLM inputs."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    class PreparedVisionDPOTrainer(DPOTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch

            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite DPO loss; refusing to publish a completed adapter")
            return result

        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            # Pinned internal hook: preprocessing above verifies prefix boundaries and
            # preserves exact end-of-turn tokens, instead of adding a second EOS.
            return dataset

    args = DPOConfig(
        output_dir=str(output),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        beta=config.beta,
        loss_type="sigmoid",
        max_length=None,
        max_prompt_length=None,
        max_completion_length=None,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        use_cpu=model_config.device == "cpu",
        bf16=model_config.dtype == "bfloat16",
        fp16=False,
        seed=config.seed,
        data_seed=config.seed,
        max_grad_norm=config.max_grad_norm,
        weight_decay=0.0,
        lr_scheduler_type="constant",
    )
    return PreparedVisionDPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        processing_class=processor,
        train_dataset=Dataset.from_list(encoded),
    )


def train_dpo(
    data: str | Path,
    output: str | Path,
    model_config: ModelConfig,
    config: PreferenceConfig,
    protected_manifests: list[str | Path],
) -> dict:
    """Train an experimental image-conditioned DPO adapter and save a reloadable artifact."""
    import torch
    from transformers import set_seed

    compatibility = check_compatibility(model_config)
    if not compatibility["supported"]:
        raise ValueError("Multimodal DPO unsupported: " + "; ".join(compatibility["limitations"]))
    rows, dataset = load_preferences(data, protected_manifests)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("DPO output already exists")
    set_seed(config.seed)
    backend = TransformersBackend(model_config)
    fingerprint = local_base_hash(model_config.model_id)
    if fingerprint is None and not backend.resolved_revision:
        raise ValueError("Cannot resolve base checkpoint identity")
    model = attach_lora(backend.model, config)
    model.config.use_cache = False
    encoded = encode_pairs(backend.processor, model, rows, config, model_config)
    trainer = make_trainer(
        model, backend.processor, encoded, output / "trainer", config, model_config
    )
    if not trainer.is_vision_model or not trainer.is_peft_model or trainer.ref_model is not None:
        raise ValueError("Unexpected TRL vision/PEFT/reference path")
    batch = trainer._prepare_inputs(trainer.data_collator([encoded[0]]))
    if "pixel_values" not in batch:
        raise ValueError("TRL discarded image tensors")
    with torch.no_grad():
        reference = trainer.compute_ref_log_probs(batch)
        policy = trainer.concatenated_forward(model, batch)
        if not all(
            torch.isfinite(value).all()
            for value in (*reference, policy["chosen_logps"], policy["rejected_logps"])
        ):
            raise ValueError("Nonfinite multimodal policy/reference likelihoods")
    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory during multimodal DPO; reduce batch/image size "
            "or use more memory. No completed adapter was published."
        ) from exc
    if trainer.state.global_step < 1 or not math.isfinite(result.training_loss):
        raise ValueError("DPO did not complete with finite loss")
    if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
        raise ValueError("Nonfinite DPO adapter parameters")
    model.eval()
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "method": "dpo",
        "experimental": True,
        "created_at": datetime.now(UTC).isoformat(),
        "software_test_only": False,
        "model_config": model_config.to_dict(),
        "training_config": asdict(config),
        "resolved_base_revision": backend.resolved_revision,
        "local_base_sha256": fingerprint,
        "dataset": dataset,
        "training_image_hashes": sorted({r["image_sha256"] for r in rows}),
        "training_example_keys": sorted({r["image_sha256"] + ":" + r["question"] for r in rows}),
        "compatibility": compatibility,
        "reference_policy": "frozen instructed base with adapters disabled",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "statistics": {
            "optimizer_steps": trainer.state.global_step,
            "training_loss": result.training_loss,
            "metrics": result.metrics,
            "log_history": trainer.state.log_history,
        },
        "adapter_files": {
            name: file_hash(output / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
    }
    (output / "training_run.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata
