"""Language-projection LoRA with a small explicit PyTorch optimization loop."""

import importlib.metadata
import json
import math
import platform
import random
import time
from datetime import UTC, datetime
from dataclasses import asdict
from pathlib import Path

from adaptvlm.config import ModelConfig
from adaptvlm.models.backend import TransformersBackend
from adaptvlm.training.adapter import file_hash, local_base_hash
from adaptvlm.training.collator import SFTCollator
from adaptvlm.training.config import SFTConfig
from adaptvlm.training.data import load_sft


def attach_lora(model, config: SFTConfig):
    """Adapt only language-model projections, freezing vision and base parameters."""
    import torch
    from peft import LoraConfig, get_peft_model

    if model.config.model_type != "idefics3":
        raise ValueError("Phase 3 training supports SmolVLM with Idefics3 architecture only")
    modules = [
        name
        for name, module in model.named_modules()
        if ".text_model." in name
        and name.rsplit(".", 1)[-1] in config.target_modules
        and isinstance(module, torch.nn.Linear)
    ]
    if not modules or {name.rsplit(".", 1)[-1] for name in modules} != set(config.target_modules):
        raise ValueError("Requested LoRA projections are missing from this language model")
    return get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=modules,
            lora_dropout=0.0,
            bias="none",
        ),
    )


def optimize(
    model, processor, examples: list[dict], config: SFTConfig, model_config: ModelConfig
) -> dict:
    """Optimize LoRA weights; normalize accumulation by supervised token count."""
    import torch
    from transformers import set_seed

    set_seed(config.seed)
    collate = SFTCollator(
        processor,
        config.max_sequence_length,
        model.config.image_token_id,
        model_config.prompt_suffix,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in model.named_parameters()
    ):
        raise ValueError("Optimization requires an adapter-only trainable model")
    # Validate every example before any optimizer update, including overlength inputs.
    for row in examples:
        collate([row])
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    model.train()
    model.config.use_cache = False
    history, updates = [], 0
    rng = random.Random(config.seed)
    start = time.perf_counter()
    for epoch in range(config.epochs):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        batches = [
            indices[i : i + config.batch_size] for i in range(0, len(indices), config.batch_size)
        ]
        for offset in range(0, len(batches), config.gradient_accumulation_steps):
            window = [
                collate([examples[i] for i in group])
                for group in batches[offset : offset + config.gradient_accumulation_steps]
            ]
            counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in window]
            tokens, loss_sum = sum(counts), 0.0
            optimizer.zero_grad(set_to_none=True)
            for batch, count in zip(window, counts, strict=True):
                inputs = {
                    key: tensor.to(
                        device=model_config.device,
                        dtype=model.dtype if tensor.is_floating_point() else tensor.dtype,
                    )
                    for key, tensor in batch.items()
                }
                loss = model(**inputs).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; no completed adapter will be saved")
                (loss * (count / tokens)).backward()
                loss_sum += float(loss.detach()) * count
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "update": updates,
                    "supervised_tokens": tokens,
                    "loss": loss_sum / tokens,
                }
            )
    if model_config.device == "cuda":
        torch.cuda.synchronize()
    model.eval()
    model.config.use_cache = True
    return {
        "optimizer_steps": updates,
        "training_seconds": time.perf_counter() - start,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "loss_history": history,
    }


def train_sft(
    data: str | Path,
    output: str | Path,
    model_config: ModelConfig,
    config: SFTConfig,
    protected_manifests: list[str | Path],
) -> dict:
    """Train a fresh adapter from a validated correction bundle; never save base weights."""
    import torch
    from transformers import set_seed

    if model_config.backend != "transformers" or model_config.family != "smolvlm":
        raise ValueError("SFT supports the real SmolVLM/Idefics3 backend only")
    if model_config.adapter_path:
        raise ValueError("Phase 3 trains a fresh adapter; resuming/stacking is unsupported")
    if model_config.dtype == "float16":
        raise ValueError(
            "Use float32 or supported CUDA bfloat16 for training; FP16 scaling is not implemented"
        )
    if (
        model_config.device == "cuda"
        and model_config.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This CUDA device does not support bfloat16; use float32")
    examples, dataset = load_sft(data, protected_manifests)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Training output exists; choose a new directory")
    set_seed(config.seed)
    backend = TransformersBackend(model_config)
    local_hash = local_base_hash(model_config.model_id)
    if not local_hash and not backend.resolved_revision:
        raise ValueError("Cannot record a resolved base checkpoint revision")
    model = attach_lora(backend.model, config)
    if model_config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        stats = optimize(model, backend.processor, examples, config, model_config)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory during SFT; reduce batch/image size or sequence "
            "length, or use a larger device. No completed adapter was saved."
        ) from exc
    if not stats["optimizer_steps"] or not math.isfinite(stats["loss_history"][-1]["loss"]):
        raise ValueError("Training did not complete")
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "software_test_only": False,
        "model_config": model_config.to_dict(),
        "training_config": asdict(config),
        "resolved_base_revision": backend.resolved_revision,
        "local_base_sha256": local_hash,
        "dataset": dataset,
        "training_image_hashes": sorted({r["image_sha256"] for r in examples}),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name() if model_config.device == "cuda" else None,
            "cuda": torch.version.cuda,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
            if model_config.device == "cuda"
            else None,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "accelerate")
            },
        },
        "statistics": stats,
        "adapter_files": {
            name: file_hash(output / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
    }
    # Written last: incomplete save directories cannot be loaded as completed adapters.
    (output / "training_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
