"""Explicit inference-only quantization modes and hardware capability gates."""

import importlib.metadata


def quantization_support(config) -> dict:
    """Check prerequisites without downloading weights or claiming measured speedups."""
    reasons = []
    if config.quantization == "none":
        return {"supported": True, "mode": "none", "reasons": []}
    import torch

    if config.family != "smolvlm" or config.backend != "transformers":
        reasons.append("Quantization is implemented only for real SmolVLM/Idefics3 inference")
    if config.quantization == "dynamic_int8":
        if config.device != "cpu" or config.dtype != "float32":
            reasons.append("Dynamic int8 requires CPU float32 inputs and unquantized modules")
        if torch.backends.quantized.engine in {"none", ""}:
            reasons.append("PyTorch has no active quantized CPU engine")
    else:
        if config.device != "cuda" or not torch.cuda.is_available():
            reasons.append("bitsandbytes experiments require CUDA-enabled PyTorch and a CUDA GPU")
        try:
            version = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "0.45.5":
            reasons.append(
                f"Install the quantization extra (bitsandbytes==0.45.5); found {version}"
            )
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability < ((7, 5) if config.quantization == "bnb_int8" else (6, 0)):
                reasons.append(
                    "GPU compute capability is below the configured quantization requirement"
                )
            if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                reasons.append("GPU does not support bfloat16")
    return {"supported": not reasons, "mode": config.quantization, "reasons": reasons}


def bnb_loading_kwargs(config) -> dict:
    """Build explicit single-GPU loading settings; never offload silently."""
    import torch
    from transformers import BitsAndBytesConfig

    settings = {"llm_int8_skip_modules": ["vision_model", "connector", "lm_head"]}
    if config.quantization == "bnb_int8":
        settings["load_in_8bit"] = True
    elif config.quantization == "bnb_nf4":
        settings.update(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.dtype),
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError("Expected a bitsandbytes mode")
    return {"quantization_config": BitsAndBytesConfig(**settings), "device_map": {"": 0}}


def dynamic_quantize(model):
    """Quantize only language-model linear layers; merge an adapter before conversion."""
    import torch

    if model.config.model_type != "idefics3":
        raise ValueError("Dynamic int8 supports Idefics3 only")
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload(safe_merge=True)
    names = {
        name
        for name, module in model.named_modules()
        if ".text_model." in name and isinstance(module, torch.nn.Linear)
    }
    if not names:
        raise ValueError("No language-model linear layers available for quantization")
    quantized = torch.ao.quantization.quantize_dynamic(
        model, qconfig_spec=names, dtype=torch.qint8, inplace=False
    )
    count = sum(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear) for module in quantized.modules()
    )
    if not count:
        raise ValueError("Dynamic int8 did not convert any layers")
    return quantized, count
