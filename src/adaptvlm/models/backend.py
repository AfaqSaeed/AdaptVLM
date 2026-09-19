"""Explicit backend loading; no model downloads at import time."""

import time
from PIL import Image

from adaptvlm.config import ModelConfig


class MockBackend:
    """Constant-answer plumbing stub, never a research baseline."""

    resolved_revision = "local-mock"

    def generate(self, image: Image.Image, question: str, config: ModelConfig) -> str:
        return "red"


class TransformersBackend:
    """SmolVLM and Qwen2.5-VL single-image inference."""

    def __init__(self, config: ModelConfig):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select device: cpu or install CUDA PyTorch.")
        from adaptvlm.training.adapter import adapter_revision

        revision, self.adapter_metadata = adapter_revision(config)
        self.processor = AutoProcessor.from_pretrained(
            config.model_id, revision=revision, trust_remote_code=False
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                config.model_id,
                revision=revision,
                torch_dtype=getattr(torch, config.dtype),
                trust_remote_code=False,
                attn_implementation="eager",
            )
            .to(config.device)
            .eval()
        )
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if config.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, config.adapter_path, is_trainable=False
            ).eval()

    def generate(self, image: Image.Image, question: str, config: ModelConfig) -> str:
        import torch
        from transformers import set_seed

        set_seed(config.seed)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question + config.prompt_suffix},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(config.device)
        # Preserve integer token IDs while matching image tensors to model precision.
        for key, value in inputs.items():
            if value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def load_backend(config: ModelConfig) -> tuple[MockBackend | TransformersBackend, float]:
    """Load weights once and return backend plus elapsed load time."""
    start = time.perf_counter()
    backend = MockBackend() if config.backend == "mock" else TransformersBackend(config)
    return backend, time.perf_counter() - start
