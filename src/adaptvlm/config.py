"""Validated model configuration."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "transformers"
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    revision: str = "main"
    family: str = "smolvlm"
    device: str = "cpu"
    dtype: str = "float32"
    max_new_tokens: int = 32
    seed: int = 42
    prompt_suffix: str = " Answer with a short word or phrase."
    adapter_path: str | None = None
    quantization: str = "none"

    def __post_init__(self):
        if self.quantization not in {"none", "dynamic_int8", "bnb_int8", "bnb_nf4"}:
            raise ValueError("Unknown quantization mode")
        if self.backend == "mock" and self.quantization != "none":
            raise ValueError("Mock models cannot measure quantization")
        if self.adapter_path is not None and (
            self.backend == "mock" or not self.adapter_path.strip()
        ):
            raise ValueError("Adapters require a real backend and a nonempty local path")
        if self.backend not in {"mock", "transformers"}:
            raise ValueError("backend must be mock or transformers")
        if self.family not in {"smolvlm", "qwen2_5_vl"}:
            raise ValueError("Unsupported model family; add an explicit processor implementation")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dtype")
        if self.device == "cpu" and self.dtype == "float16":
            raise ValueError("Use float32 or bfloat16 on CPU")
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be in [1, 4096]")

    def to_dict(self) -> dict:
        """Return serializable generation and loading settings."""
        return asdict(self)


def load_config(path: str | Path) -> ModelConfig:
    """Read a strict YAML model configuration."""
    with open(path, encoding="utf-8") as stream:
        return ModelConfig(**yaml.safe_load(stream))
