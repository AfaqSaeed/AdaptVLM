"""Configuration for the supported nonquantized LoRA training path."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SFTConfig:
    """Training settings; no automatic test-driven hyperparameter selection."""

    learning_rate: float = 0.0001
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_sequence_length: int = 2048
    seed: int = 42
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "epochs",
            "lora_rank",
            "lora_alpha",
            "max_sequence_length",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.target_modules or any(
            t not in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
            for t in self.target_modules
        ):
            raise ValueError("Use supported language-model linear projection names")


def load_sft_config(path: str | Path) -> SFTConfig:
    """Load strict YAML training settings."""
    return SFTConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
