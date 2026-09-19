"""Serialized inference with measured latency and explicit failures."""

import time
from dataclasses import asdict, dataclass
from threading import Lock

from PIL import Image, ImageOps

from adaptvlm.models.backend import load_backend
from adaptvlm.config import ModelConfig


@dataclass
class Prediction:
    answer: str
    latency_seconds: float
    peak_gpu_memory_bytes: int | None
    error: str | None = None

    def to_dict(self) -> dict:
        """Return JSON-ready output."""
        return asdict(self)


class InferenceEngine:
    """Reuse a loaded model across requests; serialize GPU measurements."""

    def __init__(self, config: ModelConfig, backend=None):
        self.config = config
        self.lock = Lock()
        if backend is None:
            self.backend, self.load_seconds = load_backend(config)
        else:
            self.backend, self.load_seconds = backend, 0.0

    def predict(self, image: Image.Image, question: str) -> Prediction:
        """Answer one image question, reporting recoverable CUDA OOM as a failure."""
        if not isinstance(image, Image.Image) or not question or not question.strip():
            raise ValueError("Provide an image and a non-empty question")
        image = ImageOps.exif_transpose(image).convert("RGB")
        with self.lock:
            gpu = self.config.device == "cuda" and self.config.backend != "mock"
            if gpu:
                import torch

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                answer = self.backend.generate(image, question.strip(), self.config)
                if gpu:
                    torch.cuda.synchronize()
                return Prediction(
                    answer,
                    time.perf_counter() - start,
                    torch.cuda.max_memory_allocated() if gpu else None,
                )
            except RuntimeError as exc:
                if not gpu or not isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                torch.cuda.empty_cache()
                return Prediction(
                    "",
                    time.perf_counter() - start,
                    None,
                    "CUDA out of memory: reduce image size/tokens or use CPU",
                )
