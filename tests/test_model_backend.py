"""Check multimodal input plumbing with tiny tensors and fake model weights."""

import importlib.util
import pytest
from PIL import Image
from adaptvlm.config import ModelConfig
from adaptvlm.models.backend import TransformersBackend
from adaptvlm.inference.engine import InferenceEngine

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Install model extra"
)


def test_image_template_and_prompt_trimming():
    import torch
    from transformers import BatchFeature

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["type"] == "image"
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "formatted prompt"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["formatted prompt"]
            assert isinstance(kwargs["images"][0], Image.Image)
            return BatchFeature(
                {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones(1, 3, 2, 2)}
            )

        def batch_decode(self, output, **kwargs):
            assert output.tolist() == [[3]]
            return ["red"]

    class Model:
        dtype = torch.float32

        def generate(self, **kwargs):
            assert kwargs["input_ids"].dtype == torch.int64
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3]])

    backend = TransformersBackend.__new__(TransformersBackend)
    backend.processor, backend.model = Processor(), Model()
    assert backend.generate(Image.new("RGB", (2, 2)), "Color?", ModelConfig()) == "red"


def test_cuda_oom_is_reported(monkeypatch):
    import torch

    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)

    class Backend:
        def generate(self, *args):
            raise torch.cuda.OutOfMemoryError("simulated")

    engine = InferenceEngine(ModelConfig(device="cuda"), Backend())
    result = engine.predict(Image.new("RGB", (2, 2)), "Color?")
    assert result.error and "out of memory" in result.error
    assert result.answer == ""
