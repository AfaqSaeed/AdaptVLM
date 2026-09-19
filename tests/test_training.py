"""Offline Phase 3 checks, including actual tiny multimodal LoRA optimization."""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from adaptvlm.config import ModelConfig
from adaptvlm.feedback.store import FeedbackStore
from adaptvlm.training.config import SFTConfig
from adaptvlm.training.data import load_sft, prepare_sft

ROOT = Path(__file__).resolve().parents[1]


def candidate():
    config = ModelConfig(model_id="test-only-provenance")
    return {
        "answer": "red",
        "model_id": config.model_id,
        "resolved_revision": "test",
        "generation_parameters": config.to_dict(),
        "latency_seconds": 0.1,
        "error": None,
    }


@pytest.fixture
def sft_inputs(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    image = Image.new("RGB", (16, 16), "red")
    for question, answer in [("Color?", "blue"), ("What color?", "blue")]:
        req = store.create_request(image, question, [candidate()], "train")
        store.add_correction(req["id"], False, answer)
    bundle = tmp_path / "feedback"
    store.export(bundle, True)
    Image.new("RGB", (16, 16), "green").save(tmp_path / "heldout.png")
    protected = tmp_path / "evaluation.json"
    protected.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "synthetic": False,
                "name": "test-only",
                "examples": [
                    {
                        "id": "heldout",
                        "image": "heldout.png",
                        "question": "Color?",
                        "answers": ["green"],
                        "category": "color",
                        "split": "test",
                    }
                ],
            }
        )
    )
    return store, bundle, protected


def test_conversion_requires_eligible_and_protects_current_data(sft_inputs, tmp_path):
    store, bundle, protected = sft_inputs
    data = tmp_path / "sft"
    info = prepare_sft(bundle, data, [protected])
    rows, _ = load_sft(data, [protected])
    assert info["count"] == 2 and all(r["answer"] == "blue" for r in rows)
    store.export(tmp_path / "audit")
    with pytest.raises(ValueError, match="eligible-only"):
        prepare_sft(tmp_path / "audit", tmp_path / "bad", [protected])
    # Replacing held-out pixels with training pixels invalidates old exports.
    Image.new("RGB", (16, 16), "red").save(tmp_path / "heldout.png")
    with pytest.raises(ValueError, match="protected"):
        prepare_sft(bundle, tmp_path / "stale", [protected])
    with pytest.raises(ValueError):
        load_sft(data, [protected])


def test_duplicate_and_conflicting_corrections(sft_inputs, tmp_path):
    store, _, protected = sft_inputs
    image = Image.new("RGB", (16, 16), "red")
    req = store.create_request(image, "Color?", [candidate()], "train")
    store.add_correction(req["id"], False, "blue")
    store.export(tmp_path / "duplicate", True)
    assert prepare_sft(tmp_path / "duplicate", tmp_path / "sft", [protected])["count"] == 2
    req = store.create_request(image, "Color?", [candidate()], "train")
    store.add_correction(req["id"], False, "yellow")
    store.export(tmp_path / "conflict", True)
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_sft(tmp_path / "conflict", tmp_path / "bad", [protected])


def test_correct_answers_and_preferences_are_not_sft(sft_inputs, tmp_path):
    store, _, protected = sft_inputs
    image = Image.new("RGB", (16, 16), "red")
    req = store.create_request(image, "Question?", [candidate()], "train")
    store.add_correction(req["id"], True)
    other = candidate() | {"answer": "blue"}
    req = store.create_request(image, "Pair?", [candidate(), other], "train")
    store.add_preference(req["id"], "a")
    store.export(tmp_path / "mixed", True)
    info = prepare_sft(tmp_path / "mixed", tmp_path / "sft", [protected])
    assert info["count"] == 2 and info["skipped_noncorrections"] == 2


def test_sft_config_errors():
    for args in (
        {"batch_size": 0},
        {"epochs": 0},
        {"learning_rate": float("nan")},
        {"target_modules": ["vision_model"]},
        {"seed": -1},
    ):
        with pytest.raises(ValueError):
            SFTConfig(**args)


def tiny_checkpoint(path):
    """Construct real Idefics3 weights/processor locally; random weights are test-only."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Idefics3Processor,
        Idefics3ImageProcessor,
        Idefics3Config,
        Idefics3ForConditionalGeneration,
        LlamaConfig,
    )

    torch.set_num_threads(1)
    torch.manual_seed(7)
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "<pad>",
                "<bos>",
                "<eos>",
                "<unk>",
                "user",
                "assistant",
                ":",
                "Color",
                "What",
                "color",
                "?",
                "red",
                "blue",
                "green",
                "<image>",
                "<fake_token_around_image>",
                "<global-img>",
            ]
        )
    }
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<image>", "<fake_token_around_image>", "<global-img>"],
    )
    template = (
        "{% for m in messages %}{{m['role'] + ': '}}{% for c in m['content'] %}"
        "{% if c['type'] == 'image' %}{{'<image>'}}{% else %}{{c['text']}}{% endif %}"
        "{% endfor %}{{'<eos>'}}{% endfor %}"
        "{% if add_generation_prompt %}{{'assistant: '}}{% endif %}"
    )
    processor = Idefics3Processor(
        Idefics3ImageProcessor(
            size={"longest_edge": 16}, max_image_size={"longest_edge": 16}, do_image_splitting=False
        ),
        tok,
        image_seq_len=4,
        chat_template=template,
    )
    config = Idefics3Config(
        image_token_id=tok.convert_tokens_to_ids("<image>"),
        pad_token_id=0,
        scale_factor=2,
        text_config=LlamaConfig(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        ).to_dict(),
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 16,
            "patch_size": 4,
        },
    )
    config._attn_implementation = "eager"
    model = Idefics3ForConditionalGeneration(config)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return model, processor


requires_training = pytest.mark.skipif(
    importlib.util.find_spec("peft") is None, reason="Install training extra"
)


@requires_training
def test_real_multimodal_lora_roundtrip_and_comparison(sft_inputs, tmp_path):
    import torch
    from adaptvlm.training.collator import SFTCollator
    from adaptvlm.training.runner import train_sft
    from adaptvlm.models.backend import TransformersBackend
    from adaptvlm.training.adapter import read_adapter
    from adaptvlm.evaluation.comparison import compare_adapter

    _, bundle, protected = sft_inputs
    data, base = tmp_path / "sft", tmp_path / "base"
    prepare_sft(bundle, data, [protected])
    original, processor = tiny_checkpoint(base)
    model_config = ModelConfig(model_id=str(base), prompt_suffix="", max_new_tokens=2)
    rows, _ = load_sft(data, [protected])
    collate = SFTCollator(processor, 128, original.config.image_token_id)
    batch = collate(rows)
    assert batch["pixel_values"].numel() > 0
    for i in range(2):
        supervised = batch["labels"][i][batch["labels"][i] != -100].tolist()
        assert supervised == [
            processor.tokenizer.convert_tokens_to_ids("blue"),
            processor.tokenizer.eos_token_id,
        ]
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    with pytest.raises(ValueError, match="never truncated"):
        SFTCollator(processor, 2, original.config.image_token_id)(rows)
    adapter = tmp_path / "adapter"
    # Three epochs with an incomplete accumulation window exercises remainder handling.
    result = train_sft(
        data,
        adapter,
        model_config,
        SFTConfig(
            epochs=3,
            gradient_accumulation_steps=3,
            learning_rate=0.01,
            max_sequence_length=128,
            lora_rank=2,
            lora_alpha=4,
        ),
        [protected],
    )
    assert result["statistics"]["optimizer_steps"] == 3
    assert result["statistics"]["trainable_parameters"] < result["statistics"]["total_parameters"]
    assert not (adapter / "model.safetensors").exists()
    loaded = TransformersBackend(replace(model_config, adapter_path=str(adapter)))
    assert all(not p.requires_grad for p in loaded.model.parameters())
    assert any(
        "lora_B" in name and torch.count_nonzero(p).item() > 0
        for name, p in loaded.model.named_parameters()
    )
    original_state = original.state_dict()
    for name, parameter in loaded.model.base_model.model.named_parameters():
        if "lora_" not in name:
            assert torch.equal(parameter, original_state[name.replace(".base_layer", "")])
    # Mark test artifact honestly before evaluating the random miniature checkpoint.
    metadata = read_adapter(adapter)
    metadata["software_test_only"] = True
    (adapter / "training_run.json").write_text(json.dumps(metadata))
    report = compare_adapter(model_config, adapter, protected, tmp_path / "comparison")
    assert report["software_test_only"]
    assert (tmp_path / "comparison/comparison.json").is_file()
    with pytest.raises(ValueError, match="base model"):
        TransformersBackend(replace(model_config, model_id="wrong", adapter_path=str(adapter)))
    # Tampering is rejected before loading weights.
    with (adapter / "adapter_model.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        read_adapter(adapter)


def test_comparison_regression_categories_and_mismatches():
    from adaptvlm.evaluation.comparison import compare_reports
    from adaptvlm.evaluation.runner import evaluate
    from adaptvlm.inference.engine import InferenceEngine

    before = evaluate(
        InferenceEngine(ModelConfig(backend="mock")), ROOT / "examples/synthetic/manifest.json"
    )
    after = copy.deepcopy(before)
    for row in after["predictions"]:
        row["normalized_match"] = 1 - row["normalized_match"]
    report = compare_reports(before, after)
    assert len(report["improved"]) == len(report["degraded"]) == 1
    assert report["normalized_accuracy_delta"] == 0
    after["config"]["seed"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        compare_reports(before, after)


@requires_training
def test_accumulation_matches_full_batch_with_unequal_target_lengths(sft_inputs, tmp_path):
    import torch
    from transformers import set_seed
    from adaptvlm.training.runner import attach_lora, optimize

    _, bundle, protected = sft_inputs
    prepare_sft(bundle, tmp_path / "sft", [protected])
    rows, _ = load_sft(tmp_path / "sft", [protected])
    rows[1]["answer"] = "blue blue blue"
    model, processor = tiny_checkpoint(tmp_path / "base")
    config = SFTConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        max_sequence_length=128,
        lora_rank=2,
        lora_alpha=4,
    )
    set_seed(config.seed)
    full = attach_lora(copy.deepcopy(model), config)
    set_seed(config.seed)
    accumulated = attach_lora(copy.deepcopy(model), config)
    model_config = ModelConfig(prompt_suffix="")
    full_result = optimize(full, processor, rows, config, model_config)
    accumulated_result = optimize(
        accumulated,
        processor,
        rows,
        replace(config, batch_size=1, gradient_accumulation_steps=3),
        model_config,
    )
    assert full_result["optimizer_steps"] == accumulated_result["optimizer_steps"] == 1
    for name, parameter in full.named_parameters():
        if "lora_" in name:
            assert torch.allclose(
                parameter, dict(accumulated.named_parameters())[name], atol=1e-6, rtol=1e-4
            ), name


@requires_training
def test_unstable_template_is_rejected(sft_inputs, tmp_path):
    from adaptvlm.training.collator import SFTCollator

    _, bundle, protected = sft_inputs
    prepare_sft(bundle, tmp_path / "sft", [protected])
    rows, _ = load_sft(tmp_path / "sft", [protected])
    model, processor = tiny_checkpoint(tmp_path / "base")
    render = processor.apply_chat_template

    def broken(messages, **kwargs):
        value = render(messages, **kwargs)
        return "blue " + value if kwargs.get("add_generation_prompt") else value

    processor.apply_chat_template = broken
    with pytest.raises(ValueError, match="boundary"):
        SFTCollator(processor, 128, model.config.image_token_id)(rows)


def test_sft_data_tampering_and_missing_protected_manifest(sft_inputs, tmp_path):
    _, bundle, protected = sft_inputs
    data = tmp_path / "sft"
    prepare_sft(bundle, data, [protected])
    with pytest.raises(ValueError, match="at least one"):
        load_sft(data, [])
    with (data / "examples.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_sft(data, [protected])


def test_training_config_refuses_unsupported_backends_before_loading(tmp_path):
    pytest.importorskip("transformers")
    from adaptvlm.training.runner import train_sft

    for config in (
        ModelConfig(backend="mock"),
        ModelConfig(family="qwen2_5_vl"),
        ModelConfig(device="cuda", dtype="float16"),
    ):
        with pytest.raises(ValueError):
            train_sft("missing", tmp_path / "out", config, SFTConfig(), ["missing"])
