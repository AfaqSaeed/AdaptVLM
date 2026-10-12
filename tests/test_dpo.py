"""Preference integrity and a real tiny TRL multimodal DPO experiment."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from adaptvlm.config import ModelConfig
from adaptvlm.feedback.store import FeedbackStore
from adaptvlm.training.preferences import prepare_preferences, load_preferences
from adaptvlm.training.dpo import PreferenceConfig, check_compatibility


def candidate(answer):
    config = ModelConfig(model_id="test-only")
    return {
        "answer": answer,
        "model_id": config.model_id,
        "generation_parameters": config.to_dict(),
        "resolved_revision": "test",
        "latency_seconds": 0.1,
        "error": None,
    }


@pytest.fixture
def pair_inputs(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    req = store.create_request(
        Image.new("RGB", (16, 16), "red"), "Color?", [candidate("blue"), candidate("red")], "train"
    )
    vote = "a" if req["display_order"][0] == 0 else "b"
    store.add_preference(req["id"], vote)
    bundle = tmp_path / "feedback"
    store.export(bundle, True)
    Image.new("RGB", (16, 16), "green").save(tmp_path / "heldout.png")
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "synthetic": True,
                "name": "fixture",
                "examples": [
                    {
                        "id": "test",
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
    return store, bundle, manifest


def test_pair_conversion_mapping_and_protection(pair_inputs, tmp_path):
    _, bundle, manifest = pair_inputs
    prepare_preferences(bundle, tmp_path / "pairs", [manifest])
    rows, info = load_preferences(tmp_path / "pairs", [manifest])
    assert info["count"] == 1
    assert rows[0]["chosen"] == "blue" and rows[0]["rejected"] == "red"
    Image.new("RGB", (16, 16), "red").save(tmp_path / "heldout.png")
    with pytest.raises(ValueError, match="protected"):
        prepare_preferences(bundle, tmp_path / "bad", [manifest])


def test_conflicting_votes_fail(pair_inputs, tmp_path):
    store, _, manifest = pair_inputs
    req = store.create_request(
        Image.new("RGB", (16, 16), "red"), "Color?", [candidate("blue"), candidate("red")], "train"
    )
    store.add_preference(req["id"], "a" if req["display_order"][0] == 1 else "b")
    store.export(tmp_path / "conflicts", True)
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_preferences(tmp_path / "conflicts", tmp_path / "bad", [manifest])


def test_no_pairs_from_corrections_or_audit(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    req = store.create_request(Image.new("RGB", (2, 2)), "Q?", [candidate("red")], "train")
    store.add_correction(req["id"], False, "blue")
    store.export(tmp_path / "eligible", True)
    protected = Path(__file__).resolve().parents[1] / "examples/synthetic/manifest.json"
    with pytest.raises(ValueError, match="No eligible"):
        prepare_preferences(tmp_path / "eligible", tmp_path / "bad", [protected])
    store.export(tmp_path / "audit")
    with pytest.raises(ValueError, match="eligible-only"):
        prepare_preferences(tmp_path / "audit", tmp_path / "bad", [protected])


def test_unsupported_compatibility_and_beta():
    assert not check_compatibility(ModelConfig(backend="mock"))["supported"]
    assert not check_compatibility(ModelConfig(family="qwen2_5_vl"))["supported"]
    with pytest.raises(ValueError):
        PreferenceConfig(beta=0)


@pytest.mark.skipif(importlib.util.find_spec("trl") is None, reason="Install dpo extra")
def test_real_trl_dpo_roundtrip(pair_inputs, tmp_path):
    import torch
    from test_training import tiny_checkpoint
    from adaptvlm.training.dpo import train_dpo, make_trainer, encode_pairs
    from adaptvlm.training.runner import attach_lora
    from adaptvlm.models.backend import TransformersBackend
    from adaptvlm.training.adapter import read_adapter
    from adaptvlm.evaluation.comparison import compare_adapter

    _, bundle, manifest = pair_inputs
    data, base, adapter = tmp_path / "pairs", tmp_path / "base", tmp_path / "adapter"
    prepare_preferences(bundle, data, [manifest])
    original, processor = tiny_checkpoint(base)
    config = ModelConfig(model_id=str(base), prompt_suffix="", max_new_tokens=2)
    settings = PreferenceConfig(
        epochs=2, max_sequence_length=128, learning_rate=0.01, lora_rank=2, lora_alpha=4
    )
    assert check_compatibility(config)["supported"]
    rows, _ = load_preferences(data, [manifest])
    policy = attach_lora(original, settings)
    encoded = encode_pairs(processor, policy, rows, settings, config)
    trainer = make_trainer(policy, processor, encoded, tmp_path / "probe", settings, config)
    batch = trainer._prepare_inputs(trainer.data_collator(encoded))
    assert "pixel_values" in batch and trainer.is_vision_model
    ref_before = trainer.compute_ref_log_probs(batch)
    output = trainer.concatenated_forward(policy, batch)
    assert torch.allclose(output["chosen_logps"], ref_before[0])
    # LoRA perturbation changes policy, but reference remains exactly the frozen base.
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B" in name:
                parameter.add_(0.1)
    ref_after = trainer.compute_ref_log_probs(batch)
    assert all(torch.equal(a, b) for a, b in zip(ref_before, ref_after, strict=True))
    assert not torch.allclose(
        trainer.concatenated_forward(policy, batch)["chosen_logps"], ref_after[0]
    )
    result = train_dpo(data, adapter, config, settings, [manifest])
    assert result["statistics"]["optimizer_steps"] == 2
    loaded = TransformersBackend(replace(config, adapter_path=str(adapter)))
    assert any("lora_B" in n and torch.count_nonzero(p) for n, p in loaded.model.named_parameters())
    metadata = read_adapter(adapter)
    metadata["software_test_only"] = True
    (adapter / "training_run.json").write_text(json.dumps(metadata))
    comparison = compare_adapter(config, adapter, manifest, tmp_path / "comparison")
    assert comparison["software_test_only"]
    assert not (adapter / "model.safetensors").exists()


@pytest.mark.skipif(importlib.util.find_spec("trl") is None, reason="Install dpo extra")
def test_matched_sft_dpo_comparison_and_limitations(pair_inputs, tmp_path):
    from test_training import tiny_checkpoint
    from adaptvlm.training.data import prepare_sft
    from adaptvlm.training.runner import train_sft
    from adaptvlm.training.config import SFTConfig
    from adaptvlm.training.dpo import train_dpo
    from adaptvlm.evaluation.methods import compare_methods

    store, _, manifest = pair_inputs
    req = store.create_request(
        Image.new("RGB", (16, 16), "red"), "Color?", [candidate("red")], "train"
    )
    store.add_correction(req["id"], False, "blue")
    store.export(tmp_path / "mixed", True)
    prepare_sft(tmp_path / "mixed", tmp_path / "sft-data", [manifest])
    prepare_preferences(tmp_path / "mixed", tmp_path / "dpo-data", [manifest])
    tiny_checkpoint(tmp_path / "base")
    model = ModelConfig(model_id=str(tmp_path / "base"), prompt_suffix="", max_new_tokens=2)
    settings = dict(
        epochs=1,
        gradient_accumulation_steps=1,
        max_sequence_length=128,
        learning_rate=0.01,
        lora_rank=2,
        lora_alpha=4,
    )
    train_sft(tmp_path / "sft-data", tmp_path / "sft", model, SFTConfig(**settings), [manifest])
    train_dpo(
        tmp_path / "dpo-data", tmp_path / "dpo", model, PreferenceConfig(**settings), [manifest]
    )
    result = compare_methods(
        model, tmp_path / "sft", tmp_path / "dpo", manifest, tmp_path / "matched"
    )
    assert result["comparable"] and result["sft_to_dpo"]["software_test_only"]
    metadata_path = tmp_path / "dpo/training_run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["training_example_keys"] = ["different question"]
    metadata_path.write_text(json.dumps(metadata))
    result = compare_methods(
        model, tmp_path / "sft", tmp_path / "dpo", manifest, tmp_path / "unmatched"
    )
    assert not result["comparable"]
    assert not (tmp_path / "unmatched/sft").exists()


def test_version_gate_and_pair_integrity(pair_inputs, tmp_path, monkeypatch):
    import hashlib

    _, bundle, manifest = pair_inputs
    prepare_preferences(bundle, tmp_path / "pairs", [manifest])
    path = tmp_path / "pairs/pairs.jsonl"
    row = json.loads(path.read_text())
    row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
    path.write_text(json.dumps(row) + "\n")
    info_path = tmp_path / "pairs/manifest.json"
    info = json.loads(info_path.read_text())
    info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="mapping"):
        load_preferences(tmp_path / "pairs", [manifest])
    monkeypatch.setattr("importlib.metadata.version", lambda _: "unexpected")
    check = check_compatibility(ModelConfig())
    assert not check["supported"] and len(check["limitations"]) == 4


@pytest.mark.skipif(importlib.util.find_spec("trl") is None, reason="Install dpo extra")
def test_dpo_encoder_rejects_truncation_and_identical_tokens(pair_inputs, tmp_path):
    from test_training import tiny_checkpoint
    from adaptvlm.training.dpo import encode_pairs

    _, bundle, manifest = pair_inputs
    prepare_preferences(bundle, tmp_path / "pairs", [manifest])
    rows, _ = load_preferences(tmp_path / "pairs", [manifest])
    model, processor = tiny_checkpoint(tmp_path / "base")
    with pytest.raises(ValueError, match="never truncated"):
        encode_pairs(
            processor,
            model,
            rows,
            PreferenceConfig(max_sequence_length=2),
            ModelConfig(prompt_suffix=""),
        )
    rows[0]["chosen"] = rows[0]["rejected"]
    with pytest.raises(ValueError, match="tokenize identically"):
        encode_pairs(processor, model, rows, PreferenceConfig(), ModelConfig(prompt_suffix=""))
