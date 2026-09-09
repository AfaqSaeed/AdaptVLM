"""Test the local VQAv2 import without downloading copyrighted images."""

import importlib.util
import json
from pathlib import Path
from PIL import Image
from adaptvlm.datasets.manifest import load_manifest


def test_vqa_import_is_deterministic_and_disjoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_vqav2.py"
    spec = importlib.util.spec_from_file_location("prepare_vqa", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = tmp_path / "images"
    images.mkdir()
    questions, annotations = [], []
    for i in range(5):
        Image.new("RGB", (4, 4), (i * 40, 0, 0)).save(images / f"COCO_val2014_{i:012d}.jpg")
        questions.append({"question_id": i, "question": "Color?"})
        annotations.append(
            {
                "question_id": i,
                "image_id": i,
                "question_type": "what color",
                "answers": [{"answer": "red"}] * 10,
            }
        )
    q, a = tmp_path / "q.json", tmp_path / "a.json"
    q.write_text(json.dumps({"questions": questions}))
    a.write_text(json.dumps({"annotations": annotations}))
    module.prepare(q, a, images, tmp_path / "first", 5)
    module.prepare(q, a, images, tmp_path / "second", 5)
    data, fingerprint = load_manifest(tmp_path / "first/manifest.json")
    assert {r["split"] for r in data["examples"]} == {"train", "validation", "test"}
    assert fingerprint == load_manifest(tmp_path / "second/manifest.json")[1]
    assert not data["synthetic"]
