"""Versioned local dataset manifests and leakage checks."""

import hashlib
import json
from pathlib import Path

from PIL import Image


def load_manifest(path: str | Path) -> tuple[dict, str]:
    """Validate all splits before selecting examples; hash decoded images."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("synthetic"), bool):
        raise ValueError("Manifest requires schema_version=1 and synthetic boolean")
    seen_ids, image_splits = set(), {}
    if not data.get("examples"):
        raise ValueError("Dataset is empty")
    digest = hashlib.sha256(path.read_bytes())
    for row in data["examples"]:
        if row["id"] in seen_ids:
            raise ValueError("Duplicate example ID")
        seen_ids.add(row["id"])
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        if not row["question"].strip() or not row["category"].strip():
            raise ValueError("Question and category must be non-empty")
        if not row["answers"] or any(
            not isinstance(a, str) or not a.strip() for a in row["answers"]
        ):
            raise ValueError("Answers must contain non-empty strings")
        image_path = (path.parent / row["image"]).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError("Image path escapes dataset directory")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            content = str(rgb.size).encode() + rgb.tobytes()
        image_hash = hashlib.sha256(content).hexdigest()
        if image_hash in image_splits and image_splits[image_hash] != row["split"]:
            raise ValueError("Duplicate image leakage across splits")
        image_splits[image_hash] = row["split"]
        row["image_path"] = str(image_path)
        row["image_sha256"] = image_hash
        digest.update(content)
    return data, digest.hexdigest()
