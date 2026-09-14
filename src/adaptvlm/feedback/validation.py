"""Validate portable raw-feedback bundles independently of the database."""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

from adaptvlm.feedback.store import PREFERENCES, SPLITS, image_digest, text, validate_candidate


def validate_export(directory: str | Path) -> dict:
    """Check schema, IDs, provenance, image integrity and training-eligibility flags."""
    try:
        return _validate_export(directory)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"Malformed feedback bundle: {exc}") from exc


def _validate_export(directory: str | Path) -> dict:
    root = Path(directory).resolve()
    info = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = (root / "feedback.jsonl").read_bytes()
    if info.get("schema_version") != 1 or type(info.get("eligible_only")) is not bool:
        raise ValueError("Invalid export manifest")
    if hashlib.sha256(payload).hexdigest() != info.get("sha256"):
        raise ValueError("Feedback file hash mismatch")
    seen, requests = set(), set()
    for line in payload.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("schema_version") != 1 or row.get("split") not in SPLITS:
            raise ValueError("Invalid feedback schema or split")
        for field, seen_ids in (("feedback_id", seen), ("id", requests)):
            uuid.UUID(row[field])
            if row[field] in seen_ids:
                raise ValueError("Duplicate feedback or request ID")
            seen_ids.add(row[field])
        if row["session_id"] is not None:
            uuid.UUID(row["session_id"])
        for field in ("created_at", "feedback_at"):
            if datetime.fromisoformat(row[field]).utcoffset() is None:
                raise ValueError("Timestamp must include timezone")
        text(row["question"], "Question", 4000)
        candidates = [validate_candidate(c) for c in row["candidates"]]
        order = row["display_order"]
        if sorted(order) != list(range(len(candidates))) or any(type(i) is not int for i in order):
            raise ValueError("Invalid display order")
        if row["kind"] == "correction":
            if (
                len(candidates) != 1
                or type(row["correct"]) is not bool
                or row["preference"] is not None
            ):
                raise ValueError("Invalid correction record")
            if row["corrected_answer"] is not None:
                text(row["corrected_answer"], "Corrected answer")
                if row["correct"]:
                    raise ValueError("Correct answers cannot have replacements")
            usable = row["correct"] or bool(row["corrected_answer"])
        elif row["kind"] == "preference":
            if (
                len(candidates) != 2
                or row["preference"] not in PREFERENCES
                or row["correct"] is not None
                or row["corrected_answer"] is not None
            ):
                raise ValueError("Invalid preference record")
            usable = row["preference"] in {"a", "b"}
            if usable and candidates[0]["answer"] == candidates[1]["answer"]:
                raise ValueError("Identical answers cannot form a directional preference")
        else:
            raise ValueError("Invalid feedback kind")
        expected = (
            order[0 if row["preference"] == "a" else 1] if row["preference"] in {"a", "b"} else None
        )
        if row["preferred_candidate_index"] != expected:
            raise ValueError("Preference mapping disagrees with display order")
        for field in ("protected", "software_test_only", "training_eligible"):
            if type(row[field]) is not bool:
                raise ValueError(f"{field} must be boolean")
        if row["split"] in {"validation", "test"} and not row["protected"]:
            raise ValueError("Held-out feedback must be protected")
        if (
            any(c["generation_parameters"]["backend"] == "mock" for c in candidates)
            and not row["software_test_only"]
        ):
            raise ValueError("Mock feedback must be marked software_test_only")
        eligible = (
            row["split"] == "train"
            and not row["protected"]
            and not row["software_test_only"]
            and bool(usable)
        )
        if row["training_eligible"] != eligible or (info["eligible_only"] and not eligible):
            raise ValueError("Invalid training eligibility")
        if row["image"] != f"images/{row['image_sha256']}.png":
            raise ValueError("Invalid image reference")
        image_path = (root / row["image"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Image path escapes export directory")
        with Image.open(image_path) as image:
            if image_digest(image) != row["image_sha256"]:
                raise ValueError("Image hash mismatch")
    if len(seen) != info["count"]:
        raise ValueError("Export count mismatch")
    return info
