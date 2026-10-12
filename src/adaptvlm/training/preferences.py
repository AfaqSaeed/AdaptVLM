"""Portable genuine-pair feedback conversion with current held-out exclusion."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from adaptvlm.feedback.store import image_digest, text, validate_candidate
from adaptvlm.feedback.validation import validate_export
from adaptvlm.training.data import protection


def prepare_preferences(
    bundle: str | Path, output: str | Path, protected_manifests: list[str | Path]
) -> dict:
    """Resolve randomized A/B votes; never manufacture pairs from corrections or ties."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    source = validate_export(bundle)
    if not source["eligible_only"]:
        raise ValueError("Preference conversion requires an eligible-only feedback export")
    protected, provenance = protection(protected_manifests)
    rows, seen, skipped = [], {}, 0
    for line in (bundle / "feedback.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["image_sha256"] in protected:
            raise ValueError("Feedback overlaps a currently protected image")
        if record["kind"] != "preference" or record["preference"] not in {"a", "b"}:
            skipped += 1
            continue
        winner = record["preferred_candidate_index"]
        chosen = record["candidates"][winner]["answer"].strip()
        rejected = record["candidates"][1 - winner]["answer"].strip()
        if chosen == rejected:
            raise ValueError("Identical responses are not directional preference data")
        key = (record["image_sha256"], record["question"], tuple(sorted((chosen, rejected))))
        if key in seen:
            if seen[key]["chosen"] != chosen:
                raise ValueError("Conflicting votes for the same response pair")
            seen[key]["feedback_ids"].append(record["feedback_id"])
            continue
        row = {
            "id": record["feedback_id"],
            "feedback_ids": [record["feedback_id"]],
            "image": record["image"],
            "image_sha256": record["image_sha256"],
            "question": record["question"],
            "chosen": chosen,
            "rejected": rejected,
            "split": "train",
            "candidates": record["candidates"],
            "preferred_candidate_index": winner,
            "display_order": record["display_order"],
            "preference": record["preference"],
        }
        rows.append(row)
        seen[key] = row
    if not rows:
        raise ValueError("No eligible directional human preferences available")
    if output.exists():
        raise ValueError("Preference output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preferences-", dir=output.parent))
    try:
        (staging / "images").mkdir()
        for row in rows:
            shutil.copyfile(bundle / row["image"], staging / row["image"])
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (staging / "pairs.jsonl").write_bytes(payload)
        info = {
            "schema_version": 1,
            "kind": "pairwise-preferences",
            "count": len(rows),
            "software_test_only": False,
            "source_feedback_sha256": source["sha256"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "protected_manifests": provenance,
            "skipped_nonpreferences": skipped,
        }
        (staging / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        load_preferences(staging, protected_manifests)
        staging.rename(output)
        return info
    finally:
        if staging.exists():
            if staging.resolve().parent != output.parent or not staging.name.startswith(
                ".preferences-"
            ):
                raise RuntimeError("Unexpected preference staging path")
            shutil.rmtree(staging)


def load_preferences(
    directory: str | Path, protected_manifests: list[str | Path]
) -> tuple[list[dict], dict]:
    """Validate data integrity, winner mapping, train-only provenance and current exclusions."""
    root = Path(directory).resolve()
    info = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = (root / "pairs.jsonl").read_bytes()
    if (
        info.get("schema_version") != 1
        or info.get("kind") != "pairwise-preferences"
        or info.get("software_test_only") is not False
    ):
        raise ValueError("Expected real pairwise-preference data")
    if hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise ValueError("Preference file hash mismatch")
    protected, current = protection(protected_manifests)
    if not {p["sha256"] for p in info["protected_manifests"]}.issubset(
        {p["sha256"] for p in current}
    ):
        raise ValueError("Supply all original protected manifests plus current held-out sets")
    rows, keys, ids = [], set(), set()
    for line in payload.decode().splitlines():
        row = json.loads(line)
        row["candidates"] = [validate_candidate(c) for c in row["candidates"]]
        for field in ("question", "chosen", "rejected"):
            text(row[field], field)
        if row["split"] != "train" or row["image_sha256"] in protected:
            raise ValueError("Protected images cannot be used for DPO")
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError("Identical preference responses")
        if row["preference"] not in {"a", "b"} or sorted(row["display_order"]) != [0, 1]:
            raise ValueError("Invalid directional preference")
        winner = row["display_order"][0 if row["preference"] == "a" else 1]
        if (
            row["preferred_candidate_index"] != winner
            or len(row["candidates"]) != 2
            or row["chosen"] != row["candidates"][winner]["answer"].strip()
            or row["rejected"] != row["candidates"][1 - winner]["answer"].strip()
        ):
            raise ValueError("Preference mapping mismatch")
        if any(c["generation_parameters"]["backend"] == "mock" for c in row["candidates"]):
            raise ValueError("Mock preferences are not training data")
        key = (
            row["image_sha256"],
            row["question"],
            tuple(sorted((row["chosen"], row["rejected"]))),
        )
        if key in keys or row["id"] in ids:
            raise ValueError("Duplicate preference pair or ID")
        keys.add(key)
        ids.add(row["id"])
        path = (root / row["image"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image escapes preference directory")
        with Image.open(path) as image:
            if image_digest(image) != row["image_sha256"]:
                raise ValueError("Preference image hash mismatch")
        rows.append(row | {"image_path": str(path)})
    if not rows or len(rows) != info["count"]:
        raise ValueError("Empty preference dataset or count mismatch")
    return rows, info
