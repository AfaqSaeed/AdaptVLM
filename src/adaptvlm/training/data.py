"""Strict conversion of correction feedback into portable SFT examples."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from adaptvlm.datasets.manifest import load_manifest
from adaptvlm.feedback.store import image_digest, text
from adaptvlm.feedback.validation import validate_export


def protection(manifests: list[str | Path]) -> tuple[set[str], list[dict]]:
    """Collect current protected image hashes; exclude all synthetic fixtures as well."""
    if not manifests:
        raise ValueError("Supply at least one current evaluation/protected manifest")
    hashes, provenance = set(), []
    for path in manifests:
        dataset, digest = load_manifest(path)
        provenance.append({"name": dataset["name"], "sha256": digest})
        for row in dataset["examples"]:
            if row["split"] != "train" or dataset["synthetic"]:
                hashes.add(row["image_sha256"])
                with Image.open(row["image_path"]) as image:
                    hashes.add(image_digest(ImageOps.exif_transpose(image)))
    return hashes, provenance


def prepare_sft(
    bundle: str | Path, output: str | Path, protected_manifests: list[str | Path]
) -> dict:
    """Convert only human-corrected answers; reject conflicting labels and stale exports."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    info = validate_export(bundle)
    if not info["eligible_only"]:
        raise ValueError("SFT requires an eligible-only feedback export, not an audit bundle")
    protected, provenance = protection(protected_manifests)
    rows, skipped, seen = [], 0, {}
    for line in (bundle / "feedback.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["image_sha256"] in protected:
            raise ValueError("Feedback overlaps a currently protected image; regenerate the export")
        # Deliberately do not convert preference winners or uncorrected model answers.
        if record["kind"] != "correction" or record["correct"] or not record["corrected_answer"]:
            skipped += 1
            continue
        key = (record["image_sha256"], record["question"].strip())
        answer = record["corrected_answer"].strip()
        if key in seen:
            if seen[key]["answer"] != answer:
                raise ValueError("Conflicting corrections for the same image and question")
            seen[key]["feedback_ids"].append(record["feedback_id"])
            continue
        row = {
            "id": record["feedback_id"],
            "image": record["image"],
            "image_sha256": record["image_sha256"],
            "question": record["question"],
            "answer": answer,
            "split": "train",
            "feedback_ids": [record["feedback_id"]],
        }
        rows.append(row)
        seen[key] = row
    if not rows:
        raise ValueError("No eligible human corrections available for SFT")
    if output.exists():
        raise ValueError("SFT output already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sft-", dir=output.parent))
    try:
        (staging / "images").mkdir()
        for row in rows:
            shutil.copyfile(bundle / row["image"], staging / row["image"])
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (staging / "examples.jsonl").write_bytes(payload)
        metadata = {
            "schema_version": 1,
            "kind": "correction-sft",
            "count": len(rows),
            "software_test_only": False,
            "source_feedback_sha256": info["sha256"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "protected_manifests": provenance,
            "skipped_noncorrections": skipped,
        }
        (staging / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        load_sft(staging, protected_manifests)
        staging.rename(output)
        return metadata
    finally:
        if staging.exists():
            if staging.resolve().parent != output.parent or not staging.name.startswith(".sft-"):
                raise RuntimeError("Unexpected SFT staging path")
            shutil.rmtree(staging)


def load_sft(
    directory: str | Path, protected_manifests: list[str | Path]
) -> tuple[list[dict], dict]:
    """Recheck train-only rows, content integrity and current protected sets before training."""
    root = Path(directory).resolve()
    info = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = (root / "examples.jsonl").read_bytes()
    if (
        info.get("schema_version") != 1
        or info.get("kind") != "correction-sft"
        or info.get("software_test_only") is not False
    ):
        raise ValueError("Expected a real correction-SFT bundle")
    if hashlib.sha256(payload).hexdigest() != info.get("sha256"):
        raise ValueError("SFT data hash mismatch")
    protected, current = protection(protected_manifests)
    if not {p["sha256"] for p in info["protected_manifests"]}.issubset(
        {p["sha256"] for p in current}
    ):
        raise ValueError("Supply every manifest used during conversion, plus current held-out sets")
    rows, seen_ids, seen_keys = [], set(), set()
    for line in payload.decode().splitlines():
        row = json.loads(line)
        if row["split"] != "train" or row["image_sha256"] in protected:
            raise ValueError("Held-out/protected data cannot be used for SFT")
        text(row["question"], "Question", 4000)
        text(row["answer"], "Answer")
        key = (row["image_sha256"], row["question"])
        if row["id"] in seen_ids or key in seen_keys:
            raise ValueError("Duplicate SFT example")
        seen_ids.add(row["id"])
        seen_keys.add(key)
        path = (root / row["image"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes SFT bundle")
        with Image.open(path) as image:
            if image_digest(image) != row["image_sha256"]:
                raise ValueError("SFT image hash mismatch")
        rows.append(row | {"image_path": str(path)})
    if not rows or len(rows) != info["count"]:
        raise ValueError("Empty SFT bundle or incorrect row count")
    return rows, info
