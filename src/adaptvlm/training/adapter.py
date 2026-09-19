"""Adapter provenance and integrity helpers shared by training and inference."""

import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    """Stream SHA-256 without loading an entire weight file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_base_hash(model_id: str) -> str | None:
    """Fingerprint local checkpoint files; remote bases use resolved Hub commits."""
    root = Path(model_id)
    if not root.is_dir():
        return None
    files = sorted(
        p for p in root.iterdir() if p.is_file() and (p.suffix in {".json", ".safetensors", ".bin"})
    )
    if not files:
        raise ValueError("Local base checkpoint has no configuration/weight files")
    return hashlib.sha256("\n".join(f"{p.name}:{file_hash(p)}" for p in files).encode()).hexdigest()


def read_adapter(path: str | Path) -> dict:
    """Require a completed AdaptVLM training artifact and unmodified adapter files."""
    root = Path(path).resolve()
    info = json.loads((root / "training_run.json").read_text(encoding="utf-8"))
    if info.get("schema_version") != 1 or info.get("status") != "complete":
        raise ValueError("Adapter training did not complete")
    hashes = info.get("adapter_files", {})
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(hashes):
        raise ValueError("Adapter is missing required file fingerprints")
    for name, expected in hashes.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or file_hash(candidate) != expected:
            raise ValueError("Adapter file integrity check failed")
    return info


def adapter_revision(config) -> tuple[str, dict | None]:
    """Pin adapted inference to its recorded base and reject incompatible configurations."""
    if not config.adapter_path:
        return config.revision, None
    info = read_adapter(config.adapter_path)
    base = info["model_config"]
    if config.model_id != base["model_id"] or config.family != base["family"]:
        raise ValueError("Adapter base model/family does not match inference configuration")
    revision = info["resolved_base_revision"] or base["revision"]
    if config.revision not in {"main", revision}:
        raise ValueError("Adapter base revision mismatch")
    if local_base_hash(config.model_id) != info["local_base_sha256"]:
        raise ValueError("Local base checkpoint differs from adapter's training base")
    return revision, info
