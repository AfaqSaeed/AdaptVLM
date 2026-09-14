"""SQLite feedback snapshots, split protection and portable JSONL exports."""

import hashlib
import json
import math
import secrets
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageOps

from adaptvlm.datasets.manifest import load_manifest

SPLITS = {"unassigned", "train", "validation", "test"}
PREFERENCES = {"a", "b", "both", "neither"}


def timestamp() -> str:
    """Return an unambiguous UTC timestamp."""
    return datetime.now(UTC).isoformat()


def text(value: str, field: str, limit: int = 20000) -> str:
    """Require bounded, nonblank text without coercing arbitrary values."""
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field} must be non-empty text of at most {limit} characters")
    return value.strip()


def image_digest(image: Image.Image) -> str:
    """Match Phase 1's hash of decoded RGB pixels and dimensions."""
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def validate_candidate(candidate: dict) -> dict:
    """Validate provenance and measurements for one generated response."""
    if not isinstance(candidate, dict):
        raise ValueError("Candidate must be an object")
    candidate = dict(candidate)
    for field in ("answer", "model_id"):
        candidate[field] = text(candidate.get(field), field)
    if not isinstance(candidate.get("generation_parameters"), dict):
        raise ValueError("generation_parameters must be an object")
    parameters = candidate["generation_parameters"]
    if parameters.get("backend") not in {"mock", "transformers"}:
        raise ValueError("Candidate must record its backend")
    if parameters.get("model_id") != candidate["model_id"]:
        raise ValueError("Candidate model ID disagrees with generation parameters")
    if candidate.get("error"):
        raise ValueError("Failed generations cannot receive feedback")
    latency = candidate.get("latency_seconds")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise ValueError("latency_seconds must be a nonnegative number")
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("latency_seconds must be a nonnegative finite number")
    json.dumps(candidate, allow_nan=False)
    return candidate


class FeedbackStore:
    """Own a local database and content-addressed image files beside it.

    Connections are short-lived and transactions serialize writers. The database is a
    trusted local resource, not a multi-user authorization boundary.
    """

    def __init__(self, database: str | Path):
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.images = self.database.parent / f"{self.database.stem}_images"
        self.images.mkdir(exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise ValueError(f"Unsupported feedback database version: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    sha256 TEXT PRIMARY KEY,
                    protected INTEGER NOT NULL DEFAULT 0 CHECK(protected IN (0,1)),
                    synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN (0,1))
                );
                CREATE TABLE IF NOT EXISTS manifests (
                    sha256 TEXT PRIMARY KEY, name TEXT NOT NULL, registered_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY,
                    image_sha256 TEXT NOT NULL REFERENCES images(sha256),
                    question TEXT NOT NULL,
                    split TEXT NOT NULL CHECK(split IN ('unassigned','train','validation','test')),
                    kind TEXT NOT NULL CHECK(kind IN ('correction','preference')),
                    candidates TEXT NOT NULL,
                    display_order TEXT NOT NULL,
                    session_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE REFERENCES requests(id),
                    correct INTEGER CHECK(correct IN (0,1)),
                    corrected_answer TEXT,
                    preference TEXT CHECK(preference IN ('a','b','both','neither')),
                    created_at TEXT NOT NULL,
                    CHECK((correct IS NOT NULL AND preference IS NULL) OR
                          (correct IS NULL AND preference IS NOT NULL))
                );
                CREATE INDEX IF NOT EXISTS request_image ON requests(image_sha256);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connection(self):
        """Open a transactional connection with foreign keys enabled."""
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def register_manifest(self, path: str | Path) -> None:
        """Protect validation/test pixels, including feedback collected previously.

        Also register EXIF-oriented hashes to catch uploads normalized by the UI.
        Protection is monotonic: registering a train manifest cannot undo it.
        """
        data, fingerprint = load_manifest(path)
        entries = []
        for row in data["examples"]:
            with Image.open(row["image_path"]) as image:
                hashes = {row["image_sha256"], image_digest(ImageOps.exif_transpose(image))}
            for digest in hashes:
                entries.append((digest, int(row["split"] != "train"), int(data["synthetic"])))
        with self.connection() as db:
            db.executemany(
                """
                INSERT INTO images(sha256,protected,synthetic) VALUES(?,?,?)
                ON CONFLICT(sha256) DO UPDATE SET
                    protected=MAX(protected,excluded.protected),
                    synthetic=MAX(synthetic,excluded.synthetic)
            """,
                entries,
            )
            db.execute(
                "INSERT OR IGNORE INTO manifests VALUES(?,?,?)",
                (fingerprint, data["name"], timestamp()),
            )

    def create_request(
        self,
        image: Image.Image,
        question: str,
        candidates: list[dict],
        split: str = "unassigned",
        session_id: str | None = None,
    ) -> dict:
        """Persist immutable response provenance and a randomized display order."""
        question = text(question, "Question", 4000)
        if split not in SPLITS:
            raise ValueError("Invalid data split")
        if not isinstance(image, Image.Image):
            raise ValueError("Provide an image")
        if not isinstance(candidates, list) or len(candidates) not in {1, 2}:
            raise ValueError("A request needs one response or a pair")
        candidates = [validate_candidate(c) for c in candidates]
        if session_id is not None:
            # Only a random UUID is accepted, not an email or other personal identifier.
            if not isinstance(session_id, str):
                raise ValueError("Session identifier must be a UUID string")
            session_id = str(uuid.UUID(session_id))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.info.clear()  # Persist pixels, not EXIF/GPS or other uploaded metadata.
        digest = image_digest(image)
        destination = self.images / f"{digest}.png"
        # Each writer uses its own temp file, then atomically installs identical pixels.
        with tempfile.NamedTemporaryFile(dir=self.images, suffix=".png", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            image.save(temporary, format="PNG")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        order = list(range(len(candidates)))
        secrets.SystemRandom().shuffle(order)
        request_id = str(uuid.uuid4())
        kind = "correction" if len(candidates) == 1 else "preference"
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO images(sha256) VALUES(?)", (digest,))
            if split in {"validation", "test"}:
                db.execute("UPDATE images SET protected=1 WHERE sha256=?", (digest,))
            db.execute(
                "INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    digest,
                    question,
                    split,
                    kind,
                    json.dumps(candidates, allow_nan=False),
                    json.dumps(order),
                    session_id,
                    timestamp(),
                ),
            )
        return self.get_request(request_id)

    def get_request(self, request_id: str) -> dict:
        """Return the saved snapshot and its displayed responses."""
        with self.connection() as db:
            row = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown response; generate an answer first")
        result = dict(row)
        result["candidates"] = json.loads(result["candidates"])
        result["display_order"] = json.loads(result["display_order"])
        result["displayed_answers"] = [
            result["candidates"][i]["answer"] for i in result["display_order"]
        ]
        return result

    def add_correction(
        self, request_id: str, correct: bool, corrected_answer: str | None = None
    ) -> str:
        """Record correctness and an optional replacement; reject repeat submissions."""
        request = self.get_request(request_id)
        if request["kind"] != "correction" or type(correct) is not bool:
            raise ValueError("Select correct or incorrect for a single response")
        corrected_answer = (
            corrected_answer.strip() if isinstance(corrected_answer, str) else corrected_answer
        )
        if corrected_answer:
            corrected_answer = text(corrected_answer, "Corrected answer")
        if correct and corrected_answer:
            raise ValueError("A correct answer must not have a replacement; select incorrect")
        if corrected_answer is not None and not isinstance(corrected_answer, str):
            raise ValueError("Corrected answer must be text")
        return self._insert_feedback(request_id, int(correct), corrected_answer or None, None)

    def add_preference(self, request_id: str, preference: str) -> str:
        """Save A/B/both/neither relative to the persisted randomized display order."""
        request = self.get_request(request_id)
        if request["kind"] != "preference" or preference not in PREFERENCES:
            raise ValueError("Select A, B, both or neither for a response pair")
        answers = request["displayed_answers"]
        if preference in {"a", "b"} and answers[0].strip() == answers[1].strip():
            raise ValueError("Identical answers cannot form a directional preference")
        return self._insert_feedback(request_id, None, None, preference)

    def _insert_feedback(self, request_id, correct, corrected, preference) -> str:
        feedback_id = str(uuid.uuid4())
        try:
            with self.connection() as db:
                db.execute(
                    "INSERT INTO feedback VALUES(?,?,?,?,?,?)",
                    (
                        feedback_id,
                        request_id,
                        correct,
                        corrected,
                        preference,
                        timestamp(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Feedback was already submitted for this response") from exc
        return feedback_id

    def records(self) -> list[dict]:
        """Read feedback with current split protection, preserving ties and rejections."""
        with self.connection() as db:
            rows = db.execute("""
                SELECT r.*, f.id AS feedback_id, f.correct, f.corrected_answer,
                       f.preference, f.created_at AS feedback_at, i.protected, i.synthetic
                FROM feedback f JOIN requests r ON r.id=f.request_id
                JOIN images i ON i.sha256=r.image_sha256
                ORDER BY f.created_at, f.id
            """).fetchall()
        result = []
        for raw in rows:
            row = dict(raw)
            candidates = [validate_candidate(c) for c in json.loads(row.pop("candidates"))]
            order = json.loads(row.pop("display_order"))
            if sorted(order) != list(range(len(candidates))):
                raise ValueError("Corrupt display order in feedback database")
            row["candidates"] = candidates
            row["display_order"] = order
            row["correct"] = None if row["correct"] is None else bool(row["correct"])
            row["schema_version"] = 1
            row["image"] = f"images/{row['image_sha256']}.png"
            row["software_test_only"] = bool(row.pop("synthetic")) or any(
                c["generation_parameters"]["backend"] == "mock" for c in candidates
            )
            row["protected"] = bool(row["protected"])
            # These raw records are not SFT/DPO examples. Conversion is Phase 3/4.
            row["training_eligible"] = (
                row["split"] == "train"
                and not row["protected"]
                and not row["software_test_only"]
                and (
                    (
                        row["kind"] == "correction"
                        and (row["correct"] or bool(row["corrected_answer"]))
                    )
                    or (row["kind"] == "preference" and row["preference"] in {"a", "b"})
                )
            )
            row["preferred_candidate_index"] = (
                order[0 if row["preference"] == "a" else 1]
                if row["preference"] in {"a", "b"}
                else None
            )
            result.append(row)
        return result

    def summary(self) -> dict:
        """Return aggregate collection counts for the local dashboard."""
        rows = self.records()
        with self.connection() as db:
            requests = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        return {
            "requests": requests,
            "feedback": len(rows),
            "pending": requests - len(rows),
            "corrections": sum(r["kind"] == "correction" for r in rows),
            "preferences": {
                p: sum(r["preference"] == p for r in rows) for p in sorted(PREFERENCES)
            },
            "training_eligible": sum(r["training_eligible"] for r in rows),
            "protected": sum(r["protected"] for r in rows),
            "software_test_only": sum(r["software_test_only"] for r in rows),
        }

    def export(self, output: str | Path, eligible_only: bool = False) -> dict:
        """Atomically export a validated snapshot with images; never overwrite a bundle."""
        output = Path(output).resolve()
        if output.exists():
            raise ValueError("Export directory already exists; choose a new directory")
        rows = self.records()
        with self.connection() as db:
            manifests = [
                dict(row)
                for row in db.execute("SELECT * FROM manifests ORDER BY sha256").fetchall()
            ]
        if eligible_only:
            rows = [r for r in rows if r["training_eligible"]]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".feedback-export-", dir=output.parent))
        try:
            (temporary / "images").mkdir()
            for row in rows:
                source = self.images / f"{row['image_sha256']}.png"
                with Image.open(source) as image:
                    if image_digest(image) != row["image_sha256"]:
                        raise ValueError("Stored image hash mismatch")
                shutil.copyfile(source, temporary / row["image"])
            with (temporary / "feedback.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            info = {
                "schema_version": 1,
                "created_at": timestamp(),
                "count": len(rows),
                "eligible_only": eligible_only,
                "registered_manifests": manifests,
                "purpose": "raw feedback; audit exports may contain held-out data; not training input",
                "sha256": hashlib.sha256((temporary / "feedback.jsonl").read_bytes()).hexdigest(),
            }
            (temporary / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            from adaptvlm.feedback.validation import validate_export

            validate_export(temporary)
            temporary.rename(output)
            return info
        finally:
            if temporary.exists():
                if temporary.resolve().parent != output.parent or not temporary.name.startswith(
                    ".feedback-export-"
                ):
                    raise RuntimeError("Refusing to remove an unexpected export staging directory")
                shutil.rmtree(temporary)
