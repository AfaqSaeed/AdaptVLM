# Phase 2 feedback design

## Storage and provenance

The default database is `data/feedback.sqlite3`; change it with `--database`.
Canonical RGB PNG images live in a sibling `feedback_images` directory, named by decoded-pixel
SHA-256. Uploaded EXIF and other image metadata are stripped from persisted images. Back up
both the database and its image directory. Git ignores the default data directory, database
files and image-store directories. Keep custom export locations out of version control.

SQLite schema version 1 uses images, manifests, requests and feedback tables. Parameterized SQL,
foreign keys, transactions and a unique feedback-per-request constraint prevent orphaned votes
and duplicate submissions. Per-operation connections support simultaneous local browser sessions.
A saved request is immutable: image reference/hash, question, declared split, model IDs,
configured and resolved checkpoint revisions, generation settings, measurements, creation time,
and candidate display order. Feedback adds correctness/replacement or A/B/both/neither plus its
own UTC timestamp. Session IDs are random UUIDs; no names or emails are requested.

The local Gradio app is intended for a trusted operator, without public sharing or authentication.
All local users can review/export all feedback. It is not an internet-facing multi-user service.
Requests that fail inference are not stored as rateable answers. Failed UI generations clear the
previous response state. Double submission returns an error instead of silently replacing a vote.
To change a review, generate a new request; in-place feedback editing is not implemented.

## Preference semantics

Candidate order is randomized with SystemRandom and persisted before display. Each candidate
retains its model/configuration provenance. A/B refers to displayed position; exported
`preferred_candidate_index` maps the vote back to the original candidates array. Both/neither
have no winner and never become directional preferences. The app does not expose candidate
identity next to A/B while voting. Identical answers cannot receive A/B votes.

Use `--comparison-config configs/comparison.yaml` for an alternate prompt on the same checkpoint.
Changing only generation settings reuses loaded weights and a shared inference lock. A distinct
model config loads separate weights. Prompt variants do not guarantee distinct responses or
constitute baseline-versus-adapted evaluation. No fabricated alternatives or preferences are used.

## Split and eligibility rules

Uploads default to `unassigned`. Explicitly choose `train` only for adaptation data. All
validation/test images in registered manifests are protected by decoded-pixel hashes, including
EXIF-oriented variants. Declaring an upload validation/test also protects other feedback on that
image. Protection is persistent and monotonic; subsequent train assignments cannot remove it.
The current evaluation manifest is automatically registered at app startup and before CLI
summary/export. Additional manifests must be registered with repeatable `--protected-manifest`.
Late registration also excludes previously collected matching feedback from future exports.

Training eligibility requires all of:
- Explicit train assignment, no registered validation/test match.
- No mock candidate and no registered synthetic fixture image.
- A correct response, an incorrect response with a nonblank correction, or a directional A/B vote.

Incorrect-without-correction, ties, neither, unassigned and held-out feedback are preserved for
review/audit but excluded from eligible exports. The exporter does not turn eligible records into
training samples; downstream phases must validate again and honor eligibility flags.

Exact decoded-pixel matching catches identical images with differing file metadata or PNG
encoding. It does not detect resized, recompressed or near-duplicate images, unknown held-out
sets, or pretraining contamination. Register all known datasets; the system cannot infer the
provenance of an arbitrary upload. Previously exported bundles are immutable snapshots: rebuild
exports after registering new datasets and recheck protection before future training.

## Portable export schema

Each new bundle contains:
- `feedback.jsonl`: one schema-version-1 raw record per review, with image reference, request and
  feedback UUIDs/timestamps, question, candidates/provenance, display order, declared split,
  correction/preference, protection and software-test flags, eligibility, and resolved winner index.
- `images/<sha256>.png`: only the images referenced by included records.
- `manifest.json`: record count, mode, creation time, registered manifest fingerprints, and
  SHA-256 of the JSONL file.

The standalone validator checks required fields and types, candidate provenance, UUIDs,
timezones, image paths/hashes, unique IDs, vote mapping, feedback consistency and eligibility.
Export stages in a temporary sibling directory, validates, then renames; it refuses overwrite.
Corrupt/missing images fail the export without publishing a partial bundle. The UI packages the
validated directory into a ZIP. Hashes establish integrity, not authenticity against someone
who rewrites both the data and manifest. Review/export snapshots reflect the registry at export;
never use old bundles to bypass later protection.

Audit exports preserve every review, including held-out data. The UI defaults to eligible-only;
the CLI requires `--eligible-only` for that mode. Empty eligible exports are valid and explicitly
report count 0. Training remains Phase 3, and preference optimization remains Phase 4.
