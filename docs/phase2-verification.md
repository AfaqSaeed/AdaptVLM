# Phase 2 verification â€” 2026-11-04

Implemented SQLite feedback storage, immutable request/image snapshots, correctness and optional
corrections, randomized pairwise preferences, validation, portable JSONL/image exports, and a
Gradio review/export dashboard. No training or preference optimization was implemented.

## Verified

- Full pytest suite: **40 passed**, including the 11 Phase 1 checks.
- Ruff lint and formatting checks passed.
- Phase 1 mock baseline evaluation still generates its report.
- Local Gradio HTTP/API smoke test generated an answer, saved a correction, generated a response
  pair, saved a both-acceptable vote, refreshed review counts, and downloaded an audit ZIP.
  It verified 2 audit records and 0 training-eligible records for mock/held-out fixtures, then
  shut down the server and removed its temporary database.
- Storage tests cover persistence after reopen, both display orders and all four preference
  choices, duplicate/concurrent submissions, identical candidates, invalid feedback, failed
  generations, mock/synthetic exclusion, unassigned and held-out exclusion, late manifest
  registration, irreversible image protection, image metadata removal, and model-free CLI use.
- Export tests cover round-trip validation, tampered image detection, invalid vote mapping,
  refusal to overwrite, and missing-image failure without partial publication.
- Actual UI callback regression test covers API sessions without page-load events, session
  continuity/separation, stale-state clearing after failed requests, duplicate reviews and ties.

The HTTP smoke test caught a Gradio callable-state initialization issue for API clients.
Session UUIDs now initialize lazily in the generation callback; a regression test covers it.
Gradio 5 emits deprecation warnings for private event listeners (`api_name=False`); the project
constrains Gradio below 6. These warnings do not fail the tests.

## Verification boundaries

Tests use mocks or fabricated candidates, not real human research feedback. They establish
collection/export correctness, not model quality or adaptation benefits. No real checkpoint,
GPU, SFT or DPO run was performed in Phase 2. Phase 1's real-model download limitation remains.
The UI was exercised through HTTP/API and callbacks; browser visual layout/manual upload was
not inspected. Local socket checks ran outside the Windows sandbox restriction documented in
Phase 1. No demo feedback is included as a research dataset.

Phase 3 was subsequently implemented after user approval; see [Phase 3 verification](phase3-verification.md).
