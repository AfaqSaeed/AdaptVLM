# Phase 6 verification

Verified on November 4, 2026. The [research report](../reports/research-report.md) covers every
requested Phase 6 section. Research ablations are explicitly proposed, not presented as executed.

The fresh full suite passed **65 tests in 38.58 seconds**, with no failures, errors or skips.
There were 24 existing Gradio deprecation warnings. Socket-dependent tests ran outside the
Windows sandbox restriction, as in earlier phases. The local JUnit record is
`reports/phase6-validation.xml`; its SHA-256 is retained in the evidence snapshot.

The [evidence extractor](../scripts/build_report_evidence.py) was run against that JUnit result
and the saved Phase 5 tiny-model benchmark. It validates sample counts and normalized scores,
retains actual predictions and measurement settings, and hashes source files and raw inputs.
The resulting [snapshot](../reports/evidence/phase6-evidence.json) is intended for version control.
It records the dirty working tree: HEAD alone does not identify the tested source.

No new full-model or real-data experiment was performed in Phase 6. The retained float32 and
dynamic-int8 measurements concern a randomly initialized miniature Idefics3 model and synthetic
images. CUDA NF4 was unsupported. Neither adaptation benefits nor practical efficiency gains
have been established. The report states these limits beside the results.
