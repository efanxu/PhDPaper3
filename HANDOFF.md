# Handoff

## Current goal

Keep `PhDPaper3` as a lightweight, unified, reproducible and fair forecasting
framework. `PhDPaper3_old2` is read-only reference material and must never be
modified or imported at runtime.

## Completed

- Audited the old formal SDWPF data, window, normalization, NodeSharedLSTM,
  loss, metrics, training and checkpoint behavior.
- Added strict public and model YAML loading.
- Added shared parquet loading, chronological split, train-only normalization,
  sliding windows, Dataset and deterministic DataLoaders.
- Added one Trainer, loss, metrics, Evaluator, checkpoint and reproducibility
  implementation.
- Added the NodeSharedLSTM reference model with the old `(B, N, H)` output.
- Added model checks, training/evaluate-only/repeatability scripts and tests.

## Layout and commands

Read `MODEL_INTEGRATION_INDEX.md` for the fixed reading order. The main command
is `python scripts/run_model.py`; `--smoke` is a short, explicitly recorded
runtime limit. `scripts/check_model.py --full-shape` validates the formal tensor
shape. `scripts/compare_repeated_runs.py --seed 2026` compares two short runs.

## Verification state

Compile, strict config/data checks, model interface, formal GPU full-shape,
short smoke training, checkpoint reload, repeatability and the complete pytest
suite have passed. Exact commands and results are recorded in
`docs/MIGRATION_REPORT.md`. A formal 20-epoch run was not started; the smoke
run is intentionally a separate, explicitly limited artifact.

## Known constraints and next step

The formal configuration enables the old CUDA AMP setting, so a formal training
run requires CUDA; the code intentionally refuses a silent CPU fallback. Run
the ordered checks, fix only observed implementation errors, and do not
optimize the reference algorithm during migration. Other models can be added
after NodeSharedLSTM is fully accepted.

Do not reintroduce StudySpec, ModelSpec, model revisions, source closures,
runtime profiles, readiness, certificates, anchors, campaigns, manifests,
attestations or hand-maintained capability/evidence systems.

## Repository boundary

All changes, tests, docs, commits and pushes belong only to
`D:\PaperProject\PhDPaper3`. Do not modify, move, rename, delete, commit in or
run against `D:\PaperProject\PhDPaper3_old2`; do not restore its deleted remote
or create links/runtime dependencies between the projects.
