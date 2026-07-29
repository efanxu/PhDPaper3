# Migration report

## Scope and boundary

The only writable repository is `D:\PaperProject\PhDPaper3`. The old project
`D:\PaperProject\PhDPaper3_old2` was read as a local, read-only reference. The
new framework does not import it, link to it, or require any of its artifacts at
runtime.

## Old-project sources actually read

The following files were inspected for migration semantics:

- `configs/project_profiles/sdwpf_base_v1.yaml`: dataset, feature, target,
  mask, split, window, scaler, loss, metric and training declarations.
- `configs/models/benchmark/node_shared_lstm.yaml` and
  `studies/benchmarks/node_shared_lstm/model_config.yaml`: the reference model
  structure and its historical derived dimensions.
- `studies/benchmarks/node_shared_lstm/study.yaml`: frozen benchmark choices,
  input capability boundary, batch size and output identity.
- `configs/comparison_groups/sdwpf_base_v1_benchmarks.yaml`: Adam, scheduler,
  checkpoint tie-breaking, shuffle, DataLoader and accumulation semantics.
- `configs/parameters/registry.yaml`: defaults for optimizer, AMP, scheduler,
  clipping, loader seeds and early stopping.
- `src/phdpaper3_experiments/engine/data.py`: parquet reads, array layout,
  frozen scalers, window dataset normalization and batch layout.
- `src/phdpaper3_experiments/engine/training.py`: accumulated hybrid loss,
  AMP, gradient clipping, optimizer step, checkpoint save/reload and evaluation
  orchestration.
- `src/phdpaper3_experiments/engine/formal.py`: formal loader construction,
  seed streams, ReduceLROnPlateau, early stopping, best-checkpoint selection,
  reload and test evaluation.
- `src/phdpaper3_experiments/evaluation/evaluator.py`: model-owned metric
  entry point.
- `src/phdpaper3_protocol/scoring.py`: exact loss, MAE, RMSE, R2, SMAPE, MAPE
  and SDWPF Official Score formulas.
- `src/phdpaper3_protocol/batch.py`: label-isolated model input bundle and
  legacy `(batch, nodes, horizon)` target shape.
- `src/phdpaper3_models/node_shared_lstm.py` and
  `src/phdpaper3_models/common.py`: NodeSharedLSTM graph and shape invariants.
- `artifacts/sdwpf_base_v1/manifests/data_manifest.json`,
  `feature_manifest.json`, `timestamp_manifest.json` and
  `turbine_order.json`: row/key order, feature order, timestamps and nodes.
- `artifacts/sdwpf_base_v1/splits/split_boundaries.json` and
  `split_manifest.json`: exact chronological boundaries.
- `artifacts/sdwpf_base_v1/windows/window_manifest.json` and the four window
  arrays: counts, starts, stride and filtered all-invalid train windows.
- `artifacts/sdwpf_base_v1/scalers/scaler_manifest.json` plus the two scaler
  arrays: train-only fit scopes and statistics.
- `tests/test_engine_contract.py`, `tests/test_model.py` and
  `tests/test_historical_compatibility.py`: executable shape, loss, AMP,
  accumulation, checkpoint and historical behavior expectations.

The old repository contains no raw SDWPF parquet files. The local new-project
files under `dataset/` are the protocol-named cleaned files used for the data
comparison below.

## Public behavior mapping

| Old behavior | New implementation |
| --- | --- |
| Protocol-named parquet reads and time/node grid validation | `src/data/loader.py` |
| Time-major node order and feature order | `src/data/loader.py` + `configs/experiment.yaml` |
| Chronological floor-based split | `src/data/split.py` |
| Train-only population standardization | `src/data/normalization.py` |
| Lookback/horizon/stride and all-invalid train filtering | `src/data/window.py` |
| History input, normalized target and boolean mask | `src/data/dataset.py` |
| Shuffle, batch sizes, worker seeds and loader generators | `src/data/dataloader.py` |
| Python/NumPy/PyTorch/CUDA deterministic setup | `src/engine/reproducibility.py` |
| Masked score-aligned hybrid loss | `src/engine/losses.py` |
| MAE/RMSE/R2/SMAPE/MAPE/official score | `src/engine/metrics.py` |
| Validation and H3/H6/H10 test evaluation | `src/engine/evaluator.py` |
| Adam, ReduceLROnPlateau, clipping and early stopping | `src/engine/trainer.py` |
| Best/last checkpoint and reload | `src/engine/checkpoint.py` |
| NodeSharedLSTM computation graph | `src/models/node_shared_lstm/model.py` |
| Model import convention | `src/models/base.py` and `src/models/loader.py` |
| Ordinary environment/run metadata | `src/runtime/environment.py` and `src/runtime/run_info.py` |

## Consistency evidence

The audited new-project parquet files match the old artifacts on the following
invariants:

- 7,042,906 rows form 52,559 timestamps × 134 nodes.
- Start/end are `2021-01-01 00:10` and `2021-12-31 23:50`; node order is 1–134.
- The 16 feature columns and their order are unchanged.
- Split ends are 42,047 and 47,303, yielding 42,047/5,256/5,256 timestamps.
- Window starts are train 6,983, filtered train 6,632, validation 1,701 and
  test 5,103, with 351 all-invalid train candidates removed.
- First/last forecast starts are 144/42,036, 42,191/47,291 and 47,447/52,549.
- Input means/scales and valid-train target mean/scale match the old scaler
  artifacts within `1e-5` absolute/relative tolerance. The largest independently
  recomputed input scale difference before float32 serialization was about
  `7.1e-8`; target mean/scale differences were below `2e-13`.

These checks are encoded in `tests/test_data_pipeline.py` against the small
`tests/fixtures/legacy_manifest.json`; no old-project path is needed to run the
new tests.

## Deliberately not migrated

The old StudySpec/ModelSpec/revision/source-closure/runtime-profile/readiness,
certificate/anchor/campaign/manifest/attestation and public protocol packages
were not migrated. They do not contribute to the four goals of the new
framework and would recreate the complexity explicitly excluded by the task.
Git commit, environment, resolved config and ordinary run metadata remain
automatic but are not a governance protocol.

Only NodeSharedLSTM is migrated in this pass. Graph, static-feature and other
model code is not copied until the common path is accepted. The old formal
implementation did not apply physical power clipping; the new formal config
therefore keeps `evaluation.physical_clip: false` rather than silently changing
the metric behavior. Bounds are available only for an explicitly approved
future change.

## Verification results

The final statuses below are updated after the current environment checks:

| Check | Status | Evidence |
| --- | --- | --- |
| `python -m compileall src scripts tests` | PASS | Python 3.11.15, no compile errors |
| Configuration validation and forbidden model fields | PASS | Preflight passed; strict-field tests passed |
| Data pipeline and old-manifest comparison | PASS | `tests/test_data_pipeline.py`: 2 passed |
| Model interface/import | PASS | `tests/test_model_interface.py`: 4 passed; parameter count 31,978 |
| Full-shape forward/backward | PASS | GPU `[32,144,134,16] -> [32,134,10]`, finite output/gradients |
| Short smoke training | PASS | 1 epoch, 2 updates, validation/test limited to 2 batches |
| Checkpoint save/reload | PASS | evaluate-only completed; first 64 validation/test predictions matched exactly |
| Repeatability comparison | PASS | 2 runs, max prediction difference 0.0, metric difference 0.0 |
| Full pytest | PASS | 8 passed in 9.09 seconds |

## Known risks

- Formal AMP remains CUDA-only to preserve the old explicit no-CPU-fallback
  behavior. CPU-only machines can still run non-AMP interface tests, but cannot
  claim a formal AMP training result without an explicit configuration change.
- The local data loader materializes the cleaned arrays in memory. This is
  appropriate for the current SDWPF baseline and is simpler than the old frozen
  artifact protocol; a larger dataset would need a streaming implementation.
- Physical clipping is documented as absent in the old implementation and is
  disabled by default in the new one.

## Next model migration

Read `MODEL_INTEGRATION_INDEX.md`, add one directory under
`src/models/<model_name>/` and one structure YAML under `configs/models/`, then
run the listed import, full-shape, smoke, reload and repeatability checks. Do
not modify public data or evaluation modules for model-specific convenience.
