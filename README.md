# PhDPaper3

PhDPaper3 is a reproducible SDWPF time-series forecasting framework. Shared
data loading, training, checkpointing, evaluation and performance measurement
are used by every model.

## Install and data

```powershell
python -m pip install -e .
```

Place the two protocol-named local parquet files in `dataset/`. Raw data,
checkpoints, predictions and logs are ignored by Git. Public experiment
defaults are in `configs/experiment.yaml`; model structure values are in
`configs/models/<model>.yaml`. Each model YAML may declare
`runtime.environment: tslib` or `runtime.environment: tsl`; omitted runtime
uses `tslib`. The parent command resolves the corresponding Conda Python
automatically. Use `--environment-preflight-only` on `train`, `check`,
`preflight` or `repeatability` to check the environments without starting
model workers.

## One command entry point

All user tasks use `scripts/run.py`:

```powershell
python scripts\run.py --help
python scripts\run.py train --help
```

The complete generated parameter reference is
[`docs/COMMAND_REFERENCE.md`](docs/COMMAND_REFERENCE.md).

## Stable examples

Single model:

```powershell
python scripts\run.py train `
  --model lstm `
  --run-id formal_seed2026 `
  --device cuda
```

Multiple models run one after another in independent Python processes:

```powershell
python scripts\run.py train `
  --model lstm crossformer stcn `
  --run-id mixed_models_seed2026 `
  --device cuda `
  --smoke `
  --batch-size 1 `
  --eval-batch-size 1
```

Every training command first runs an isolated shape check for its resolved
configuration. New model integration should also run an independent default
formal-shape check with no public shape overrides:

```powershell
python scripts\run.py check `
  --model lstm crossformer stcn `
  --run-id formal_shape_seed2026 `
  --device cuda `
  --full-shape
```

Environment-only validation:

```powershell
python scripts\run.py train `
  --model lstm `
  --run-id environment_preflight `
  --device cuda `
  --environment-preflight-only
```

Continue, archive-and-replace, or create a new ID explicitly:

```powershell
python scripts\run.py train --model lstm crossformer --run-id benchmark_seed2026 --resume
python scripts\run.py train --model lstm --run-id formal_seed2026 --overwrite
python scripts\run.py train --model lstm --run-id formal_seed2026 --id-suffix rerun1
```

Repeatability uses two independent workers per model:

```powershell
python scripts\run.py repeatability --model lstm --run-id repeatability_seed2026 --device cuda
```

Results are stored in `results/<model>/<run_id>/`. Multi-model dispatch
metadata and paper-ready efficiency rows are stored in
`results/_runs/<effective_run_id>/`, including `summary.csv`,
`performance_summary.csv` and the complete `model_comparison.csv`.
`model_comparison.csv` is the two-row grouped paper/Excel view, while
`model_comparison_flat.csv` is the one-row programmatic view. Existing completed
runs can be summarized without retraining:

```text
python scripts/run.py summarize --run-id <run-id>
```

Temporal NodeShared baselines share parameters across turbines. Their public
sample batch is unchanged; the shared GPU execution layer uses node chunk 32 by
default.

Stable status locations are:

```text
Batch status:       results/_runs/<run-id>/status.json
Model status:       results/<model>/<run-id>/run_info.json
Standalone checks:  results/_checks/<check-id>/
```

For new model integration, read
[`MODEL_INTEGRATION_INDEX.md`](MODEL_INTEGRATION_INDEX.md) first.
