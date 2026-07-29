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
`configs/models/<model>.yaml`.

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
  --model node_shared_lstm `
  --run-id formal_seed2026 `
  --device cuda
```

Multiple models run one after another in independent Python processes:

```powershell
python scripts\run.py train `
  --model node_shared_lstm dlinear patchtst `
  --run-id benchmark_seed2026 `
  --device cuda
```

Continue, archive-and-replace, or create a new ID explicitly:

```powershell
python scripts\run.py train --model node_shared_lstm dlinear --run-id benchmark_seed2026 --resume
python scripts\run.py train --model node_shared_lstm --run-id formal_seed2026 --overwrite
python scripts\run.py train --model node_shared_lstm --run-id formal_seed2026 --id-suffix rerun1
```

Repeatability uses two independent workers per model:

```powershell
python scripts\run.py repeatability --model node_shared_lstm --run-id repeatability_seed2026 --device cuda
```

Results are stored in `results/<model>/<run_id>/`. Multi-model dispatch
metadata and paper-ready efficiency rows are stored in
`results/_runs/<effective_run_id>/`, including `performance_summary.csv`.

For new model integration, read
[`MODEL_INTEGRATION_INDEX.md`](MODEL_INTEGRATION_INDEX.md) first.
