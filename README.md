# PhDPaper3

PhDPaper3 is a small time-series forecasting experiment framework. Every model
uses the same SDWPF data loading, chronological split, train-only
normalization, windows, optimizer, masked loss, metrics, checkpoint selection
and evaluate-only path so model comparisons remain fair and repeatable.

## Install

Use Python 3.11 or the existing environment:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe -m pip install -e .
```

## Data and configuration

The local SDWPF parquet files belong in `dataset/`. They are intentionally not
committed. The single formal public configuration is
`configs/experiment.yaml`; model structure values are in
`configs/models/<model_name>.yaml`.

## Run one model

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run_model.py `
  --model node_shared_lstm `
  --config configs\experiment.yaml `
  --model-config configs\models\node_shared_lstm.yaml `
  --run-id node_shared_lstm_seed2026 `
  --smoke
```

Full formal training omits `--smoke`. Full-shape interface validation is:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\check_model.py `
  --model node_shared_lstm `
  --config configs\experiment.yaml `
  --model-config configs\models\node_shared_lstm.yaml `
  --full-shape
```

Repeatability is checked with two short runs:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\compare_repeated_runs.py `
  --model node_shared_lstm `
  --config configs\experiment.yaml `
  --model-config configs\models\node_shared_lstm.yaml `
  --seed 2026
```

Results are stored under `results/<model_name>/<run_id>/`. The directory
contains resolved configuration, environment/run metadata, best and last
checkpoints, history, metrics and optional predictions.

To add a model, first read `MODEL_INTEGRATION_INDEX.md`; normally only one
model Python file and one model YAML are needed.
