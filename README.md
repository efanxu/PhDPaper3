# PhDPaper3

PhDPaper3 is a small, reproducible SDWPF time-series forecasting framework.
Every model uses the same data loading, chronological split, train-only
normalization, windows, optimizer, masked loss, metrics and checkpoint path.

## Install and data

Use Python 3.11 or the existing environment:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe -m pip install -e .
```

Place the local SDWPF parquet files in `dataset/`; they are intentionally not
committed. Public experiment defaults are in `configs/experiment.yaml`, and
model structure values are in `configs/models/<model_name>.yaml`.

## One command entry point

All project tasks use:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py <command> [arguments]
```

The parser and current parameter-to-YAML mapping live in
`src/cli/command_schema.py`. Live help and the generated full reference are:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py --help
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py train --help
```

See [`docs/COMMAND_REFERENCE.md`](docs/COMMAND_REFERENCE.md) for generated
parameter types, defaults, YAML fields, groups and examples.

## Common commands

Default training inherits all public experiment values from YAML:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py train `
  --model node_shared_lstm `
  --run-id node_shared_lstm_seed2026 `
  --device cuda
```

An explicit override affects only this run and is recorded:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py train `
  --model node_shared_lstm `
  --run-id node_shared_lstm_batch4_seed2026 `
  --device cuda `
  --batch-size 4
```

Multiple overrides can be combined:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py train `
  --model node_shared_lstm `
  --run-id node_shared_lstm_custom_seed2026 `
  --device cuda `
  --batch-size 4 `
  --epochs 10 `
  --learning-rate 0.0005 `
  --lookback 96 `
  --eval-horizons 3 6 10
```

Evaluate a checkpoint with compatibility checks:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py evaluate `
  --model node_shared_lstm `
  --run-id node_shared_lstm_eval `
  --device cuda `
  --checkpoint results\node_shared_lstm\node_shared_lstm_seed2026\best.pt
```

Run shape, preflight and repeatability checks:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py check `
  --model node_shared_lstm --device cuda --full-shape --batch-size 4

D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py preflight `
  --model node_shared_lstm --device cuda --batch-size 4

D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py repeatability `
  --model node_shared_lstm --device cuda --batch-size 4
```

Run several models with one configuration and continue after a model error:

```powershell
D:\Apps\Miniconda3\envs\env_tslib\python.exe scripts\run.py batch `
  --models node_shared_lstm --device cuda --continue-on-error
```

The priority is:

```text
内置结构默认 < configs/experiment.yaml < 命令行显式覆盖
```

Each training/evaluation run stores the complete final configuration in
`resolved_config.yaml`, only explicit public overrides in `cli_overrides.yaml`,
and the original command in `command.json`. Results live under
`results/<model_name>/<run_id>/`.

For model integration instructions, read `MODEL_INTEGRATION_INDEX.md` first.
