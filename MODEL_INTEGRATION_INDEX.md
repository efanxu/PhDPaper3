# Model integration index

This is the navigation entry for adding a model. The public command and all
shared experiment behavior remain in `scripts/run.py` and the modules below.

## Read these shared files

1. `configs/experiment.yaml`
2. `configs/models/<model_name>.yaml`
3. `src/cli/command_schema.py`
4. `src/runtime/config.py`
5. `src/data/loader.py`
6. `src/data/split.py`
7. `src/data/normalization.py`
8. `src/data/window.py`
9. `src/data/dataset.py`
10. `src/data/dataloader.py`
11. `src/engine/reproducibility.py`
12. `src/engine/checkpoint.py`
13. `src/engine/trainer.py`
14. `src/engine/evaluator.py`
15. `src/runtime/performance.py`
16. `src/models/base.py`
17. `src/models/loader.py`
18. `src/cli/orchestrator.py`
19. `src/runtime/environments.py`
20. `src/integrations/time_series_library.py`

Use `src/resources/graph.py` or `src/resources/static_features.py` only when
the model genuinely needs the corresponding shared resource. `DataInfoView`
contains input-only feature metadata (`feature_columns`, `input_power_column`
and its resolved index); it never carries targets or masks.

## Add only the model implementation

Create:

```text
src/models/<model_name>/model.py
configs/models/<model_name>.yaml
```

The model YAML has only these root sections:

```yaml
runtime:
  environment: tslib  # or tsl; omitted means the default in configs/environments.yaml

model:
  # model-owned structure parameters only
```

Model parameters are passed to `build_model(model_config, data_info)` from the
`model:` mapping only. Time-series models are Node Shared unless the task
explicitly requires cross-node modelling. They are intentionally not copied into a shared field
allowlist or public command documentation.  Public experiment parameters are
rejected recursively if they appear inside model mappings or list items.

Model runtime selection comes from `configs/models/<model>.yaml`:

```text
runtime.environment: tslib
runtime.environment: tsl
```

When `runtime` is omitted, the default is `tslib`.  Environment definitions
come from `configs/environments.yaml`; the parent scheduler resolves the
target Python and starts one independent worker process per model.  Model code
must not activate Conda or start another Trainer.

The local `Time-Series-Library` tree is a read-only model source. Use
`src/integrations/time_series_library.py` for controlled explicit file-path
loading so the project's `models.base` remains the active package. Pure `tsl`
models import the formal `tsl` package directly and must not depend on
Time-Series-Library. Graph models obtain deterministic shared graph buffers
from `src/resources/graph.py`; non-graph models do not require graph files.

The scheduler resolves and preflights each distinct runtime environment once
per batch, then validates every requested model in its selected environment
before starting independent workers. A
new model automatically contributes its test metrics and performance values
to `summary.csv`, `performance_summary.csv` and
`model_comparison.csv`.  Model code must not implement a second Trainer,
timing path, environment switcher or result aggregator.

## Fixed acceptance sequence

New models complete, in order: `INTERFACE_SMALL`, environment preflight,
model preflight, `RESOLVED_SHAPE`, Smoke, `FORMAL_DEFAULT_SHAPE`, and
Repeatability before a formal Full run. Every `train` command independently
runs `RESOLVED_SHAPE` before the training worker. Smoke validates the resolved
command batch; `check --full-shape` with no public shape override validates the
YAML-default formal batch. Report batch-1 and default-batch results separately.

Environment preflight does not validate the real data node order. Data
preflight, resolved/formal shape checks and Smoke validate node IDs against
the public graph resources. Persisted states classify configuration, resource,
OOM and worker-crash failures separately.

The model module must expose:

```python
def build_model(model_config, data_info):
    ...
```

The returned object subclasses `models.base.ForecastModel`, consumes
`models.base.ModelInput`, and returns `(batch, nodes, horizon)`. Labels and
masks stay in the shared Trainer and Evaluator.

## Automatically inherited behavior

The shared path provides single-model training, multi-model subprocess
isolation, smoke/full-shape checks, epoch checkpoints, compatible resume,
archive-and-replace, ID-suffix reruns, evaluate-only runs, repeatability,
training and inference timing, throughput, parameter counts, GPU memory
measurement and CSV aggregation. New model code must not implement its own
CLI, Trainer, Evaluator, checkpoint format, resume logic, result directory or
subprocess scheduler.

## Stable result contracts

Paper metrics are read from the top level of each `metrics_test_h*.json`.
The paper CSV groups 3-step, 6-step and 10-step metrics, while display floats
use exactly three decimals. `model_comparison.csv` is the two-row paper view;
`model_comparison_flat.csv` is the programmatic one-row view. Both can be
regenerated from existing artifacts with `summarize`, without training.

Status schema v2 guarantees only the small top-level state vocabulary and the
stable failure classifications; specific details belong in `error` and
`details`. Status readers must accept existing schema-v1 records. Public graph
configuration is owned by `resources.graph.k` in `experiment.yaml`; graph-model
checkpoints must validate the graph configuration before resume or evaluation.

The model YAML is structure-only. Public data, split, loss, optimizer, batch
and evaluation semantics come from `configs/experiment.yaml` and explicit
command-line overrides.

## Documentation impact rules

Reading this index never modifies Markdown.  Adding an ordinary model
parameter changes only the model YAML and model code; it does not require
updating `README.md`, this index, `HANDOFF.md` or
`docs/COMMAND_REFERENCE.md`.  A change to `src/cli/command_schema.py` must
regenerate and check `docs/COMMAND_REFERENCE.md`.  Changes to required paths,
the model YAML shape, `build_model`, `ModelInput`, output shape, environment
selection or universal model checks update this index.  Core scheduler,
environment or limitation changes update `HANDOFF.md`.  `README.md` changes
only for stable user-entry changes such as installation, data location,
commands, help or result locations.

## Commands

Use the generated reference for current options:

```powershell
python scripts\generate_command_reference.py
python scripts\generate_command_reference.py --check
```

Do not reintroduce StudySpec, ModelSpec, certificates, declarations,
manifests, readiness protocols or model-specific experiment documents.
