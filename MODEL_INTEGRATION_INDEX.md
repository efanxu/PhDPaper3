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

All commands use `runtime.reproducibility_mode: controlled_nonstrict`: seeds and
DataLoader generators remain fixed, cuDNN algorithm selection remains fixed,
and global/default deterministic CUDA algorithms remain disabled. A model may
declare a narrowly scoped deterministic CUDA training/shape-validation
requirement when a concrete CUDA forward/backward nondeterminism has been
demonstrated; the current RA-DS-PFD relation-spatial adapters use this local
scope, while spatial-disabled RA-DS-PFD and unrelated models remain
non-strict. The local scope restores the previous global state immediately
afterward. Repeatability keeps structural fields exact and compares
floating-point losses, metrics and predictions with the recorded
absolute/relative tolerances.

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

## Universal Node-Shared execution

Temporal models that apply one independent, parameter-shared forecasting
function per node inherit `models.base.NodeSharedForecastModel`. The public
sample batch remains the `training.*_batch_size` value; node micro-batching is
a separate execution detail from the single public `runtime.node_shared_chunk_size`
setting, currently `32`. A node count need not divide evenly (134 becomes
`32/32/32/32/6`); the final range is natural, ordered, unpadded, and is never
dropped or reweighted. Every optimizer update still covers every node, and
masked loss uses global absolute-error sum, squared-error sum, and valid-target
count across all sample accumulation batches and node chunks.

The only runtime value source for `runtime.node_shared_chunk_size` is
`configs/experiment.yaml`; production code has no independent fallback, and the
resolved value must be passed into `build_execution_plan()`. Resume compatibility
compares the chunk only when the current plan uses `node_shared_microbatch`;
`full_nodes` and `full_spatiotemporal` checkpoints remain compatible when only the
chunk value changes. `FORMAL_DEFAULT_SHAPE` uses the resolved training AMP
autocast policy. If the target GPU is occupied, formal GPU validation remains
pending and is not a reason to lower the configured chunk.

`NodeSharedForecastModel` models without PyTorch batch-dependent normalization
use the shared node micro-batch executor. Native `_BatchNorm` modules force
full-node execution and must not be changed to LayerNorm; a formal full-shape
OOM in that case is recorded as OOM. True spatiotemporal models keep
`[B,L,N,C]` and use full execution. Chunk size is never model-specific and is
never changed automatically; compatible NodeShared OOM may only cause one
uniform public-config reduction after a formal target-GPU gate. `check`,
`train`, validation, `evaluate`, repeatability and Full all use this same
execution planner. New model adapters implement only their node-range forward;
they must not add chunk loops, loss accumulation, optimizer or timing paths.

## Minimal validation principle

Validation must be proportional to a concrete correctness or recovery risk.
Keep checks that prevent silent shape, dtype, node-order, edge-direction,
configuration, non-finite-value, checkpoint-compatibility or data-boundary
errors.

Do not add hashes, certificates, manifests, registries, duplicated identity
fields or fail-closed machinery unless there is a documented external trust
boundary, a concrete failure that the check prevents, and an actual consumer
that uses the result. Metadata that is already stored in readable form must
not receive an additional hash only for appearance of rigor.

Prefer simple versioned filenames, explicit schemas, direct structural
validation and ordinary tests. Reproducibility evidence may retain compact
state identities such as model `state_dict_hash` when they are actively used
to compare training states, but such identities must not be generalized into
a project-wide certificate system.

Before adding a new validation layer, confirm all three points:

1. the exact failure it prevents;
2. why existing structure or tests cannot detect that failure;
3. where the validation result is consumed.

If any point cannot be answered, do not add the validation layer.

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

### HANDOFF current-state rule

`HANDOFF.md` is a current-state handoff, not a changelog.  Update existing
sections in place and remove superseded phase status, historical test counts,
old HEADs and obsolete closeout records when a later state replaces them.
Keep only history that is still necessary to understand a current constraint
or avoid a known failure.  Git history is the authoritative historical record.
Do not grow `HANDOFF.md` by appending a new dated section for every maintenance
pass.  A new handoff should let a session without prior context see the
current real state, current limits, next step and important pitfalls.

## Commands

Use the generated reference for current options:

```powershell
python scripts\generate_command_reference.py
python scripts\generate_command_reference.py --check
```

Do not reintroduce StudySpec, ModelSpec, certificates, declarations,
manifests, readiness protocols or model-specific experiment documents.
