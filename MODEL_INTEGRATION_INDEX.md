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

Use `src/resources/graph.py` or `src/resources/static_features.py` only when
the model genuinely needs the corresponding shared resource.

## Add only the model implementation

Create:

```text
src/models/<model_name>/model.py
configs/models/<model_name>.yaml
```

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

The model YAML is structure-only. Public data, split, loss, optimizer, batch
and evaluation semantics come from `configs/experiment.yaml` and explicit
command-line overrides.

## Commands

Use the generated reference for current options:

```powershell
python scripts\generate_command_reference.py
python scripts\generate_command_reference.py --check
```

Do not reintroduce StudySpec, ModelSpec, certificates, declarations,
manifests, readiness protocols or model-specific experiment documents.
