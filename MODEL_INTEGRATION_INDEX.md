# Model integration index

This is the only navigation entry for adding or auditing a model. Start every
model task with:

> 先读取仓库根目录 MODEL_INTEGRATION_INDEX.md，并按照其中的读取顺序完成本次模型接入。

## 1. Mandatory reading order

1. `configs/experiment.yaml`
2. `src/runtime/config.py`
3. `src/data/loader.py`
4. `src/data/split.py`
5. `src/data/normalization.py`
6. `src/data/window.py`
7. `src/data/dataset.py`
8. `src/data/dataloader.py`
9. `src/engine/reproducibility.py`
10. `src/engine/trainer.py`
11. `src/engine/losses.py`
12. `src/engine/metrics.py`
13. `src/engine/evaluator.py`
14. `src/models/base.py`
15. `src/models/loader.py`
16. `tests/test_model_interface.py`
17. `tests/test_full_shape.py`
18. `tests/test_repeatability.py`

Read these only when the model needs the corresponding approved resource:

- graph model: `src/resources/graph.py`
- static features: `src/resources/static_features.py`

## 2. Sources of truth

Public experiment parameters have one source: `configs/experiment.yaml`.

Model structure parameters have one source:
`configs/models/<model_name>.yaml`.

The loader rejects unknown fields and rejects public fairness parameters inside
model YAML. Runtime metadata, the resolved YAML copy and Git commit are
recorded automatically in a run directory; they are not a second protocol.

## 3. Where a model goes

Implement the model in:
`src/models/<model_name>/model.py`.

Put its structure values in:
`configs/models/<model_name>.yaml`.

The implementation must expose:

```python
def build_model(model_config, data_info):
    ...
```

The public input is `models.base.ModelInput`; labels and masks belong only to
the Trainer, loss and Evaluator. The standard output layout is the legacy
formal layout `(batch, nodes, horizon)`. A model must subclass
`ForecastModel`, validate its output shape, and keep model-specific logic out
of the data and training modules.

An Adapter is not the default. Add the smallest local conversion only when a
model cannot consume `ModelInput` directly; never duplicate windowing,
DataLoader, loss, Trainer, evaluator or checkpoint selection.

## 4. Default modification boundary

An ordinary model integration must not modify:

- split or normalization;
- window generation, target or mask semantics;
- loss, metrics or evaluator;
- early stopping or checkpoint selection;
- the formal public YAML.

If a shared implementation appears wrong, report the evidence first. Do not
silently change it to make one model run.

Do not introduce StudySpec, ModelSpec, model revisions, source closures,
runtime profiles, readiness, certificates, anchors, campaigns, membership
manifests, attestations, manual evidence indexes or model-specific protocol
documents.

## 5. Required checks

Run from the repository root with the configured interpreter:

```powershell
python -m compileall src scripts tests
python scripts/preflight.py --model <model_name> --config configs/experiment.yaml
python scripts/check_model.py --model <model_name> --config configs/experiment.yaml --model-config configs/models/<model_name>.yaml
python scripts/check_model.py --model <model_name> --config configs/experiment.yaml --model-config configs/models/<model_name>.yaml --full-shape
python -m pytest tests/test_model_interface.py -q
python -m pytest tests/test_full_shape.py -q
python scripts/run_model.py --model <model_name> --config configs/experiment.yaml --model-config configs/models/<model_name>.yaml --run-id <run_id> --smoke
python scripts/evaluate.py --model <model_name> --config configs/experiment.yaml --resume results/<model_name>/<run_id>/best.pt --run-id <eval_run_id>
python scripts/compare_repeated_runs.py --model <model_name> --config configs/experiment.yaml --model-config configs/models/<model_name>.yaml --seed <configured_seed>
python -m pytest -q
```

The full-shape check must use the formal batch, lookback, node, feature and
horizon values. A regular smoke may use explicit smoke limits and must be
reported separately. A successful integration leaves `best.pt`, `last.pt`,
resolved configuration, evaluation metrics and ordinary run metadata under
`results/<model_name>/<run_id>/`; generated results are ignored by Git.
