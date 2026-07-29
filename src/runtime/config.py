"""Strict loading for the two formal YAML configuration layers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import math

import yaml

from .losses import LOSS_NAMES


class ConfigError(ValueError):
    """Raised when a formal configuration is missing or ambiguous."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated public experiment configuration.

    The object retains the exact YAML values. Runtime-only choices such as a
    device or a run id are deliberately kept outside this object.
    """

    source: Path
    values: dict[str, Any]

    @property
    def data(self) -> dict[str, Any]:
        return self.values["data"]

    @property
    def split(self) -> dict[str, Any]:
        return self.values["split"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.values["sampling"]

    @property
    def training(self) -> dict[str, Any]:
        return self.values["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.values["evaluation"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.values["runtime"]

    def copy_values(self) -> dict[str, Any]:
        return deepcopy(self.values)


_EXPERIMENT_KEYS = {"data", "split", "sampling", "training", "evaluation", "runtime"}
_DATA_KEYS = {
    "dataset",
    "data_root",
    "model_input_file",
    "eval_target_file",
    "turbine_id_column",
    "timestamp_column",
    "target_column",
    "input_power_column",
    "mask_column",
    "feature_columns",
    "num_nodes",
    "lookback",
    "max_pred_len",
    "eval_horizons",
}
_SPLIT_KEYS = {"method", "train_ratio", "val_ratio", "test_ratio"}
_SAMPLING_KEYS = {
    "train_stride",
    "val_stride",
    "test_stride",
    "train_shuffle",
    "val_shuffle",
    "test_shuffle",
    "loader_seed_offsets",
    "drop_last",
}
_TRAINING_KEYS = {
    "seed",
    "epochs",
    "effective_batch_size",
    "train_batch_size",
    "val_batch_size",
    "test_batch_size",
    "gradient_accumulation_steps",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "betas",
    "epsilon",
    "scheduler",
    "scheduler_factor",
    "scheduler_patience",
    "scheduler_threshold",
    "scheduler_threshold_mode",
    "scheduler_min_lr",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "loss",
    "amp",
    "amp_dtype",
    "amp_cache_enabled",
    "gradient_clip",
    "gradient_clip_norm_type",
    "gradient_clip_error_if_nonfinite",
    "gradient_clip_foreach",
}
_EVALUATION_KEYS = {
    "metrics",
    "physical_clip",
    "physical_min_kw",
    "physical_max_kw",
    "checkpoint_selection",
}
_CHECKPOINT_KEYS = {"split", "horizon", "metric", "mode"}
_RUNTIME_KEYS = {"num_workers", "deterministic", "save_predictions", "pin_memory"}

_MODEL_FORBIDDEN_KEYS = {
    "dataset",
    "data_root",
    "feature_columns",
    "target_column",
    "target",
    "target_mask",
    "mask_column",
    "lookback",
    "max_pred_len",
    "horizon",
    "eval_horizons",
    "batch_size",
    "train_batch_size",
    "val_batch_size",
    "test_batch_size",
    "epochs",
    "loss",
    "metrics",
    "early_stopping",
    "early_stopping_patience",
    "physical_clip",
    "physical_min_kw",
    "physical_max_kw",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "early_stopping",
    "seed",
    "optimizer",
    "learning_rate",
}
_MODEL_RUNTIME_KEYS = {"environment"}
_MODEL_ENVIRONMENTS = {"tslib", "tsl"}


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return value


def _keys(value: dict[str, Any], allowed: set[str], required: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown field at {path}: {unknown[0]}")
    missing = sorted(required - set(value))
    if missing:
        raise ConfigError(f"missing field at {path}: {missing[0]}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path} must be finite")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path} must be >= {minimum}")
    return number


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _validate_experiment(values: dict[str, Any]) -> None:
    _keys(values, _EXPERIMENT_KEYS, _EXPERIMENT_KEYS, "root")

    data = values["data"]
    _keys(data, _DATA_KEYS, _DATA_KEYS, "data")
    for field in (
        "dataset",
        "data_root",
        "model_input_file",
        "eval_target_file",
        "turbine_id_column",
        "timestamp_column",
        "target_column",
        "input_power_column",
        "mask_column",
    ):
        _string(data[field], f"data.{field}")
    features = data["feature_columns"]
    if not isinstance(features, list) or not features or not all(
        isinstance(item, str) and item for item in features
    ):
        raise ConfigError("data.feature_columns must be a non-empty list of strings")
    if len(set(features)) != len(features):
        raise ConfigError("data.feature_columns must not contain duplicates")
    forbidden = {
        data["target_column"],
        data["mask_column"],
        "Modification_Reason",
        "anomaly_label",
        "audit_label",
    }
    leaked = sorted(forbidden.intersection(features))
    if leaked:
        raise ConfigError(f"data.feature_columns contains forbidden input: {leaked[0]}")
    _integer(data["num_nodes"], "data.num_nodes", minimum=1)
    _integer(data["lookback"], "data.lookback", minimum=1)
    _integer(data["max_pred_len"], "data.max_pred_len", minimum=1)
    horizons = data["eval_horizons"]
    if not isinstance(horizons, list) or not horizons:
        raise ConfigError("data.eval_horizons must be a non-empty list")
    for index, horizon in enumerate(horizons):
        _integer(horizon, f"data.eval_horizons[{index}]", minimum=1)
        if horizon > data["max_pred_len"]:
            raise ConfigError("data.eval_horizons cannot exceed data.max_pred_len")
    if horizons != sorted(set(horizons)):
        raise ConfigError("data.eval_horizons must be sorted and unique")

    split = values["split"]
    _keys(split, _SPLIT_KEYS, _SPLIT_KEYS, "split")
    if split["method"] != "chronological_ratio":
        raise ConfigError("split.method must be chronological_ratio")
    ratios = [_number(split[name], f"split.{name}", minimum=0.0) for name in (
        "train_ratio", "val_ratio", "test_ratio"
    )]
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ConfigError("split ratios must be positive and sum to 1.0")

    sampling = values["sampling"]
    _keys(sampling, _SAMPLING_KEYS, _SAMPLING_KEYS, "sampling")
    for name in ("train_stride", "val_stride", "test_stride"):
        _integer(sampling[name], f"sampling.{name}", minimum=1)
    for name in ("train_shuffle", "val_shuffle", "test_shuffle", "drop_last"):
        _boolean(sampling[name], f"sampling.{name}")
    offsets = sampling["loader_seed_offsets"]
    if not isinstance(offsets, dict):
        raise ConfigError("sampling.loader_seed_offsets must be a mapping")
    _keys(offsets, {"train", "validation", "test"}, {"train", "validation", "test"}, "sampling.loader_seed_offsets")
    for name, offset in offsets.items():
        _integer(offset, f"sampling.loader_seed_offsets.{name}", minimum=0)

    training = values["training"]
    _keys(training, _TRAINING_KEYS, _TRAINING_KEYS, "training")
    _integer(training["seed"], "training.seed", minimum=0)
    for name in (
        "epochs",
        "train_batch_size",
        "val_batch_size",
        "test_batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "scheduler_patience",
        "early_stopping_patience",
    ):
        _integer(training[name], f"training.{name}", minimum=1)
    if training["effective_batch_size"] != training["train_batch_size"] * training["gradient_accumulation_steps"]:
        raise ConfigError("training.effective_batch_size must equal train_batch_size * gradient_accumulation_steps")
    if training["optimizer"] != "Adam":
        raise ConfigError("training.optimizer must be Adam")
    if training["scheduler"] != "reduce_on_plateau":
        raise ConfigError("training.scheduler must be reduce_on_plateau")
    if training["loss"] not in LOSS_NAMES:
        raise ConfigError(f"training.loss must be one of: {', '.join(LOSS_NAMES)}")
    learning_rate = _number(training["learning_rate"], "training.learning_rate", minimum=0.0)
    if learning_rate <= 0.0:
        raise ConfigError("training.learning_rate must be greater than 0")
    _number(training["weight_decay"], "training.weight_decay", minimum=0.0)
    betas = training["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ConfigError("training.betas must contain two numbers")
    for index, beta in enumerate(betas):
        value = _number(beta, f"training.betas[{index}]")
        if not 0.0 <= value < 1.0:
            raise ConfigError("training.betas must be in [0, 1)")
    _number(training["epsilon"], "training.epsilon", minimum=0.0)
    _number(training["scheduler_factor"], "training.scheduler_factor", minimum=0.0)
    if not 0.0 < float(training["scheduler_factor"]) < 1.0:
        raise ConfigError("training.scheduler_factor must be in (0, 1)")
    _number(training["scheduler_threshold"], "training.scheduler_threshold", minimum=0.0)
    if training["scheduler_threshold_mode"] not in {"rel", "abs"}:
        raise ConfigError("training.scheduler_threshold_mode must be rel or abs")
    _number(training["scheduler_min_lr"], "training.scheduler_min_lr", minimum=0.0)
    _number(training["early_stopping_min_delta"], "training.early_stopping_min_delta", minimum=0.0)
    for name in (
        "amp",
        "amp_cache_enabled",
        "gradient_clip_error_if_nonfinite",
        "gradient_clip_foreach",
    ):
        _boolean(training[name], f"training.{name}")
    if training["amp_dtype"] != "float16":
        raise ConfigError("training.amp_dtype must be float16")
    _number(training["gradient_clip"], "training.gradient_clip", minimum=0.0)
    _number(training["gradient_clip_norm_type"], "training.gradient_clip_norm_type", minimum=0.0)

    evaluation = values["evaluation"]
    _keys(evaluation, _EVALUATION_KEYS, _EVALUATION_KEYS, "evaluation")
    metrics = evaluation["metrics"]
    if not isinstance(metrics, list) or not metrics or not all(isinstance(item, str) for item in metrics):
        raise ConfigError("evaluation.metrics must be a non-empty list of strings")
    _boolean(evaluation["physical_clip"], "evaluation.physical_clip")
    for name in ("physical_min_kw", "physical_max_kw"):
        value = evaluation[name]
        if value is not None:
            _number(value, f"evaluation.{name}")
    if evaluation["physical_clip"]:
        if evaluation["physical_min_kw"] is None or evaluation["physical_max_kw"] is None:
            raise ConfigError("physical clipping requires both bounds")
        if evaluation["physical_min_kw"] > evaluation["physical_max_kw"]:
            raise ConfigError("physical_min_kw must not exceed physical_max_kw")
    selection = evaluation["checkpoint_selection"]
    _keys(selection, _CHECKPOINT_KEYS, _CHECKPOINT_KEYS, "evaluation.checkpoint_selection")
    if selection["split"] not in {"validation", "val"}:
        raise ConfigError("checkpoint_selection.split must be validation")
    if selection["horizon"] != "all" and selection["horizon"] not in data["eval_horizons"]:
        raise ConfigError("checkpoint_selection.horizon must be all or an eval horizon")
    if selection["metric"] not in metrics:
        raise ConfigError("checkpoint_selection.metric must be listed in evaluation.metrics")
    if selection["mode"] not in {"min", "max"}:
        raise ConfigError("checkpoint_selection.mode must be min or max")

    runtime = values["runtime"]
    _keys(runtime, _RUNTIME_KEYS, _RUNTIME_KEYS, "runtime")
    _integer(runtime["num_workers"], "runtime.num_workers", minimum=0)
    for name in ("deterministic", "save_predictions", "pin_memory"):
        _boolean(runtime[name], f"runtime.{name}")


def load_experiment_config(path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    """Load and strictly validate the one public experiment YAML."""

    source = Path(path).resolve()
    values = _load_mapping(source)
    _validate_experiment(values)
    return ExperimentConfig(source=source, values=values)


def _normalise_cli_overrides(cli_overrides: Any) -> dict[tuple[str, ...], Any]:
    """Convert a namespace or sparse mapping into YAML paths and values.

    The option-to-YAML mapping is owned by ``cli.command_schema``.  Keeping
    this conversion here means every command uses exactly the same application
    and validation path.
    """

    from cli.command_schema import PUBLIC_OVERRIDE_BY_DEST, PUBLIC_OVERRIDE_BY_PATH

    if cli_overrides is None:
        return {}
    if not isinstance(cli_overrides, Mapping):
        cli_overrides = vars(cli_overrides)

    result: dict[tuple[str, ...], Any] = {}

    def visit(mapping: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> None:
        for raw_key, value in mapping.items():
            key = str(raw_key)
            path_key = ".".join((*prefix, key))
            if not prefix and key in PUBLIC_OVERRIDE_BY_DEST:
                spec = PUBLIC_OVERRIDE_BY_DEST[key]
                if value is not None:
                    result[spec.yaml_path] = deepcopy(value)
                continue
            if path_key in PUBLIC_OVERRIDE_BY_PATH:
                spec = PUBLIC_OVERRIDE_BY_PATH[path_key]
                if value is not None:
                    result[spec.yaml_path] = deepcopy(value)
                continue
            if isinstance(value, Mapping):
                visit(value, (*prefix, key))

    visit(cli_overrides)
    expanded: dict[tuple[str, ...], Any] = {}
    from cli.command_schema import PUBLIC_OVERRIDE_BY_DEST

    for path, value in result.items():
        spec = PUBLIC_OVERRIDE_BY_DEST.get(".".join(path))
        if spec is None:
            spec = next((candidate for candidate in PUBLIC_OVERRIDE_BY_DEST.values() if candidate.yaml_path == path), None)
        if spec is None:
            expanded[path] = value
        else:
            for target in spec.yaml_paths:
                expanded[target] = deepcopy(value)
    result = expanded
    for path, value in list(result.items()):
        if isinstance(value, tuple):
            result[path] = list(value)
    return result


def cli_overrides_from_namespace(namespace: Any) -> dict[str, Any]:
    """Return explicit CLI overrides as dotted YAML paths."""

    return {
        ".".join(path): value
        for path, value in _normalise_cli_overrides(namespace).items()
    }


def cli_overrides_as_nested(cli_overrides: Any) -> dict[str, Any]:
    """Return only explicit overrides in the sparse nested YAML shape."""

    nested: dict[str, Any] = {}
    for path, value in _normalise_cli_overrides(cli_overrides).items():
        current = nested
        for part in path[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"conflicting CLI override path: {'.'.join(path)}")
            current = child
        current[path[-1]] = deepcopy(value)
    return nested


def _validate_feature_columns_against_dataset(
    values: dict[str, Any],
    *,
    project_root: Path,
) -> None:
    """Validate overridden feature names when the local parquet exists.

    Config-only commands must remain usable without the private dataset.  The
    full schema check is therefore performed opportunistically here and is
    repeated by ``data.loader`` before data is consumed.
    """

    data = values["data"]
    data_root = Path(data["data_root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root
    model_path = data_root / data["model_input_file"]
    if not model_path.is_file():
        return
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return
    names = set(parquet.read_schema(model_path).names)
    missing = sorted(set(data["feature_columns"]) - names)
    if missing:
        raise ConfigError(
            "data.feature_columns contains a column missing from the model input parquet: "
            f"{missing[0]}"
        )


def apply_cli_overrides(
    experiment_config: ExperimentConfig,
    cli_overrides: Any,
    *,
    project_root: Path | None = None,
) -> ExperimentConfig:
    """Apply explicit command-line values and validate the final config.

    ``None`` means no override.  The returned object always contains a deep
    copy, so the original YAML-derived configuration is never mutated.
    """

    if not isinstance(experiment_config, ExperimentConfig):
        raise TypeError("experiment_config must be an ExperimentConfig")
    values = experiment_config.copy_values()
    normalised = _normalise_cli_overrides(cli_overrides)
    for path, value in normalised.items():
        current: dict[str, Any] = values
        for part in path[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                raise ConfigError(f"invalid CLI override path: {'.'.join(path)}")
            current = child
        current[path[-1]] = deepcopy(value)

    # effective_batch_size is a derived field in the existing YAML contract;
    # changing the train batch must keep that invariant valid without making a
    # second command-line synonym for it.
    batch_path = ("training", "train_batch_size")
    if batch_path in normalised:
        values["training"]["effective_batch_size"] = (
            int(values["training"]["train_batch_size"])
            * int(values["training"]["gradient_accumulation_steps"])
        )

    _validate_experiment(values)
    root = (project_root or experiment_config.source.parent.parent).resolve()
    _validate_feature_columns_against_dataset(values, project_root=root)
    return ExperimentConfig(source=experiment_config.source, values=values)


def load_resolved_experiment_config(
    path: str | Path = "configs/experiment.yaml",
    cli_overrides: Any = None,
    *,
    project_root: Path | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load the public YAML and apply one shared set of CLI overrides."""

    base = load_experiment_config(path)
    resolved = apply_cli_overrides(base, cli_overrides, project_root=project_root)
    return resolved, cli_overrides_from_namespace(cli_overrides)


def _validate_model_value(value: Any, path: str) -> None:
    """Validate the portable scalar/container subset accepted in model YAML."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"{path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"{path} contains a non-string mapping key")
            _validate_model_value(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_model_value(child, f"{path}[{index}]")
        return
    raise ConfigError(f"{path} contains a value that is not YAML-serializable")


def _find_model_public_parameter(value: Any, path: str = "model") -> tuple[str, str] | None:
    """Find public experiment fields at any depth, including list mappings."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key in _MODEL_FORBIDDEN_KEYS:
                return key, f"{path}.{key}"
            result = _find_model_public_parameter(child, f"{path}.{key}")
            if result is not None:
                return result
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result = _find_model_public_parameter(child, f"{path}[{index}]")
            if result is not None:
                return result
    return None


def _load_model_document(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {source}")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid model YAML: {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"model configuration root must be a mapping: {source}")
    if not all(isinstance(key, str) for key in loaded):
        raise ConfigError("model configuration root keys must be strings")

    # Keep the error useful for users migrating a legacy flat file that leaked
    # a public experiment option, while all other root fields get the strict
    # new-shape error below.
    root_public = sorted(set(loaded) & _MODEL_FORBIDDEN_KEYS)
    if root_public:
        raise ConfigError(f"model config cannot define public parameter: {root_public[0]}")
    unknown_root = sorted(set(loaded) - {"runtime", "model"})
    if unknown_root:
        raise ConfigError(
            f"model config root may only contain runtime and model: {unknown_root[0]}"
        )

    runtime = loaded.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ConfigError("model.runtime must be a mapping")
    if not all(isinstance(key, str) for key in runtime):
        raise ConfigError("model.runtime keys must be strings")
    unknown_runtime = sorted(set(runtime) - _MODEL_RUNTIME_KEYS)
    if unknown_runtime:
        raise ConfigError(f"unknown field at runtime: {unknown_runtime[0]}")
    environment = runtime.get("environment", "tslib")
    if not isinstance(environment, str) or environment not in _MODEL_ENVIRONMENTS:
        raise ConfigError(
            "runtime.environment must be one of: "
            + ", ".join(sorted(_MODEL_ENVIRONMENTS))
        )

    model = loaded.get("model")
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping")
    if not model:
        raise ConfigError("model config must contain at least one model field")
    _validate_model_value(model, "model")
    forbidden = _find_model_public_parameter(model)
    if forbidden is not None:
        name, location = forbidden
        raise ConfigError(
            f"model config cannot define public parameter {name!r} at {location}"
        )
    return {
        "runtime": deepcopy(dict(runtime)),
        "model": deepcopy(dict(model)),
    }


def load_model_config_document(path: str | Path) -> dict[str, Any]:
    """Load and validate the complete ``runtime``/``model`` document."""

    return _load_model_document(path)


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load only model-owned parameters for ``build_model``.

    The runtime selection is deliberately removed at this boundary so model
    constructors can never receive ``runtime.environment`` accidentally.
    """

    return deepcopy(_load_model_document(path)["model"])


def model_runtime_environment(path: str | Path, *, default: str = "tslib") -> str:
    """Return the declared model environment, applying the configured default."""

    document = _load_model_document(path)
    environment = document["runtime"].get("environment", default)
    if environment not in _MODEL_ENVIRONMENTS:
        raise ConfigError(f"unsupported model runtime environment: {environment}")
    return str(environment)


def resolved_config_values(config: ExperimentConfig, *, project_root: Path) -> dict[str, Any]:
    """Return a serializable copy with non-semantic path resolution metadata."""

    values = config.copy_values()
    data_root = (project_root / values["data"]["data_root"]).resolve()
    values["resolved"] = {
        "config_file": str(config.source),
        "data_root": str(data_root),
        "model_input_file": str(data_root / values["data"]["model_input_file"]),
        "eval_target_file": str(data_root / values["data"]["eval_target_file"]),
    }
    return values
