"""Load selected Time-Series-Library modules without shadowing ``src/models``.

The upstream repository is treated as a read-only source tree.  Modules are
loaded from explicit file paths under a private namespace, while its ordinary
``layers`` imports are available through a short-lived, controlled source-root
entry.  The project's ``models`` package remains first on ``sys.path`` and is
never replaced in ``sys.modules``.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from collections.abc import Iterator
from typing import Any

import torch


_NAMESPACE = "_phdpaper3_time_series_library"


def resolve_time_series_library_model(
    model_name: str,
    *,
    source_root: str | Path,
) -> Path:
    """Resolve ``<source_root>/models/<model_name>.py`` by exact path."""

    if not model_name or Path(model_name).name != model_name or "/" in model_name or "\\" in model_name:
        raise ValueError(f"invalid Time-Series-Library model name: {model_name!r}")
    root = Path(source_root).resolve()
    path = root / "models" / f"{model_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Time-Series-Library model does not exist: {path}")
    return path


@contextmanager
def _controlled_source_root(source_root: Path) -> Iterator[None]:
    """Expose upstream imports without replacing the project's ``models`` package.

    Crossformer imports both ``layers.*`` and ``models.PatchTST``.  Those are
    installed only while its source file executes; all original project modules
    and temporary upstream aliases are restored before this context exits.
    """

    original = list(sys.path)
    root_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    # The project source path is expected to be first.  Inserting at index 1
    # keeps it ahead of the upstream root even when callers use a bare Python
    # interpreter instead of scripts/run.py.
    insert_at = 1 if sys.path else 0
    sys.path.insert(insert_at, root_text)
    aliases = ("layers", "models")
    original_modules = {
        name: value
        for name, value in sys.modules.items()
        if any(name == alias or name.startswith(f"{alias}.") for alias in aliases)
    }
    for name in list(original_modules):
        sys.modules.pop(name, None)
    for alias in aliases:
        package = ModuleType(alias)
        package.__path__ = [str(source_root / alias)]  # type: ignore[attr-defined]
        package.__package__ = alias
        sys.modules[alias] = package
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == alias or name.startswith(f"{alias}.") for alias in aliases):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original


def load_time_series_library_model(
    model_name: str,
    *,
    source_root: str | Path,
    module_name: str | None = None,
) -> ModuleType:
    """Load one upstream model module under a unique internal module name."""

    path = resolve_time_series_library_model(model_name, source_root=source_root)
    root = Path(source_root).resolve()
    safe_name = model_name.replace("-", "_")
    qualified_name = module_name or f"{_NAMESPACE}.models.{safe_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    # Ensure the project package exists before the controlled context swaps it
    # temporarily.  Restoring the exact object is the isolation invariant.
    importlib.import_module("models")
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for upstream model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        with _controlled_source_root(root):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def load_time_series_library_model_class(
    model_name: str,
    *,
    source_root: str | Path,
    class_name: str = "Model",
) -> type:
    """Load and return the conventional upstream ``Model`` class."""

    module = load_time_series_library_model(model_name, source_root=source_root)
    value = getattr(module, class_name, None)
    if not isinstance(value, type):
        raise TypeError(f"upstream model {model_name!r} does not define class {class_name!r}")
    return value


def resolve_time_series_library_source_root(
    project_root: str | Path | None,
    *,
    model_name: str,
) -> Path:
    """Resolve the read-only upstream source root for a project model."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = root / "Time-Series-Library"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"{model_name} requires Time-Series-Library source root: {source_root}"
        )
    return source_root


def validate_time_series_library_data_info(data_info, *, model_name: str) -> None:
    """Fail closed on the input-only metadata required by a temporal adapter."""

    raw_feature_columns = getattr(data_info, "feature_columns", ())
    input_power_column = getattr(data_info, "input_power_column", "")
    input_power_index = getattr(data_info, "input_power_index", -1)
    num_features = getattr(data_info, "num_features", 0)
    num_nodes = getattr(data_info, "num_nodes", 0)
    lookback = getattr(data_info, "lookback", 0)
    horizon = getattr(data_info, "max_pred_len", 0)
    integer_values = (input_power_index, num_features, num_nodes, lookback, horizon)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
        raise ValueError(f"{model_name} requires integer input metadata")
    feature_columns = tuple(raw_feature_columns) if isinstance(raw_feature_columns, (tuple, list)) else ()
    if (
        not feature_columns
        or not all(isinstance(column, str) and column for column in feature_columns)
        or len(feature_columns) != num_features
        or num_features < 1
    ):
        raise ValueError(
            f"{model_name} requires feature_columns aligned with num_features"
        )
    if not isinstance(input_power_column, str) or not input_power_column:
        raise ValueError(f"{model_name} requires input_power_column metadata")
    if not 0 <= input_power_index < num_features:
        raise ValueError(f"{model_name} requires a valid input_power_index")
    if feature_columns[input_power_index] != input_power_column:
        raise ValueError(
            f"{model_name} input_power_index does not match input_power_column"
        )
    if num_nodes < 1 or lookback < 1 or horizon < 1:
        raise ValueError(
            f"{model_name} requires positive num_nodes, lookback and max_pred_len"
        )


def validate_time_series_library_config_fields(
    model_config: dict[str, Any],
    *,
    model_name: str,
    fields: set[str],
) -> None:
    """Validate the exact model-owned field set before loading upstream code."""

    if not isinstance(model_config, dict):
        raise ValueError(f"{model_name} model config must be a mapping")
    unknown = sorted(set(model_config) - fields)
    missing = sorted(fields - set(model_config))
    if unknown:
        raise ValueError(f"{model_name} model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"{model_name} model config is missing field: {missing[0]}")


def run_time_series_library_forecast(
    x: torch.Tensor,
    upstream: torch.nn.Module,
    *,
    horizon: int,
    input_power_index: int,
    model_name: str,
    decoder_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Run one upstream forecast on flattened node-batch history and select power."""

    if x.ndim != 4:
        raise ValueError(f"{model_name} helper expects x with shape (B, L, K, C)")
    batch, steps, nodes, channels = x.shape
    node_history = x.permute(0, 2, 1, 3).contiguous().view(
        batch * nodes, steps, channels
    )
    upstream_output = upstream(node_history, None, decoder_input, None)
    expected = (batch * nodes, int(horizon), channels)
    if not isinstance(upstream_output, torch.Tensor):
        raise TypeError(f"upstream {model_name} output must be a torch.Tensor")
    if tuple(upstream_output.shape) != expected:
        raise ValueError(
            f"upstream {model_name} output must have shape {expected}, "
            f"got {tuple(upstream_output.shape)}"
        )
    if not torch.isfinite(upstream_output).all():
        raise FloatingPointError(f"upstream {model_name} output contains NaN or Inf")
    output = upstream_output[..., int(input_power_index)].reshape(
        batch, nodes, int(horizon)
    )
    return output, int(batch), int(nodes)
