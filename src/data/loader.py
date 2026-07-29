"""Read the protocol-named SDWPF parquet files into one time-major layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from runtime.config import ExperimentConfig
from runtime.paths import resolve_data_root


@dataclass(frozen=True)
class DataArrays:
    """Aligned arrays with shape ``x[T, N, C]``, ``target[T, N]``."""

    x: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    timestamps: np.ndarray
    node_ids: np.ndarray


@dataclass(frozen=True)
class DataInfo:
    num_time_steps: int
    num_nodes: int
    num_features: int
    lookback: int
    max_pred_len: int
    feature_columns: tuple[str, ...]
    node_ids: tuple[int, ...]
    start_timestamp: str
    end_timestamp: str

    @classmethod
    def from_arrays(cls, arrays: DataArrays, config: ExperimentConfig) -> "DataInfo":
        data = config.data
        timestamps = arrays.timestamps.astype("datetime64[ns]")
        return cls(
            num_time_steps=int(arrays.x.shape[0]),
            num_nodes=int(arrays.x.shape[1]),
            num_features=int(arrays.x.shape[2]),
            lookback=int(data["lookback"]),
            max_pred_len=int(data["max_pred_len"]),
            feature_columns=tuple(data["feature_columns"]),
            node_ids=tuple(int(value) for value in arrays.node_ids),
            start_timestamp=str(timestamps[0]),
            end_timestamp=str(timestamps[-1]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_time_steps": self.num_time_steps,
            "num_nodes": self.num_nodes,
            "num_features": self.num_features,
            "lookback": self.lookback,
            "max_pred_len": self.max_pred_len,
            "feature_columns": list(self.feature_columns),
            "node_ids": list(self.node_ids),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }


def _array_from_column(path: Path, column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    return table.column(column).combine_chunks().to_numpy(zero_copy_only=False)


def _read_feature_matrix(path: Path, columns: list[str]) -> np.ndarray:
    table = pq.read_table(path, columns=columns)
    values = [
        table.column(column).combine_chunks().to_numpy(zero_copy_only=False)
        for column in columns
    ]
    return np.column_stack(values).astype(np.float32, copy=False)


def _grid_sort_order(timestamps: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    time_int = timestamps.astype("datetime64[ns]").astype(np.int64, copy=False)
    return np.lexsort((node_ids.astype(np.int64, copy=False), time_int))


def _validate_or_sort_grid(
    timestamps: np.ndarray,
    node_ids: np.ndarray,
    *,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(timestamps) != len(node_ids):
        raise ValueError("timestamp and node-id columns have different lengths")
    order = _grid_sort_order(timestamps, node_ids)
    if not np.array_equal(order, np.arange(len(order), dtype=np.int64)):
        timestamps = timestamps[order]
        node_ids = node_ids[order]
    if len(timestamps) % num_nodes:
        raise ValueError("data rows are not divisible by the configured node count")
    time_count = len(timestamps) // num_nodes
    timestamps_2d = timestamps.reshape(time_count, num_nodes)
    nodes_2d = node_ids.reshape(time_count, num_nodes)
    if not np.all(timestamps_2d == timestamps_2d[:, :1]):
        raise ValueError("each timestamp must have one complete node row")
    if not np.all(nodes_2d == nodes_2d[:1]):
        raise ValueError("node order changes between timestamps")
    if len(np.unique(nodes_2d[0])) != num_nodes:
        raise ValueError("duplicate node ids exist within a timestamp")
    if np.any(timestamps_2d[1:, 0] <= timestamps_2d[:-1, 0]):
        raise ValueError("timestamps must be strictly increasing after sorting")
    return timestamps_2d[:, 0], nodes_2d[0], order


def _validate_key_alignment(
    left_timestamps: np.ndarray,
    left_nodes: np.ndarray,
    right_timestamps: np.ndarray,
    right_nodes: np.ndarray,
) -> None:
    if not np.array_equal(left_timestamps, right_timestamps):
        raise ValueError("model input and evaluation target timestamps differ")
    if not np.array_equal(left_nodes, right_nodes):
        raise ValueError("model input and evaluation target node order differ")


def load_data(config: ExperimentConfig, *, project_root: Path | None = None) -> tuple[DataArrays, DataInfo]:
    """Load, time-sort and validate the two formal data files.

    The configured feature columns are the only columns exposed as ``x``.
    Target and mask stay in separate arrays for the Trainer and Evaluator.
    """

    root = (project_root or config.source.parent.parent).resolve()
    data_root = resolve_data_root(root, config.data["data_root"])
    model_path = data_root / config.data["model_input_file"]
    target_path = data_root / config.data["eval_target_file"]
    if not model_path.is_file() or not target_path.is_file():
        missing = [str(path) for path in (model_path, target_path) if not path.is_file()]
        raise FileNotFoundError("formal data file(s) missing: " + ", ".join(missing))

    data = config.data
    feature_columns = list(data["feature_columns"])
    model_schema = pq.read_schema(model_path)
    target_schema = pq.read_schema(target_path)
    for column in feature_columns:
        if column not in model_schema.names:
            raise ValueError(f"model input parquet is missing feature column: {column}")
    for column in (data["turbine_id_column"], data["timestamp_column"]):
        if column not in model_schema.names or column not in target_schema.names:
            raise ValueError(f"formal parquet files are missing key column: {column}")
    for column in (data["target_column"], data["mask_column"]):
        if column not in target_schema.names:
            raise ValueError(f"evaluation target parquet is missing column: {column}")

    model_ids = _array_from_column(model_path, data["turbine_id_column"]).astype(np.int64, copy=False)
    model_times = _array_from_column(model_path, data["timestamp_column"])
    target_ids = _array_from_column(target_path, data["turbine_id_column"]).astype(np.int64, copy=False)
    target_times = _array_from_column(target_path, data["timestamp_column"])
    if len(model_ids) != len(target_ids):
        raise ValueError("model input and evaluation target row counts differ")
    num_nodes = int(data["num_nodes"])
    model_timestamps, model_node_ids, model_order = _validate_or_sort_grid(
        model_times, model_ids, num_nodes=num_nodes
    )
    target_timestamps, target_node_ids, target_order = _validate_or_sort_grid(
        target_times, target_ids, num_nodes=num_nodes
    )
    _validate_key_alignment(model_timestamps, model_node_ids, target_timestamps, target_node_ids)

    x_flat = _read_feature_matrix(model_path, feature_columns)
    target_flat = _array_from_column(target_path, data["target_column"]).astype(np.float32, copy=False)
    mask_flat = _array_from_column(target_path, data["mask_column"]).astype(bool, copy=False)
    x_flat = x_flat[model_order]
    target_flat = target_flat[target_order]
    mask_flat = mask_flat[target_order]
    if not np.isfinite(x_flat).all():
        raise ValueError("model inputs contain non-finite values")
    if not np.isfinite(target_flat[mask_flat]).all():
        raise ValueError("valid targets contain non-finite values")
    arrays = DataArrays(
        x=x_flat.reshape(len(model_timestamps), num_nodes, len(feature_columns)),
        target=target_flat.reshape(len(target_timestamps), num_nodes),
        target_mask=mask_flat.reshape(len(target_timestamps), num_nodes),
        timestamps=model_timestamps,
        node_ids=model_node_ids,
    )
    info = DataInfo.from_arrays(arrays, config)
    return arrays, info
