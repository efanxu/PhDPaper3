"""Shared data loading, splitting, normalization and windowing."""

from .dataset import ForecastBatch, ForecastDataset
from .loader import DataArrays, DataInfo, load_data
from .normalization import NormalizationStats, fit_normalization
from .split import SplitBoundaries, chronological_split
from .window import WindowIndex, build_window_index

__all__ = [
    "DataArrays",
    "DataInfo",
    "ForecastBatch",
    "ForecastDataset",
    "NormalizationStats",
    "SplitBoundaries",
    "WindowIndex",
    "build_window_index",
    "chronological_split",
    "fit_normalization",
    "load_data",
]
