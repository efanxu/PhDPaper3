"""Adapters for external model sources used by PhDPaper3."""

from .time_series_library import (
    load_time_series_library_model,
    load_time_series_library_model_class,
    resolve_time_series_library_model,
)

__all__ = [
    "load_time_series_library_model",
    "load_time_series_library_model_class",
    "resolve_time_series_library_model",
]
