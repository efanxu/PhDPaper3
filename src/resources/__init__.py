"""Optional, lightweight resources for models that explicitly need them."""
"""Shared immutable resources consumed by selected model adapters."""

from .graph import (
    GraphResource,
    build_graph_resource,
    build_physical_knn_adjacency,
    dense_adjacency_to_edges,
    validate_adjacency,
    validate_edge_tensors,
)

__all__ = [
    "GraphResource",
    "build_graph_resource",
    "build_physical_knn_adjacency",
    "dense_adjacency_to_edges",
    "validate_adjacency",
    "validate_edge_tensors",
]
