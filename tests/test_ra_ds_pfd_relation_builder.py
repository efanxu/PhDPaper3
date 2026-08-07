from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.ra_ds_pfd_crossformer.relation_builder import (
    RelationBuildError,
    STATIC_EDGE_FEATURE_NAMES,
    build_trueunion_graph,
    compare_against_old_graph,
    convert_old_graph,
    pairwise_overlap_cosine,
    pairwise_overlap_cosine_torch,
    raw_delta_wspd,
    write_relation_artifact,
)
from models.ra_ds_pfd_crossformer.relation_resource import load_relation_resource


def _raw_frame(periods: int = 9) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="10min")
    rows: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(timestamps):
        for node_index, turbine_id in enumerate((10, 20, 30)):
            rows.append(
                {
                    "Tmstamp": timestamp,
                    "TurbID": turbine_id,
                    "Wspd": float(time_index + node_index * 0.25),
                    "Patv_raw": float(100 + time_index + node_index),
                    "valid_target_mask": True,
                }
            )
    return pd.DataFrame(rows)


def _location_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TurbID": [10, 20, 30],
            "x": [0.0, 1.0, 4.0],
            "y": [0.0, 0.0, 0.0],
            "Ele": [0.0, 2.0, 3.0],
        }
    )


def _build(raw: pd.DataFrame) -> object:
    return build_trueunion_graph(
        raw,
        _location_frame(),
        [10, 20, 30],
        pd.date_range("2026-01-01", periods=6, freq="10min"),
        semantic_top_k=1,
        distance_top_k=1,
        semantic_min_overlap_count=1,
    )


def test_delta_uses_only_adjacent_ten_minute_pairs() -> None:
    timestamps = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10", "2026-01-01 00:30"])
    delta = raw_delta_wspd(np.asarray([[1.0], [2.0], [4.0]]), timestamps)
    assert delta[0, 0] != delta[0, 0]
    assert delta[1, 0] == 1.0
    assert np.isnan(delta[2, 0])


def test_trueunion_is_train_only_target_independent_and_stable() -> None:
    raw = _raw_frame()
    changed = raw.copy()
    changed.loc[changed["Tmstamp"] >= pd.Timestamp("2026-01-01 01:00"), "Wspd"] = 9999.0
    changed["Patv_raw"] = -777.0
    changed["valid_target_mask"] = False
    first = _build(raw)
    second = _build(changed)
    np.testing.assert_array_equal(first.node_ids, second.node_ids)
    np.testing.assert_array_equal(first.edge_index, second.edge_index)
    np.testing.assert_array_equal(first.edge_static_features, second.edge_static_features)
    assert list(zip(first.edge_index[1], first.edge_index[0])) == sorted(
        zip(first.edge_index[1], first.edge_index[0])
    )
    assert not np.any(first.edge_index[0] == first.edge_index[1])
    assert first.semantic_edge_count == 3
    assert first.distance_edge_count == 3
    assert first.both_edge_count == 2
    assert np.isfinite(first.edge_static_features).all()
    assert set(np.unique(first.edge_static_features[:, 9:]).tolist()).issubset({0.0, 1.0})
    np.testing.assert_allclose(first.edge_static_features[:, :9].mean(axis=0), 0.0, atol=1e-6)


def test_cpu_torch_pairwise_cosine_matches_float64_reference() -> None:
    delta = np.asarray(
        [[1.0, np.nan, 0.0], [2.0, 1.0, np.nan], [3.0, 2.0, 4.0], [np.nan, 3.0, 5.0]],
        dtype=np.float64,
    )
    expected = pairwise_overlap_cosine(delta)
    actual = pairwise_overlap_cosine_torch(delta, device="cpu")
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)


def test_write_artifact_has_only_loader_fields_and_loads(tmp_path: Path) -> None:
    graph = _build(_raw_frame())
    output = write_relation_artifact(graph, tmp_path / "relation.npz")
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "node_ids",
            "edge_index",
            "edge_static_features",
            "edge_feature_names",
        }
    loaded = load_relation_resource(
        {"file": "relation.npz"}, project_root=tmp_path, node_ids=(10, 20, 30)
    )
    assert loaded.edge_count == graph.edge_count
    assert loaded.edge_feature_names == STATIC_EDGE_FEATURE_NAMES


def _old_graph_fixture(directory: Path, *, include_node_ids: bool = True) -> tuple[np.ndarray, np.ndarray]:
    directory.mkdir()
    edge_index = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    features = np.arange(26, dtype=np.float32).reshape(2, 13)
    raw = np.full((2, 13), -123.0, dtype=np.float32)
    np.save(directory / "edge_index.npy", edge_index)
    np.savez(
        directory / "edge_features.npz",
        features=features,
        raw=raw,
        names=np.asarray(STATIC_EDGE_FEATURE_NAMES),
    )
    (directory / "graph_config.json").write_text("{}", encoding="utf-8")
    manifest: dict[str, object] = {"node_count": 2}
    if include_node_ids:
        manifest["node_ids"] = [10, 20]
    (directory / "graph_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return edge_index, features


def test_old_conversion_uses_features_not_raw_and_loader_reads_result(tmp_path: Path) -> None:
    old = tmp_path / "old"
    _edge_index, features = _old_graph_fixture(old)
    output = tmp_path / "converted.npz"
    graph = convert_old_graph(old, output, expected_node_ids=(10, 20))
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["edge_static_features"], features)
    loaded = load_relation_resource(
        {"file": "converted.npz"}, project_root=tmp_path, node_ids=(10, 20)
    )
    np.testing.assert_array_equal(loaded.edge_static_features.numpy(), features)
    assert graph.edge_feature_names == STATIC_EDGE_FEATURE_NAMES


def test_old_conversion_fails_without_provable_node_order(tmp_path: Path) -> None:
    old = tmp_path / "old"
    _old_graph_fixture(old, include_node_ids=False)
    with pytest.raises(RelationBuildError, match="node order cannot be proved"):
        convert_old_graph(old, tmp_path / "converted.npz", expected_node_ids=(10, 20))


def test_rebuilt_graph_compares_against_old_fixture(tmp_path: Path) -> None:
    old = tmp_path / "old"
    edge_index, features = _old_graph_fixture(old)
    from models.ra_ds_pfd_crossformer.relation_builder import TrueUnionGraph

    graph = TrueUnionGraph(
        node_ids=np.asarray([10, 20], dtype=np.int64),
        edge_index=edge_index,
        edge_static_features=features,
        semantic_edge_count=int(np.rint(features[:, 9]).sum()),
        distance_edge_count=int(np.rint(features[:, 10]).sum()),
        both_edge_count=int(np.rint(features[:, 11]).sum()),
    )
    report = compare_against_old_graph(graph, old, expected_node_ids=(10, 20))
    assert report["passed"] is True
