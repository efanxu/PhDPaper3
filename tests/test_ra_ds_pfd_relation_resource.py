from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from models.base import DataInfoView
from models.ra_ds_pfd_crossformer.relation_resource import (
    RELATION_RESOURCE_SCHEMA_VERSION,
    STATIC_EDGE_FEATURE_NAMES,
    load_relation_resource,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "ra_ds_pfd_relation_small_v1.npz"


def _config(path: Path = FIXTURE, *, project_root: Path = ROOT) -> dict[str, object]:
    return {
        "file": str(path.relative_to(project_root)).replace("\\", "/"),
    }


def _archive_values() -> dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _write_variant(tmp_path: Path, **updates: object) -> Path:
    values: dict[str, object] = _archive_values()
    values.update(updates)
    path = tmp_path / "relation_variant.npz"
    np.savez(path, **values)
    return path


def test_file_only_config_loads_versioned_fixture() -> None:
    resource = load_relation_resource(_config(), project_root=ROOT, node_ids=(1, 2, 3))
    assert resource.schema_version == RELATION_RESOURCE_SCHEMA_VERSION
    assert resource.path == FIXTURE.resolve()


def test_fixture_has_strict_trueunion_resource_schema() -> None:
    resource = load_relation_resource(_config(), project_root=ROOT, node_ids=(1, 2, 3))
    assert resource.node_ids == (1, 2, 3)
    assert resource.edge_index.dtype == torch.int64
    assert tuple(resource.edge_index.shape) == (2, 4)
    assert resource.edge_static_features.dtype == torch.float32
    assert tuple(resource.edge_static_features.shape) == (4, 13)
    assert resource.edge_feature_names == STATIC_EDGE_FEATURE_NAMES
    pairs = list(zip(resource.edge_index[1].tolist(), resource.edge_index[0].tolist()))
    assert pairs == sorted(pairs)
    assert len(set(pairs)) == len(pairs)
    assert not bool((resource.edge_index[0] == resource.edge_index[1]).any())


@pytest.mark.parametrize("field", ["schema_version", "sha256"])
def test_relation_resource_rejects_legacy_config_fields(field: str) -> None:
    legacy = _config()
    legacy[field] = 1 if field == "schema_version" else "legacy-value"
    with pytest.raises(ValueError, match=rf"unknown field: {field}"):
        load_relation_resource(legacy, project_root=ROOT, node_ids=(1, 2, 3))


def test_relation_resource_rejects_node_order_mismatch() -> None:
    with pytest.raises(ValueError, match="node_ids"):
        load_relation_resource(_config(), project_root=ROOT, node_ids=(2, 1, 3))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", np.asarray(99, dtype=np.int64), "schema_version"),
        ("node_ids", np.asarray([1, 3, 2], dtype=np.int64), "node_ids"),
        (
            "edge_index",
            np.asarray([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=np.int64),
            "sorted",
        ),
        (
            "edge_index",
            np.asarray([[1, 0, 2, 1], [0, 1, 1, 1]], dtype=np.int64),
            "self-loop",
        ),
        (
            "edge_index",
            np.asarray([[1, 0, 2, 1], [0, 1, 1, 0]], dtype=np.int64),
            "duplicate",
        ),
        (
            "edge_index",
            np.asarray([[1, 0, 2], [0, 1, 1]], dtype=np.int64),
            "zero indegree",
        ),
        (
            "edge_index",
            np.asarray([[1, 0, 3, 1], [0, 1, 1, 2]], dtype=np.int64),
            "outside",
        ),
        (
            "edge_static_features",
            np.full((4, 13), np.nan, dtype=np.float32),
            "non-finite",
        ),
        (
            "edge_feature_names",
            np.asarray((*STATIC_EDGE_FEATURE_NAMES[:-1], "wrong")),
            "edge_feature_names",
        ),
    ],
)
def test_relation_resource_rejects_invalid_artifact_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = _write_variant(tmp_path, **{field: value})
    config = _config(path, project_root=tmp_path)
    with pytest.raises(ValueError, match=message):
        load_relation_resource(config, project_root=tmp_path, node_ids=(1, 2, 3))


def test_relation_resource_rejects_absolute_paths_and_extra_payload_fields(tmp_path: Path) -> None:
    absolute = _config()
    absolute["file"] = str(FIXTURE.resolve())
    with pytest.raises(ValueError, match="project-relative"):
        load_relation_resource(absolute, project_root=ROOT, node_ids=(1, 2, 3))

    path = _write_variant(tmp_path, target=np.zeros(1, dtype=np.float32))
    config = _config(path, project_root=tmp_path)
    with pytest.raises(ValueError, match="unsupported field"):
        load_relation_resource(config, project_root=tmp_path, node_ids=(1, 2, 3))


def test_relation_resource_is_not_available_without_public_node_order() -> None:
    with pytest.raises(ValueError, match="public data node_ids"):
        load_relation_resource(_config(), project_root=ROOT, node_ids=())


def test_resource_contract_is_independent_of_model_input_labels() -> None:
    info = DataInfoView(
        num_nodes=3,
        num_features=2,
        lookback=12,
        max_pred_len=3,
        feature_columns=("Wspd", "Patv_clean_for_input"),
        input_power_column="Patv_clean_for_input",
        input_power_index=1,
        node_ids=(1, 2, 3),
        project_root=ROOT,
    )
    resource = load_relation_resource(_config(), project_root=info.project_root, node_ids=info.node_ids)
    assert "target" not in resource.edge_feature_names
    assert "mask" not in resource.edge_feature_names
