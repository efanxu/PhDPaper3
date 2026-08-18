from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.base import DataInfoView
from models.ra_ds_pfd_crossformer.p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    P3CandidateBank,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = (
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Wdir_w",
    "Tp",
    "Patv_clean_for_input",
)


def _info(columns: tuple[str, ...] = FEATURE_COLUMNS) -> DataInfoView:
    return DataInfoView(
        num_nodes=2,
        num_features=len(columns),
        lookback=4,
        max_pred_len=2,
        feature_columns=columns,
        input_power_column="Patv_clean_for_input",
        input_power_index=columns.index("Patv_clean_for_input")
        if "Patv_clean_for_input" in columns
        else -1,
        node_ids=(1, 2),
        project_root=ROOT,
    )


def test_candidate_bank_has_stable_26_candidate_order_and_history_semantics() -> None:
    bank = P3CandidateBank(_info())
    assert bank.candidate_count == 26
    assert bank.candidate_names[:4] == (
        "Wspd.level",
        "Wspd.diff1",
        "Etmp.level",
        "Etmp.diff1",
    )
    assert bank.candidate_names[-2:] == (
        "Patv_clean_for_input.level",
        "Patv_clean_for_input.diff1",
    )

    x = torch.arange(2 * 4 * 2 * len(FEATURE_COLUMNS), dtype=torch.float32).reshape(
        2, 4, 2, len(FEATURE_COLUMNS)
    )
    result = bank(x)
    assert tuple(result.shape) == (2, 4, 2, 26)
    assert torch.isfinite(result).all()

    wspd = x[..., FEATURE_COLUMNS.index("Wspd")]
    assert torch.equal(result[..., 0], wspd)
    assert torch.equal(result[..., 1][:, 0], torch.zeros_like(wspd[:, 0]))
    assert torch.equal(result[..., 1][:, 1:], wspd[:, 1:] - wspd[:, :-1])

    patv_index = P3_BASE_FEATURES.index("Patv_clean_for_input") * 2
    patv = x[..., FEATURE_COLUMNS.index("Patv_clean_for_input")]
    assert torch.equal(result[..., patv_index], patv)
    assert torch.equal(
        result[..., patv_index + 1][:, 0],
        torch.zeros_like(patv[:, 0]),
    )


def test_candidate_bank_does_not_read_future_history_values() -> None:
    bank = P3CandidateBank(_info())
    x = torch.randn(1, 4, 2, len(FEATURE_COLUMNS))
    changed = x.clone()
    changed[:, -1, :, FEATURE_COLUMNS.index("Wspd")] += 1000.0
    original = bank(x)
    updated = bank(changed)
    assert torch.equal(original[:, :-1, :, :], updated[:, :-1, :, :])
    assert torch.equal(original[:, 0, :, 1], updated[:, 0, :, 1])


def test_candidate_bank_resolves_by_feature_name_and_rejects_bad_metadata() -> None:
    missing = tuple(column for column in FEATURE_COLUMNS if column != "Tp")
    with pytest.raises(ValueError, match="missing.*Tp"):
        P3CandidateBank(_info(missing))

    duplicate = (*FEATURE_COLUMNS, "Wspd")
    with pytest.raises(ValueError, match="duplicate"):
        P3CandidateBank(_info(duplicate))

    with pytest.raises(ValueError, match="circular direction"):
        P3CandidateBank(_info(), candidate_features=("Wdir",))

    with pytest.raises(ValueError, match="illegal feature"):
        P3CandidateBank(_info(), candidate_features=("not_a_feature",))

    with pytest.raises(ValueError, match="duplicate feature"):
        P3CandidateBank(_info(), candidate_features=("Wspd", "Wspd"))

    with pytest.raises(ValueError, match="exactly"):
        P3CandidateBank(_info(), candidate_transforms=("level",))


def test_candidate_bank_uses_canonical_order_even_if_subset_is_presented_differently() -> None:
    bank = P3CandidateBank(
        _info(),
        candidate_features=("Tp", "Wspd", "Etmp"),
        candidate_transforms=P3_CANDIDATE_TRANSFORMS,
    )
    assert bank.candidate_names == (
        "Wspd.level",
        "Wspd.diff1",
        "Etmp.level",
        "Etmp.diff1",
        "Tp.level",
        "Tp.diff1",
    )
