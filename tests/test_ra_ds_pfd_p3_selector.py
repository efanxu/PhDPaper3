from __future__ import annotations

from pathlib import Path

import torch

from models.ra_ds_pfd_crossformer.p3_feature_bank import P3_BASE_FEATURES
from models.ra_ds_pfd_crossformer.p3_propagation import P3GlobalTopKPropagation
from models.ra_ds_pfd_crossformer.p3_selector import GlobalTopKSelector
from models.ra_ds_pfd_crossformer.pfd0 import CanonicalCrossTime, PFD0SegmentMerging


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


def _names() -> tuple[str, ...]:
    return tuple(
        f"{feature}.{transform}"
        for feature in P3_BASE_FEATURES
        for transform in ("level", "diff1")
    )


def _propagation() -> P3GlobalTopKPropagation:
    return P3GlobalTopKPropagation(
        feature_columns=FEATURE_COLUMNS,
        candidate_features=P3_BASE_FEATURES,
        candidate_transforms=("level", "diff1"),
        top_k=2,
        lookback=24,
        seg_len=12,
        win_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        factor=10,
        dropout=0.0,
        source_root=ROOT / "Time-Series-Library",
    )


def test_global_selector_has_one_finite_global_score_vector_and_stable_ties() -> None:
    selector = GlobalTopKSelector(_names(), top_k=2)
    scores = selector()
    assert tuple(selector.logits.shape) == (26,)
    assert tuple(scores.shape) == (26,)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores.sum(), torch.ones(()))

    report = selector.selection_report()
    assert len(report) == 26
    assert all(set(item) == {"candidate_name", "score", "rank", "selected"} for item in report)
    assert all(torch.isfinite(torch.tensor(item["score"])) for item in report)
    assert sorted(item["rank"] for item in report) == list(range(1, 27))
    assert sum(item["selected"] for item in report) == 2
    assert [item["candidate_name"] for item in report[:2]] == [
        "Wspd.level",
        "Wspd.diff1",
    ]


def test_p3_uses_candidate_projections_and_shared_temporal_modules() -> None:
    torch.manual_seed(2026)
    propagation = _propagation()
    assert len(propagation.candidate_projections) == 26
    assert all(
        tuple(projection.weight.shape) == (16, 12)
        and projection.bias is None
        for projection in propagation.candidate_projections
    )
    assert propagation.candidate_identity.num_embeddings == 26
    assert isinstance(propagation.scale0_cross_time, CanonicalCrossTime)
    assert isinstance(propagation.scale1_cross_time, CanonicalCrossTime)
    assert isinstance(propagation.scale1_merging, PFD0SegmentMerging)
    assert sum(
        isinstance(module, CanonicalCrossTime)
        for module in propagation.modules()
    ) == 2

    x = torch.randn(2, 24, 3, 16)
    candidates0, candidates1 = propagation.encode_candidates(x)
    scale0, scale1 = propagation(x)
    assert tuple(candidates0.shape) == (2, 3, 26, 2, 16)
    assert tuple(candidates1.shape) == (2, 3, 26, 1, 16)
    assert tuple(scale0.shape) == (2, 3, 2, 16)
    assert tuple(scale1.shape) == (2, 3, 1, 16)
    scores = propagation.selector()
    assert torch.allclose(
        scale0,
        (candidates0 * scores.view(1, 1, 26, 1, 1)).sum(dim=2),
    )
    assert torch.allclose(
        scale1,
        (candidates1 * scores.view(1, 1, 26, 1, 1)).sum(dim=2),
    )


def test_soft_global_scores_provide_finite_gradient_to_all_p3_paths() -> None:
    torch.manual_seed(2026)
    propagation = _propagation().train()
    x = torch.randn(1, 24, 2, 16)
    scale0, scale1 = propagation(x)
    loss = scale0.square().mean() + scale1.square().mean()
    loss.backward()

    assert propagation.selector.logits.grad is not None
    assert torch.isfinite(propagation.selector.logits.grad).all()
    assert all(
        projection.weight.grad is not None
        and torch.isfinite(projection.weight.grad).all()
        for projection in propagation.candidate_projections
    )
    assert propagation.scale0_cross_time.MLP1[0].weight.grad is not None
    assert torch.isfinite(propagation.scale0_cross_time.MLP1[0].weight.grad).all()
    assert propagation.scale1_cross_time.MLP1[0].weight.grad is not None
    assert torch.isfinite(propagation.scale1_cross_time.MLP1[0].weight.grad).all()
    assert torch.isfinite(scale0).all()
    assert torch.isfinite(scale1).all()
