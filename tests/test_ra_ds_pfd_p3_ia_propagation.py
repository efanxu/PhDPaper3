from __future__ import annotations

from pathlib import Path

import torch

from models.ra_ds_pfd_crossformer.p3_ia_propagation import IAFixedPropagation
from models.ra_ds_pfd_crossformer.p3_propagation import P3GlobalTopKPropagation


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


def _module(selected: tuple[str, ...] = ("Wspd.level", "Wspd.diff1")) -> IAFixedPropagation:
    return IAFixedPropagation(
        feature_columns=FEATURE_COLUMNS,
        selected_candidates=selected,
        lookback=24,
        seg_len=12,
        win_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        factor=10,
        spatial_dropout=0.0,
        source_root=ROOT / "Time-Series-Library",
    )


def test_ia_fixed_candidate_history_is_causal_and_selected_only() -> None:
    module = _module()
    x = torch.randn(1, 24, 2, len(FEATURE_COLUMNS))
    changed = x.clone()
    changed[:, -1, :, FEATURE_COLUMNS.index("Wspd")] += 1000.0

    original = module.candidate_history(x)
    updated = module.candidate_history(changed)
    wspd = x[..., FEATURE_COLUMNS.index("Wspd")]
    assert tuple(original.shape) == (1, 24, 2, 2)
    assert torch.equal(original[..., 0], wspd)
    assert torch.equal(original[:, 0, :, 1], torch.zeros_like(wspd[:, 0]))
    assert torch.equal(original[:, 1:, :, 1], wspd[:, 1:] - wspd[:, :-1])
    assert torch.equal(original[:, :-1], updated[:, :-1])


def test_ia_fixed_temporal_encoder_physically_sees_k_selected_streams() -> None:
    module = _module().eval()
    x = torch.randn(2, 24, 3, len(FEATURE_COLUMNS))
    seen_batch_sizes: list[int] = []
    handles = [
        encoder.register_forward_pre_hook(
            lambda _module, args: seen_batch_sizes.append(int(args[0].shape[0]))
        )
        for encoder in (module.scale0_cross_time, module.scale1_cross_time)
    ]
    try:
        with torch.no_grad():
            scale0, scale1 = module(x)
    finally:
        for handle in handles:
            handle.remove()

    assert module.candidate_bank.candidate_count == 26
    assert module.effective_candidate_count == 2
    assert module.selected_candidate_names == ("Wspd.level", "Wspd.diff1")
    assert module.cross_time_candidate_counts == (2, 2)
    assert seen_batch_sizes == [2 * 2, 2 * 2]
    assert tuple(scale0.shape) == (2, 3, 2, 16)
    assert tuple(scale1.shape) == (2, 3, 1, 16)
    assert torch.isfinite(scale0).all() and torch.isfinite(scale1).all()


def test_ia_fixed_backward_reaches_selected_projections_without_unselected_path() -> None:
    module = _module().train()
    x = torch.randn(2, 24, 2, len(FEATURE_COLUMNS), requires_grad=True)
    scale0, scale1 = module(x)
    (scale0.square().mean() + scale1.square().mean()).backward()

    assert all(parameter.grad is not None for parameter in module.candidate_projections.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters() if parameter.grad is not None)
    patv_index = FEATURE_COLUMNS.index("Patv_clean_for_input")
    assert torch.equal(x.grad[..., patv_index], torch.zeros_like(x.grad[..., patv_index]))
    assert all("Patv_clean_for_input" not in name for name, _ in module.named_parameters())


def test_old_global_p3_k2_cross_time_still_sees_the_full_26_candidate_bank() -> None:
    module = P3GlobalTopKPropagation(
        feature_columns=FEATURE_COLUMNS,
        candidate_features=(
            "Wspd",
            "Etmp",
            "Itmp",
            "Pab1",
            "Pab2",
            "Pab3",
            "Prtv",
            "T2m",
            "Sp",
            "RelH",
            "Wspd_w",
            "Tp",
            "Patv_clean_for_input",
        ),
        candidate_transforms=("level", "diff1"),
        top_k=2,
        lookback=24,
        seg_len=12,
        win_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        factor=10,
        spatial_dropout=0.0,
        source_root=ROOT / "Time-Series-Library",
    ).eval()
    x = torch.randn(2, 24, 3, len(FEATURE_COLUMNS))
    seen_batch_sizes: list[int] = []
    handle = module.scale0_cross_time.register_forward_pre_hook(
        lambda _module, args: seen_batch_sizes.append(int(args[0].shape[0]))
    )
    try:
        with torch.no_grad():
            module(x)
    finally:
        handle.remove()
    assert module.candidate_count == 26
    assert seen_batch_sizes == [2 * 26]
