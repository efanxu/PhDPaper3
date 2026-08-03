from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from engine.losses import masked_score_aligned_hybrid
from engine.reproducibility import set_seed, state_dict_hash


def _one_step_run() -> tuple[str, float, np.ndarray]:
    os.environ["PYTHONHASHSEED"] = "2026"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(2026, reproducibility_mode="controlled_nonstrict")
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False, fused=False)
    x = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    target = torch.ones(6, 1)
    loader = DataLoader(TensorDataset(x, target), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(3027), num_workers=0)
    first_loss = None
    for batch_x, batch_target in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_x)
        loss = masked_score_aligned_hybrid(
            prediction.unsqueeze(1),
            batch_target.unsqueeze(1),
            torch.ones(2, 1, 1, dtype=torch.bool),
        )
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(x).numpy()
    return state_dict_hash(model.state_dict()), first_loss, prediction


def test_same_seed_repeats_initialization_batch_order_loss_and_predictions() -> None:
    first = _one_step_run()
    second = _one_step_run()
    assert first[0] == second[0]
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])
