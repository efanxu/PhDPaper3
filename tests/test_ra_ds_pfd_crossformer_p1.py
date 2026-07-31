from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange, repeat

from engine.checkpoint import load_checkpoint, save_checkpoint
from integrations.time_series_library import load_time_series_library_model_class
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import load_model_config, load_model_config_document


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "d_model": 16,
    "n_heads": 4,
    "d_ff": 32,
    "e_layers": 2,
    "dropout": 0.0,
    "factor": 2,
    "seg_len": 12,
    "win_size": 2,
    "spatial_disabled": True,
}
FORMAL_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "d_ff": 128,
    "e_layers": 2,
    "dropout": 0.0,
    "factor": 10,
    "seg_len": 12,
    "win_size": 2,
    "spatial_disabled": True,
}


def _info(
    nodes: int = 3,
    *,
    lookback: int = 24,
    horizon: int = 3,
    root: Path = ROOT,
    input_power_index: int = 1,
) -> DataInfoView:
    return DataInfoView(
        num_nodes=nodes,
        num_features=4,
        lookback=lookback,
        max_pred_len=horizon,
        feature_columns=("Wspd", "Patv_clean_for_input", "Wdir", "Etmp"),
        input_power_column="Patv_clean_for_input",
        input_power_index=input_power_index,
        node_ids=tuple(range(1, nodes + 1)),
        project_root=root,
    )


def _formal_info() -> DataInfoView:
    feature_columns = tuple(f"feature_{index}" for index in range(15)) + (
        "Patv_clean_for_input",
    )
    return DataInfoView(
        num_nodes=2,
        num_features=16,
        lookback=144,
        max_pred_len=10,
        feature_columns=feature_columns,
        input_power_column="Patv_clean_for_input",
        input_power_index=15,
        node_ids=(1, 2),
        project_root=ROOT,
    )


def _build(info: DataInfoView | None = None):
    return build_model("ra_ds_pfd_crossformer", dict(CONFIG), info or _info())


def _upstream(info: DataInfoView, config: dict[str, object] = CONFIG):
    upstream_class = load_time_series_library_model_class(
        "Crossformer",
        source_root=ROOT / "Time-Series-Library",
    )
    return upstream_class(
        SimpleNamespace(
            enc_in=info.num_features,
            seq_len=info.lookback,
            pred_len=info.max_pred_len,
            task_name="long_term_forecast",
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            e_layers=config["e_layers"],
            dropout=config["dropout"],
            factor=config["factor"],
        )
    )


def _upstream_trace(model, node_history: torch.Tensor):
    embedded, n_vars = model.enc_value_embedding(node_history.permute(0, 2, 1))
    encoded = rearrange(
        embedded,
        "(b d) seg_num d_model -> b d seg_num d_model",
        d=n_vars,
    )
    pre_norm = model.pre_norm(encoded + model.enc_pos_embedding)
    scale0, _ = model.encoder.encode_blocks[0](pre_norm)
    scale1, _ = model.encoder.encode_blocks[1](scale0)
    scale1_merged = model.encoder.encode_blocks[1].merge_layer(scale0)
    encoder_tokens = [pre_norm, scale0, scale1]
    decoder_input = repeat(
        model.dec_pos_embedding,
        "b ts_d l d -> (repeat b) ts_d l d",
        repeat=pre_norm.shape[0],
    )
    decoder_output = model.decoder(decoder_input, encoder_tokens)
    return embedded, pre_norm, encoder_tokens, scale1_merged, decoder_output


def test_yaml_shape_runtime_and_structure_fields() -> None:
    path = ROOT / "configs" / "models" / "ra_ds_pfd_crossformer.yaml"
    document = load_model_config_document(path)
    assert document["runtime"] == {"environment": "tslib"}
    assert set(document) == {"runtime", "model"}
    assert load_model_config(path) == document["model"]
    assert set(document["model"]) == set(CONFIG)


def test_config_rejects_unknown_missing_invalid_and_unsupported_values() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_model("ra_ds_pfd_crossformer", {**CONFIG, "unknown": 1}, _info())
    missing = dict(CONFIG)
    missing.pop("win_size")
    with pytest.raises(ValueError, match="missing"):
        build_model("ra_ds_pfd_crossformer", missing, _info())
    for field, value in (("d_model", 0), ("n_heads", 0), ("d_ff", 0), ("factor", 0)):
        invalid = {**CONFIG, field: value}
        with pytest.raises(ValueError, match="positive"):
            build_model("ra_ds_pfd_crossformer", invalid, _info())
    with pytest.raises(ValueError, match="divisible"):
        build_model("ra_ds_pfd_crossformer", {**CONFIG, "d_model": 15}, _info())
    with pytest.raises(ValueError, match="dropout"):
        build_model("ra_ds_pfd_crossformer", {**CONFIG, "dropout": 1.0}, _info())
    with pytest.raises(ValueError, match="seg_len=12"):
        build_model("ra_ds_pfd_crossformer", {**CONFIG, "seg_len": 6}, _info())
    with pytest.raises(ValueError, match="win_size=2"):
        build_model("ra_ds_pfd_crossformer", {**CONFIG, "win_size": 3}, _info())
    with pytest.raises(ValueError, match="e_layers=2"):
        build_model("ra_ds_pfd_crossformer", {**CONFIG, "e_layers": 1}, _info())


def test_spatial_enabled_false_fails_closed_and_input_power_metadata_is_strict() -> None:
    with pytest.raises(ValueError, match="spatial_disabled=false"):
        build_model("ra_ds_pfd_crossformer", {**CONFIG, "spatial_disabled": False}, _info())
    mismatched = _info(input_power_index=2)
    with pytest.raises(ValueError, match="input_power_index"):
        build_model("ra_ds_pfd_crossformer", dict(CONFIG), mismatched)


def test_trace_exposes_both_identity_insertions_and_canonical_shapes() -> None:
    model = _build().eval()
    x = torch.randn(2, 24, 3, 4)
    calls: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}

    def record(scale_id: int):
        def hook(_module, args, output):
            calls[scale_id].append((args[0], output))

        return hook

    handles = [
        model.backbone.scale0_spatial_insertion.register_forward_hook(record(0)),
        model.backbone.scale1_spatial_insertion.register_forward_hook(record(1)),
    ]
    try:
        with torch.no_grad():
            trace = model.forward_canonical_trace(ModelInput(x=x))
    finally:
        for handle in handles:
            handle.remove()

    assert len(calls[0]) == 1
    assert len(calls[1]) == 1
    for scale_id in (0, 1):
        input_value, output_value = calls[scale_id][0]
        assert input_value is output_value
        torch.testing.assert_close(input_value, output_value, atol=0.0, rtol=0.0)
        assert input_value.dtype == output_value.dtype
    assert tuple(trace.segment_embedding.shape) == (2 * 3 * 4, 2, 16)
    assert tuple(trace.pre_norm.shape) == (2 * 3, 4, 2, 16)
    assert tuple(trace.scale0_cross_time.shape) == (2 * 3 * 4, 2, 16)
    assert tuple(trace.scale0_spatial.shape) == (2 * 3 * 4, 2, 16)
    assert tuple(trace.scale0_cross_dimension.shape) == (2 * 3, 4, 2, 16)
    assert tuple(trace.scale1_merged.shape) == (2 * 3, 4, 1, 16)
    assert tuple(trace.scale1_cross_time.shape) == (2 * 3 * 4, 1, 16)
    assert tuple(trace.scale1_spatial.shape) == (2 * 3 * 4, 1, 16)
    assert tuple(trace.scale1_cross_dimension.shape) == (2 * 3, 4, 1, 16)
    assert len(trace.decoder_tokens) == 3
    assert [tuple(value.shape) for value in trace.decoder_tokens] == [
        (6, 4, 2, 16),
        (6, 4, 2, 16),
        (6, 4, 1, 16),
    ]
    assert tuple(trace.decoder_output.shape) == (6, 12, 4)
    assert tuple(trace.output.shape) == (6, 3, 4)


def test_strict_state_transfer_and_full_intermediate_numeric_equivalence() -> None:
    info = _info()
    local = _build(info).eval()
    upstream = _upstream(info).eval()
    assert list(local.backbone.state_dict()) == list(upstream.state_dict())
    assert {
        key: tuple(value.shape) for key, value in local.backbone.state_dict().items()
    } == {key: tuple(value.shape) for key, value in upstream.state_dict().items()}
    local.load_upstream_state_dict(upstream.state_dict())

    x = torch.randn(2, 24, 3, 4, dtype=torch.float32)
    node_history = x.permute(0, 2, 1, 3).reshape(6, 24, 4)
    with torch.no_grad():
        local_trace = local.backbone.forward_backbone(node_history, return_trace=True)
        (
            upstream_embedded,
            upstream_pre_norm,
            upstream_tokens,
            upstream_scale1_merged,
            upstream_decoder,
        ) = _upstream_trace(
            upstream,
            node_history,
        )
        upstream_output = upstream(node_history, None, None, None)
        local_output = local_trace.output
        local_adapter_output = local(ModelInput(x=x))
        upstream_adapter_output = upstream_output[..., 1].reshape(2, 3, 3)

    torch.testing.assert_close(local_trace.segment_embedding, upstream_embedded, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_trace.pre_norm, upstream_pre_norm, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_trace.scale0_cross_dimension, upstream_tokens[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_trace.scale1_merged, upstream_scale1_merged, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_trace.scale1_cross_dimension, upstream_tokens[2], atol=1e-6, rtol=1e-6)
    for local_token, upstream_token in zip(local_trace.decoder_tokens, upstream_tokens):
        torch.testing.assert_close(local_token, upstream_token, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_trace.decoder_output, upstream_decoder, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_output, upstream_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_adapter_output, upstream_adapter_output, atol=1e-6, rtol=1e-6)


def test_current_formal_two_scale_configuration_is_numerically_equivalent() -> None:
    info = _formal_info()
    local = build_model("ra_ds_pfd_crossformer", dict(FORMAL_CONFIG), info).eval()
    upstream = _upstream(info, FORMAL_CONFIG).eval()
    local.load_upstream_state_dict(upstream.state_dict())
    x = torch.randn(1, 144, 2, 16, dtype=torch.float32)
    node_history = x.permute(0, 2, 1, 3).reshape(2, 144, 16)
    with torch.no_grad():
        local_trace = local.backbone.forward_backbone(node_history, return_trace=True)
        upstream_output = upstream(node_history, None, None, None)
        local_adapter_output = local(ModelInput(x=x))
        upstream_adapter_output = upstream_output[..., 15].reshape(1, 2, 10)
    assert local.backbone.in_seg_num == 12
    assert local.backbone.out_seg_num == 6
    assert [tuple(value.shape) for value in local_trace.decoder_tokens] == [
        (2, 16, 12, 64),
        (2, 16, 12, 64),
        (2, 16, 6, 64),
    ]
    assert tuple(local_trace.output.shape) == (2, 10, 16)
    torch.testing.assert_close(local_trace.output, upstream_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_adapter_output, upstream_adapter_output, atol=1e-6, rtol=1e-6)


def test_odd_segment_merging_copies_upstream_padding_and_output() -> None:
    info = _info(lookback=60)
    local = _build(info).eval()
    upstream = _upstream(info).eval()
    local.load_upstream_state_dict(upstream.state_dict())
    x = torch.randn(1, 60, 3, 4)
    node_history = x.permute(0, 2, 1, 3).reshape(3, 60, 4)

    with torch.no_grad():
        local_trace = local.backbone.forward_backbone(node_history, return_trace=True)
        upstream_output = upstream(node_history, None, None, None)

    assert tuple(local_trace.pre_norm.shape) == (3, 4, 5, 16)
    assert tuple(local_trace.scale1_merged.shape) == (3, 4, 3, 16)
    assert tuple(local_trace.scale1_cross_dimension.shape) == (3, 4, 3, 16)
    assert [tuple(value.shape) for value in local_trace.decoder_tokens] == [
        (3, 4, 5, 16),
        (3, 4, 5, 16),
        (3, 4, 3, 16),
    ]
    torch.testing.assert_close(local_trace.output, upstream_output, atol=1e-6, rtol=1e-6)


def test_node_shared_boundaries_and_repeatability() -> None:
    three_info = _info(3)
    five_info = _info(5)
    three = _build(three_info).eval()
    five = _build(five_info).eval()
    upstream = _upstream(three_info)
    three.load_upstream_state_dict(upstream.state_dict())
    five.load_upstream_state_dict(upstream.state_dict())
    assert sum(parameter.numel() for parameter in three.parameters()) == sum(
        parameter.numel() for parameter in five.parameters()
    )
    x = torch.randn(1, 24, 3, 4)
    x[:, :, 1] = x[:, :, 0]
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        output = three(ModelInput(x=x))
        permuted = three(ModelInput(x=x[:, :, permutation]))
        repeated = three(ModelInput(x=x))
    torch.testing.assert_close(permuted, output[:, permutation], atol=0.0, rtol=0.0)
    torch.testing.assert_close(output[:, 0], output[:, 1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(output, repeated, atol=0.0, rtol=0.0)


def test_public_input_and_finite_boundaries() -> None:
    model = _build().eval()
    x = torch.randn(1, 24, 3, 4)
    with pytest.raises(ValueError, match="history x only"):
        model(ModelInput(x=x, time_features=torch.zeros(1, 24, 1)))
    with pytest.raises(ValueError, match="input shape"):
        model(ModelInput(x=torch.randn(1, 23, 3, 4)))
    nonfinite = x.clone()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="input contains"):
        model(ModelInput(x=nonfinite))

    def nan_forward(value: torch.Tensor) -> torch.Tensor:
        return torch.full((value.shape[0], 3, 4), float("nan"), device=value.device)

    model.backbone.forward = nan_forward  # type: ignore[method-assign]
    with pytest.raises(FloatingPointError, match="output contains"):
        model(ModelInput(x=x))


def test_forward_backward_and_all_trainable_gradients_are_finite() -> None:
    model = _build().train()
    x = torch.randn(2, 24, 3, 4, requires_grad=True)
    output = model(ModelInput(x=x))
    assert tuple(output.shape) == (2, 3, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable if parameter.grad is not None)


def test_shared_checkpoint_reload_preserves_forward_and_canonical_config(tmp_path: Path) -> None:
    model = _build().eval()
    x = torch.randn(1, 24, 3, 4)
    with torch.no_grad():
        before = model(ModelInput(x=x))
    path = tmp_path / "ra_ds_pfd_p1.pt"
    save_checkpoint(path, model, manifest={"model_config": model.canonical_model_config()})
    reloaded = _build().eval()
    manifest = load_checkpoint(path, reloaded)
    with torch.no_grad():
        after = reloaded(ModelInput(x=x))
    torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)
    saved_config = manifest["model_config"]
    assert saved_config["seg_len"] == 12
    assert saved_config["win_size"] == 2
    assert saved_config["spatial_disabled"] is True
