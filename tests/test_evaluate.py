from __future__ import annotations

from pathlib import Path

import pytest

from cli import evaluate


def test_run_id_finds_best_checkpoint_and_writes_evaluate_id(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "configs" / "experiment.yaml"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "results" / "node_shared_lstm" / "source" / "best.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    calls = []

    def fake_run_evaluate_model(**kwargs):
        calls.append(kwargs)
        return {"output_dir": str(kwargs["output_root"])}

    monkeypatch.setattr(evaluate, "run_evaluate_model", fake_run_evaluate_model)
    result = evaluate.evaluate_checkpoint(
        model_name="node_shared_lstm",
        config_path=config,
        model_config_path=None,
        checkpoint=None,
        run_id="source",
        device="cpu",
        output_root=tmp_path / "results",
    )
    assert result["output_dir"] == str(tmp_path / "results")
    assert calls[0]["checkpoint"] == source.resolve()
    assert calls[0]["run_id"] == "source__evaluate"
    assert calls[0]["evaluate_only"] is True


def test_missing_run_id_checkpoint_fails_before_run_model(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "experiment.yaml"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no checkpoint found"):
        evaluate.evaluate_checkpoint(
            model_name="node_shared_lstm",
            config_path=config,
            model_config_path=None,
            checkpoint=None,
            run_id="missing",
            device="cpu",
            output_root=tmp_path / "results",
        )


def test_explicit_checkpoint_remains_supported(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    captured = {}

    def fake_run_evaluate_model(**kwargs):
        captured.update(kwargs)
        return {"output_dir": "evaluation"}

    monkeypatch.setattr(evaluate, "run_evaluate_model", fake_run_evaluate_model)
    evaluate.evaluate_checkpoint(
        model_name="node_shared_lstm",
        config_path=tmp_path / "configs" / "experiment.yaml",
        model_config_path=None,
        checkpoint=checkpoint,
        run_id=None,
        device="cpu",
        output_root=tmp_path / "results",
    )
    assert captured["checkpoint"] == checkpoint.resolve()
    assert captured["run_id"] == "evaluate"
