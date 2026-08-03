from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cli import repeatability


def test_repeatability_dispatches_complete_model_batches_and_compares_all_horizons(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_run_training_models(**kwargs):
        models = list(kwargs["models"])
        run_id = kwargs["run_id"]
        output_root = Path(kwargs["output_root"])
        calls.append((models, run_id))
        records = []
        for index, model in enumerate(models):
            result_dir = output_root / model / run_id
            result_dir.mkdir(parents=True, exist_ok=True)
            resolved = {"data": {"eval_horizons": [6]}, "resolved": {"run_id": run_id}}
            (result_dir / "resolved_config.yaml").write_text(
                "data:\n  eval_horizons: [6]\nresolved:\n  run_id: " + run_id + "\n",
                encoding="utf-8",
            )
            (result_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "initial_weight_hash": "same",
                        "first_step_loss": 1.0,
                        "best_epoch": 1,
                        "runtime_environment": "tslib" if model == "a" else "tsl",
                        "conda_env": "env_tslib" if model == "a" else "env_tsl",
                        "python_executable": "python-a" if model == "a" else "python-b",
                        "python_version": "3.11.0",
                        "pytorch_version": "2.5.1",
                        "cuda_version": "12.4",
                    }
                ),
                encoding="utf-8",
            )
            (result_dir / "train_history.csv").write_text(
                "epoch,train_loss,monitor,learning_rate,checkpoint_selected,train_updates\n"
                "1,1.0,1.0,0.001,True,1\n",
                encoding="utf-8",
            )
            (result_dir / "metrics_validation.json").write_text('{"by_horizon": {"6": {"MAE": 1.0}}}\n', encoding="utf-8")
            (result_dir / "metrics_test_h6.json").write_text('{"MAE": 1.0}\n', encoding="utf-8")
            np.savez(result_dir / "predictions.npz", values=np.ones((1, 1, 1)))
            records.append(
                {
                    "model": model,
                    "result_dir": str(result_dir),
                    "pid": 5000 + index + (0 if run_id.endswith("repeat_a") else 10),
                }
            )
        return {"passed": True, "run_id": run_id, "run_root": str(output_root / "_runs" / run_id), "models": records}

    monkeypatch.setattr(repeatability, "run_training_models", fake_run_training_models)
    result = repeatability.compare_repeated_runs(
        models=["a", "b"],
        config_path=tmp_path / "experiment.yaml",
        model_config_path=None,
        run_id="batch",
        output_root=tmp_path / "results",
    )
    assert calls == [(["a", "b"], "batch__repeat_a"), (["b", "a"], "batch__repeat_b")]
    assert result["passed"] is True
    assert all(item["checks"]["test_horizon_set"] for item in result["models"])
    assert all(item["checks"]["different_worker_pid"] for item in result["models"])


def test_repeatability_report_contains_structured_horizon_comparisons(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run_training_models(**kwargs):
        run_id = kwargs["run_id"]
        output_root = Path(kwargs["output_root"])
        result_dir = output_root / "a" / run_id
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "resolved_config.yaml").write_text(
            "data:\n  eval_horizons: [3, 6, 10]\nresolved:\n  run_id: "
            + run_id
            + "\n",
            encoding="utf-8",
        )
        (result_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "initial_weight_hash": "same",
                    "first_step_loss": 1.0,
                    "best_epoch": 1,
                    "runtime_environment": "tslib",
                    "conda_env": "env_tslib",
                    "python_executable": "python",
                    "python_version": "3.11.0",
                    "pytorch_version": "2.5.1",
                    "cuda_version": "12.4",
                }
            ),
            encoding="utf-8",
        )
        (result_dir / "train_history.csv").write_text(
            "epoch,train_loss,monitor,learning_rate,checkpoint_selected,train_updates\n"
            "1,1.0,1.0,0.001,True,1\n",
            encoding="utf-8",
        )
        (result_dir / "metrics_validation.json").write_text(
            '{"by_horizon": {"3": {"MAE": 1.0}}}\n', encoding="utf-8"
        )
        for horizon in (3, 6, 10):
            (result_dir / f"metrics_test_h{horizon}.json").write_text(
                '{"MAE": 1.0, "RMSE": 2.0, "R2": 0.5, "SMAPE": 0.2, '
                '"MAPE": 0.3, "SDWPF Official Score": 0.4}\n',
                encoding="utf-8",
            )
        np.savez(result_dir / "predictions.npz", values=np.ones((1, 1, 1)))
        return {
            "passed": True,
            "run_id": run_id,
            "run_root": str(output_root / "_runs" / run_id),
            "models": [{"model": "a", "result_dir": str(result_dir), "pid": 1 if run_id.endswith("a") else 2}],
        }

    monkeypatch.setattr(repeatability, "run_training_models", fake_run_training_models)
    result = repeatability.compare_repeated_runs(
        models=["a"],
        config_path=tmp_path / "experiment.yaml",
        model_config_path=None,
        run_id="all-horizons",
        output_root=tmp_path / "results",
    )
    report = result["models"][0]
    assert set(report["horizons"]) == {"3", "6", "10"}
    assert all(item["match"] for item in report["horizons"].values())


def test_multi_model_repeatability_rejects_single_model_config(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="--model-config"):
        repeatability.compare_repeated_runs(
            models=["a", "b"],
            config_path=tmp_path / "experiment.yaml",
            model_config_path=tmp_path / "model.yaml",
        )
