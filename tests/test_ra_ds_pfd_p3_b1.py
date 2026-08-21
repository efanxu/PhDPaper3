from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch
from torch import nn
import yaml

from engine.checkpoint import save_checkpoint
from models.base import ForecastModel
from models.ra_ds_pfd_crossformer.p3_b1_suite import (
    VARIANT_IDS,
    load_p3_b1_suite,
    resolve_p3_b1_variants,
)
from models.ra_ds_pfd_crossformer.p3_feature_bank import P3_BASE_FEATURES, P3CandidateBank
from models.ra_ds_pfd_crossformer.p3_selector import GlobalTopKSelector
from models.ra_ds_pfd_crossformer.p3_selection import write_p3_selection_best
from runtime.config import load_model_config_document


ROOT = Path(__file__).resolve().parents[1]
B1_SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_p3_b1.yaml"
B1_SCRIPT = ROOT / "scripts" / "run_ra_ds_pfd_p3_b1.py"
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


def _load_b1_runner():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("ra_ds_pfd_p3_b1_runner", B1_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_b1_runner()


def test_b1_resolves_ld_and_l_from_canonical_p3() -> None:
    suite = load_p3_b1_suite(B1_SUITE_PATH)
    resolved = resolve_p3_b1_variants(B1_SUITE_PATH, project_root=ROOT)
    assert suite["base"]["suite_file"] == "configs/experiments/ra_ds_pfd_p3.yaml"
    assert tuple(resolved) == VARIANT_IDS
    assert resolved["B1_LD"]["p3"]["candidate_transforms"] == ["level", "diff1"]
    assert resolved["B1_L"]["p3"]["candidate_transforms"] == ["level"]
    assert resolved["B1_LD"]["p3"]["top_k"] == 2
    assert resolved["B1_L"]["p3"]["top_k"] == 2
    assert len(P3_BASE_FEATURES) * 2 == 26
    assert len(P3_BASE_FEATURES) == 13

    canonical = deepcopy(resolved["B1_LD"])
    assert canonical == resolved["B1_LD"]
    level_only = deepcopy(canonical)
    level_only["p3"]["candidate_transforms"] = ["level"]
    assert resolved["B1_L"] == level_only


def test_b1_execution_preparation_resolves_runtime_and_model_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_paths: list[Path] = []

    def load_document(path: Path):  # type: ignore[no-untyped-def]
        resolved_path = Path(path).resolve()
        loaded_paths.append(resolved_path)
        return load_model_config_document(resolved_path)

    monkeypatch.setattr(RUNNER, "load_model_config_document", load_document)
    runtime = RUNNER._base_runtime_document()
    expected_model_path = (
        ROOT / "configs" / "models" / "ra_ds_pfd_crossformer.yaml"
    ).resolve()
    assert loaded_paths == [expected_model_path]
    assert expected_model_path != (
        ROOT / "configs" / "experiments" / "ra_ds_pfd_r0_r7.yaml"
    ).resolve()
    assert runtime == {"environment": "tslib"}

    resolved = resolve_p3_b1_variants(B1_SUITE_PATH, project_root=ROOT)
    temporary_root = tmp_path / "runtime-model-yaml"
    temporary_root.mkdir()
    expected_counts = {"B1_LD": 26, "B1_L": 13}
    for variant in VARIANT_IDS:
        model_path = temporary_root / f"{variant}.yaml"
        model_path.write_text(
            yaml.safe_dump(
                RUNNER.resolved_model_document(resolved[variant], runtime),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        document = load_model_config_document(model_path)
        assert document["runtime"] == {"environment": "tslib"}
        assert document["model"] == resolved[variant]
        p3_config = document["model"]["p3"]
        assert (
            len(p3_config["candidate_features"])
            * len(p3_config["candidate_transforms"])
            == expected_counts[variant]
        )
        assert p3_config["top_k"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 3),
        ("candidate_features", list(P3_BASE_FEATURES[:-1])),
        ("selector_temperature", 0.2),
        ("selector_bisection_iterations", 32),
        ("spatial_query_mode", "per_variable"),
        ("batch_size", 1),
    ],
)
def test_b1_rejects_frozen_or_public_overrides(field: str, value: Any) -> None:
    invalid = load_p3_b1_suite(B1_SUITE_PATH)
    invalid["variants"]["B1_L"] = {
        "candidate_transforms": ["level"],
        field: value,
    }
    with pytest.raises(ValueError):
        resolve_p3_b1_variants(invalid, project_root=ROOT)


def test_b1_candidate_bank_counts_and_order_are_deterministic() -> None:
    level_diff = P3CandidateBank(
        FEATURE_COLUMNS,
        candidate_features=P3_BASE_FEATURES,
        candidate_transforms=("level", "diff1"),
    )
    level_only = P3CandidateBank(
        FEATURE_COLUMNS,
        candidate_features=P3_BASE_FEATURES,
        candidate_transforms=("level",),
    )
    assert level_diff.candidate_count == 26
    assert level_only.candidate_count == 13
    assert level_diff.candidate_names[:4] == (
        "Wspd.level",
        "Wspd.diff1",
        "Etmp.level",
        "Etmp.diff1",
    )
    assert level_only.candidate_names == tuple(f"{feature}.level" for feature in P3_BASE_FEATURES)


class _SyntheticP3Model(ForecastModel):
    def __init__(self, candidate_names: tuple[str, ...], top_k: int) -> None:
        super().__init__()
        self.selector = GlobalTopKSelector(candidate_names, top_k=top_k)

    def forward(self, inputs):  # type: ignore[no-untyped-def]
        return inputs.x.new_zeros((inputs.x.shape[0], inputs.x.shape[2], 1))

    def propagation_selection_report(self):  # type: ignore[no-untyped-def]
        return self.selector.selection_report()


def _write_synthetic_run(run_dir: Path, transforms: tuple[str, ...]) -> None:
    run_dir.mkdir(parents=True)
    features = list(P3_BASE_FEATURES)
    candidate_names = tuple(f"{feature}.{transform}" for feature in features for transform in transforms)
    model_config = {
        "pfd_mode": "pfd3_global_topk",
        "p3": {
            "mode": "global_topk",
            "top_k": 2,
            "selector_temperature": 0.1,
            "selector_bisection_iterations": 64,
            "candidate_features": features,
            "candidate_transforms": list(transforms),
        },
    }
    (run_dir / "model_config.yaml").write_text(
        yaml.safe_dump(
            {"runtime": {}, "model": model_config},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data_info = {
        "num_nodes": 1,
        "num_features": len(FEATURE_COLUMNS),
        "lookback": 4,
        "max_pred_len": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "input_power_column": "Patv_clean_for_input",
        "input_power_index": 15,
        "node_ids": [1],
    }
    (run_dir / "resolved_config.yaml").write_text(
        "resolved:\n  data_info:\n"
        + "".join(f"    {key}: {json.dumps(value)}\n" for key, value in data_info.items()),
        encoding="utf-8",
    )
    (run_dir / "run_info.json").write_text(
        json.dumps({"run_id": "synthetic-run"}),
        encoding="utf-8",
    )

    best = _SyntheticP3Model(candidate_names, top_k=2)
    last = _SyntheticP3Model(candidate_names, top_k=2)
    with torch.no_grad():
        best.selector.logits.copy_(torch.linspace(100.0, 1.0, len(candidate_names)))
        last.selector.logits.copy_(torch.linspace(1.0, 100.0, len(candidate_names)))
    save_checkpoint(
        run_dir / "best.pt",
        best,
        manifest={
            "epoch": 3,
            "is_last": False,
            "model": "ra_ds_pfd_crossformer",
            "model_config": model_config,
        },
    )
    save_checkpoint(
        run_dir / "last.pt",
        last,
        manifest={
            "epoch": 4,
            "is_last": True,
            "model": "ra_ds_pfd_crossformer",
            "model_config": model_config,
        },
    )


def test_selection_artifact_uses_best_checkpoint_and_aggregates_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import models.ra_ds_pfd_crossformer.p3_selection as selection_module

    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, ("level", "diff1"))
    monkeypatch.setattr(
        selection_module,
        "build_model",
        lambda _name, config, _info: _SyntheticP3Model(
            tuple(
                f"{feature}.{transform}"
                for feature in config["p3"]["candidate_features"]
                for transform in config["p3"]["candidate_transforms"]
            ),
            int(config["p3"]["top_k"]),
        ),
    )
    artifact = write_p3_selection_best(run_dir, variant="B1_LD", project_root=tmp_path)
    assert artifact["checkpoint_source"] == "best.pt"
    assert artifact["best_epoch"] == 3
    assert artifact["top_k"] == 2
    assert artifact["candidate_count"] == 26
    assert sum(item["selected"] for item in artifact["propagation_feature_scores"]) == 2
    assert [
        item["candidate_name"]
        for item in artifact["propagation_feature_scores"]
        if item["selected"]
    ] == ["Wspd.level", "Wspd.diff1"]
    assert sorted(item["rank"] for item in artifact["propagation_feature_scores"]) == list(range(1, 27))
    assert sum(item["score"] for item in artifact["propagation_feature_scores"]) == pytest.approx(1.0)
    assert sum(item["score"] for item in artifact["base_variable_scores"]) == pytest.approx(1.0)
    assert artifact["operator_scores"]["level_score"] + artifact["operator_scores"]["diff1_score"] == pytest.approx(1.0)
    assert json.loads((run_dir / "p3_selection_best.json").read_text(encoding="utf-8"))["best_epoch"] == 3


def test_level_only_selection_artifact_marks_diff1_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import models.ra_ds_pfd_crossformer.p3_selection as selection_module

    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, ("level",))
    monkeypatch.setattr(
        selection_module,
        "build_model",
        lambda _name, config, _info: _SyntheticP3Model(
            tuple(f"{feature}.level" for feature in config["p3"]["candidate_features"]),
            int(config["p3"]["top_k"]),
        ),
    )
    artifact = write_p3_selection_best(run_dir, variant="B1_L", project_root=tmp_path)
    assert artifact["candidate_count"] == 13
    assert artifact["operator_scores"]["level_score"] == pytest.approx(1.0)
    assert artifact["operator_scores"]["diff1_score"] is None
    assert artifact["operator_scores"]["diff1"]["not_applicable"] is True


def test_selection_artifact_rejects_same_shape_candidate_order_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import models.ra_ds_pfd_crossformer.p3_selection as selection_module

    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, ("level", "diff1"))
    document = yaml.safe_load(
        (run_dir / "model_config.yaml").read_text(encoding="utf-8")
    )
    document["model"]["p3"]["candidate_transforms"] = ["diff1", "level"]
    (run_dir / "model_config.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        selection_module,
        "build_model",
        lambda _name, config, _info: _SyntheticP3Model(
            tuple(
                f"{feature}.{transform}"
                for feature in config["p3"]["candidate_features"]
                for transform in config["p3"]["candidate_transforms"]
            ),
            int(config["p3"]["top_k"]),
        ),
    )

    with pytest.raises(
        ValueError,
        match="best\\.pt model_config does not match run/model_config\\.yaml",
    ):
        write_p3_selection_best(run_dir, variant="B1_LD", project_root=tmp_path)
    assert not (run_dir / "p3_selection_best.json").exists()


def test_b1_runner_dry_run_is_cpu_only_and_reports_both_arms(tmp_path: Path) -> None:
    output_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(B1_SCRIPT),
            "--all",
            "--device",
            "cpu",
            "--run-id",
            "p3-b1-dry-run",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["base"] == "canonical P3 / frozen R2"
    assert [item["variant"] for item in payload["arms"]] == ["B1_LD", "B1_L"]
    assert [item["candidate_count"] for item in payload["arms"]] == [26, 13]
    assert [item["top_k"] for item in payload["arms"]] == [2, 2]
    assert not output_root.exists()


def test_b1_stdout_report_contains_selected_name_score_rank_k_and_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import models.ra_ds_pfd_crossformer.p3_selection as selection_module

    run_dir = tmp_path / "B1_LD"
    _write_synthetic_run(run_dir, ("level", "diff1"))
    monkeypatch.setattr(
        selection_module,
        "build_model",
        lambda _name, config, _info: _SyntheticP3Model(
            tuple(
                f"{feature}.{transform}"
                for feature in config["p3"]["candidate_features"]
                for transform in config["p3"]["candidate_transforms"]
            ),
            int(config["p3"]["top_k"]),
        ),
    )
    artifact = write_p3_selection_best(run_dir, variant="B1_LD", project_root=tmp_path)
    RUNNER.print_p3_b1_selection_report("B1_LD", artifact)
    output = capsys.readouterr().out
    assert "P3-B1 B1_LD" in output
    assert "M = 26" in output
    assert "K = 2" in output
    assert "best.pt" in output
    assert "Wspd.level" in output
    assert "score=" in output
    assert "rank=" in output
    assert "selected=True" in output
    assert "Operator scores" in output
