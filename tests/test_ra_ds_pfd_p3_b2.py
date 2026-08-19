from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
import yaml

from engine.checkpoint import save_checkpoint
from models.base import ForecastModel
from models.ra_ds_pfd_crossformer.p3_b2_suite import (
    CANDIDATE_COUNT,
    FROZEN_OPERATOR_BASIS,
    K_GRID,
    VARIANT_IDS,
    aggregate_p3_b2_k_selection,
    load_p3_b2_suite,
    p3_b2_summary_path,
    resolve_p3_b2_variants,
    write_p3_b2_k_selection,
)
from models.ra_ds_pfd_crossformer.p3_feature_bank import P3_BASE_FEATURES
from models.ra_ds_pfd_crossformer.p3_selection import write_p3_selection_best
from models.ra_ds_pfd_crossformer.p3_selector import GlobalTopKSelector
from runtime.config import load_model_config_document


ROOT = Path(__file__).resolve().parents[1]
B2_SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_p3_b2.yaml"
B2_SCRIPT = ROOT / "scripts" / "run_ra_ds_pfd_p3_b2.py"
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


def _load_b2_runner():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("ra_ds_pfd_p3_b2_runner", B2_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_b2_runner()


def _candidate_names(top_k: int | None = None) -> tuple[str, ...]:
    del top_k
    return tuple(
        f"{feature}.{transform}"
        for feature in P3_BASE_FEATURES
        for transform in FROZEN_OPERATOR_BASIS
    )


class _SyntheticP3Model(ForecastModel):
    def __init__(self, candidate_names: tuple[str, ...], top_k: int) -> None:
        super().__init__()
        self.selector = GlobalTopKSelector(candidate_names, top_k=top_k)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):  # type: ignore[no-untyped-def]
        return inputs.x.new_zeros((inputs.x.shape[0], inputs.x.shape[2], 1))

    def propagation_selection_report(self):  # type: ignore[no-untyped-def]
        return self.selector.selection_report()


def _model_config(top_k: int) -> dict[str, Any]:
    return {
        "pfd_mode": "pfd3_global_topk",
        "p3": {
            "mode": "global_topk",
            "top_k": top_k,
            "selector_temperature": 0.1,
            "selector_bisection_iterations": 64,
            "candidate_features": list(P3_BASE_FEATURES),
            "candidate_transforms": list(FROZEN_OPERATOR_BASIS),
        },
    }


def _write_synthetic_run(
    run_dir: Path,
    *,
    top_k: int,
    validation_monitor: float | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    candidate_names = _candidate_names()
    model_config = _model_config(top_k)
    (run_dir / "model_config.yaml").write_text(
        yaml.safe_dump({"runtime": {}, "model": model_config}, sort_keys=False),
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
        json.dumps({"run_id": run_dir.name}),
        encoding="utf-8",
    )

    best = _SyntheticP3Model(candidate_names, top_k=top_k)
    last = _SyntheticP3Model(candidate_names, top_k=top_k)
    with torch.no_grad():
        best.selector.logits.copy_(torch.linspace(100.0, 1.0, len(candidate_names)))
        last.selector.logits.copy_(torch.linspace(1.0, 100.0, len(candidate_names)))
    monitor = 18.0 if validation_monitor is None else float(validation_monitor)
    selection = {
        "split": "validation",
        "horizon": "all",
        "metric": "SDWPF Official Score",
        "mode": "min",
    }
    manifest = {
        "epoch": 3,
        "monitor": monitor,
        "monitor_name": "SDWPF Official Score",
        "checkpoint_selection": selection,
        "is_last": False,
        "model": "ra_ds_pfd_crossformer",
        "model_config": model_config,
    }
    save_checkpoint(run_dir / "best.pt", best, manifest=manifest)
    last_manifest = dict(manifest)
    last_manifest["epoch"] = 4
    last_manifest["is_last"] = True
    save_checkpoint(run_dir / "last.pt", last, manifest=last_manifest)
    (run_dir / "metrics_validation.json").write_text(
        json.dumps({"monitor": monitor, "by_horizon": {}}),
        encoding="utf-8",
    )


def _write_selection_artifact(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    *,
    top_k: int,
    validation_monitor: float | None = None,
) -> dict[str, Any]:
    import models.ra_ds_pfd_crossformer.p3_selection as selection_module

    _write_synthetic_run(
        run_dir,
        top_k=top_k,
        validation_monitor=validation_monitor,
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
    return write_p3_selection_best(
        run_dir,
        variant=f"B2_K{top_k}",
        project_root=run_dir.parent,
    )


def test_b2_suite_resolves_exact_k_grid_from_canonical_p3() -> None:
    suite = load_p3_b2_suite(B2_SUITE_PATH)
    resolved = resolve_p3_b2_variants(B2_SUITE_PATH, project_root=ROOT)

    assert tuple(resolved) == VARIANT_IDS
    assert tuple(int(resolved[variant]["p3"]["top_k"]) for variant in VARIANT_IDS) == K_GRID
    assert all(
        resolved[variant]["p3"]["candidate_transforms"] == ["level", "diff1"]
        for variant in VARIANT_IDS
    )
    assert all(
        len(resolved[variant]["p3"]["candidate_features"])
        * len(resolved[variant]["p3"]["candidate_transforms"])
        == CANDIDATE_COUNT
        for variant in VARIANT_IDS
    )
    assert suite["base"]["suite_file"] == "configs/experiments/ra_ds_pfd_p3.yaml"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_transforms", ["level"]),
        ("candidate_features", list(P3_BASE_FEATURES[:-1])),
        ("selector_temperature", 0.2),
        ("selector_bisection_iterations", 32),
        ("spatial_query_mode", "per_variable"),
        ("batch_size", 1),
        ("amp", False),
        ("seed", 7),
    ],
)
def test_b2_rejects_every_non_top_k_variant_override(field: str, value: Any) -> None:
    invalid = load_p3_b2_suite(B2_SUITE_PATH)
    invalid["variants"]["B2_K2"] = {"top_k": 2, field: value}
    with pytest.raises(ValueError):
        resolve_p3_b2_variants(invalid, project_root=ROOT)


def test_b2_fails_closed_when_canonical_operator_basis_is_not_level_diff1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import models.ra_ds_pfd_crossformer.p3_b2_suite as suite_module

    canonical = resolve_p3_b2_variants(B2_SUITE_PATH, project_root=ROOT)["B2_K2"]
    canonical["p3"]["candidate_transforms"] = ["level"]
    monkeypatch.setattr(
        suite_module,
        "resolve_p3_model_config",
        lambda *_args, **_kwargs: deepcopy(canonical),
    )
    with pytest.raises(
        ValueError,
        match=r"P3-B2 currently requires the frozen Level\+Diff1 operator basis",
    ):
        suite_module.resolve_p3_b2_variants(B2_SUITE_PATH, project_root=ROOT)


@pytest.mark.parametrize("top_k", [0, -1, 27, True, 1.5])
def test_b2_rejects_top_k_outside_candidate_count(top_k: object) -> None:
    invalid = load_p3_b2_suite(B2_SUITE_PATH)
    invalid["variants"]["B2_K2"] = {"top_k": top_k}
    with pytest.raises(ValueError, match="top_k"):
        resolve_p3_b2_variants(invalid, project_root=ROOT)


def test_b2_runtime_resolution_reaches_model_yaml_and_tslib(tmp_path: Path) -> None:
    runtime = RUNNER._base_runtime_document()
    assert runtime == {"environment": "tslib"}

    resolved = resolve_p3_b2_variants(B2_SUITE_PATH, project_root=ROOT)
    model_documents: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_IDS:
        path = tmp_path / f"{variant}.yaml"
        path.write_text(
            yaml.safe_dump(
                RUNNER.resolved_model_document(resolved[variant], runtime),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        document = load_model_config_document(path)
        model_documents[variant] = document
        assert document["runtime"] == {"environment": "tslib"}
        assert document["model"] == resolved[variant]
        assert document["model"]["p3"]["candidate_transforms"] == ["level", "diff1"]
        assert (
            len(document["model"]["p3"]["candidate_features"])
            * len(document["model"]["p3"]["candidate_transforms"])
            == 26
        )

    baseline = deepcopy(model_documents["B2_K2"]["model"])
    for variant in VARIANT_IDS:
        current = deepcopy(model_documents[variant]["model"])
        current["p3"]["top_k"] = baseline["p3"]["top_k"]
        assert current == baseline


def test_b2_runner_dry_run_is_cpu_only_and_has_no_result_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(B2_SCRIPT),
            "--all",
            "--device",
            "cpu",
            "--run-id",
            "p3-b2-dry-run",
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
    assert payload["operator_basis"] == ["level", "diff1"]
    assert payload["candidate_count"] == 26
    assert payload["k_grid"] == [1, 2, 3, 4, 6, 8]
    assert payload["dry_run_summary"] == [
        "B2_K1 K=1 M=26",
        "B2_K2 K=2 M=26",
        "B2_K3 K=3 M=26",
        "B2_K4 K=4 M=26",
        "B2_K6 K=6 M=26",
        "B2_K8 K=8 M=26",
    ]
    assert not output_root.exists()


def test_b2_smoke_complete_grid_writes_readouts_but_no_k_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "smoke-results"
    args = RUNNER.build_parser().parse_args(
        [
            "--all",
            "--device",
            "cpu",
            "--run-id",
            "seed2026",
            "--output-root",
            str(output_root),
            "--smoke",
        ]
    )
    plan = RUNNER.build_plan(args)
    writer_calls: list[tuple[object, ...]] = []

    def fake_train(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    def fake_selection_writer(
        run_directory: str | Path,
        *,
        variant: str,
        project_root: str | Path,
    ) -> dict[str, Any]:
        del project_root
        run_path = Path(run_directory)
        run_path.mkdir(parents=True, exist_ok=True)
        top_k = int(variant.removeprefix("B2_K"))
        artifact = {"variant": variant, "top_k": top_k, "readout": "synthetic"}
        (run_path / "p3_selection_best.json").write_text(
            json.dumps(artifact),
            encoding="utf-8",
        )
        return artifact

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_train)
    monkeypatch.setattr(RUNNER, "write_p3_selection_best", fake_selection_writer)
    monkeypatch.setattr(RUNNER, "print_p3_b2_selection_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        RUNNER,
        "write_p3_b2_k_selection",
        lambda *call_args, **call_kwargs: writer_calls.append(
            (call_args, call_kwargs)
        ),
    )

    assert RUNNER.execute_plan(args, plan) == 0
    assert len(list(output_root.rglob("p3_selection_best.json"))) == len(VARIANT_IDS)
    assert writer_calls == []
    assert list(output_root.rglob("p3_b2_k_selection.json")) == []



@pytest.mark.parametrize("top_k", [1, 4, 8])
def test_b2_selection_reuses_best_checkpoint_for_each_cardinality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    top_k: int,
) -> None:
    artifact = _write_selection_artifact(
        monkeypatch,
        tmp_path / f"B2_K{top_k}",
        top_k=top_k,
    )
    assert artifact["checkpoint_source"] == "best.pt"
    assert artifact["candidate_count"] == 26
    assert artifact["top_k"] == top_k
    assert sum(item["selected"] for item in artifact["propagation_feature_scores"]) == top_k
    assert sorted(item["rank"] for item in artifact["propagation_feature_scores"]) == list(range(1, 27))
    assert sum(item["score"] for item in artifact["propagation_feature_scores"]) == pytest.approx(1.0)
    assert artifact["candidate_transforms"] == ["level", "diff1"]


def test_b2_selection_keeps_model_config_provenance_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "B2_K2"
    _write_selection_artifact(monkeypatch, run_dir, top_k=2)
    document = yaml.safe_load((run_dir / "model_config.yaml").read_text(encoding="utf-8"))
    document["model"]["p3"]["top_k"] = 3
    (run_dir / "model_config.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "p3_selection_best.json").unlink()
    with pytest.raises(ValueError, match="best\\.pt model_config does not match"):
        write_p3_selection_best(
            run_dir,
            variant="B2_K2",
            project_root=tmp_path,
        )
    assert not (run_dir / "p3_selection_best.json").exists()


def test_b2_summary_reports_provisional_k4_from_validation_only_and_blocks_incomplete_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validation = {
        "B2_K1": 20.0,
        "B2_K2": 18.5,
        "B2_K3": 18.0,
        "B2_K4": 17.8,
        "B2_K6": 18.1,
        "B2_K8": 18.4,
    }
    runs: dict[str, Path] = {}
    for variant in VARIANT_IDS:
        k = int(variant.removeprefix("B2_K"))
        path = tmp_path / variant
        _write_selection_artifact(
            monkeypatch,
            path,
            top_k=k,
            validation_monitor=validation[variant],
        )
        runs[variant] = path

    for path in runs.values():
        (path / "metrics_test_h3.json").write_text(
            json.dumps({"monitor": -999.0}),
            encoding="utf-8",
        )

    summary = aggregate_p3_b2_k_selection(runs, suite_run_id="seed2026")
    assert summary["suite_run_id"] == "seed2026"
    assert summary["selection_status"] == "PROVISIONAL"
    assert summary["provisional_best_k"] == 4
    assert summary["provisional_best_variant"] == "B2_K4"
    assert summary["runner_up_k"] == 3
    assert summary["runner_up_variant"] == "B2_K3"
    assert summary["validation_gap"] == pytest.approx(0.2)
    assert summary["selected_k"] is None
    assert summary["selected_variant"] is None
    assert summary["selection_uses_test"] is False
    assert summary["selection_metric"] == "SDWPF Official Score"
    assert [entry["k"] for entry in summary["runs"]] == [1, 2, 3, 4, 6, 8]
    assert all(entry["validation_monitor_source"] == "best.pt:manifest.monitor" for entry in summary["runs"])
    assert all(len(entry["selected_features"]) == entry["k"] for entry in summary["runs"])

    summary_path = p3_b2_summary_path(runs["B2_K4"])
    written = write_p3_b2_k_selection(
        summary_path,
        runs,
        suite_run_id="seed2026",
    )
    written_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_payload["selection_status"] == "PROVISIONAL"
    assert written_payload["provisional_best_k"] == 4
    assert written["selected_k"] is None

    del runs["B2_K6"]
    incomplete = aggregate_p3_b2_k_selection(runs, strict=False)
    assert incomplete["selection_status"] == "INCOMPLETE"
    assert incomplete["provisional_best_k"] is None
    assert incomplete["selected_k"] is None
    assert incomplete["selected_variant"] is None
    assert incomplete["missing_variants"] == ["B2_K6"]
    with pytest.raises(ValueError, match="K grid is incomplete"):
        aggregate_p3_b2_k_selection(runs)


def test_b2_summary_exact_validation_tie_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validation = {
        "B2_K1": 20.0,
        "B2_K2": 18.5,
        "B2_K3": 17.8,
        "B2_K4": 17.8,
        "B2_K6": 18.1,
        "B2_K8": 18.4,
    }
    runs: dict[str, Path] = {}
    for variant in VARIANT_IDS:
        k = int(variant.removeprefix("B2_K"))
        path = tmp_path / variant
        _write_selection_artifact(
            monkeypatch,
            path,
            top_k=k,
            validation_monitor=validation[variant],
        )
        runs[variant] = path

    summary = aggregate_p3_b2_k_selection(runs, suite_run_id="seed2026")
    assert summary["selection_status"] == "AMBIGUOUS"
    assert summary["ambiguous_variants"] == ["B2_K3", "B2_K4"]
    assert summary["provisional_best_k"] is None
    assert summary["provisional_best_variant"] is None
    assert summary["selected_k"] is None
    assert summary["selected_variant"] is None


def test_b2_summary_path_isolated_between_suite_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validation = {
        "B2_K1": 20.0,
        "B2_K2": 18.5,
        "B2_K3": 18.0,
        "B2_K4": 17.8,
        "B2_K6": 18.1,
        "B2_K8": 18.4,
    }
    suites: dict[str, dict[str, Path]] = {}
    for seed, offset in (("seed2026", 0.0), ("seed2027", 10.0)):
        runs: dict[str, Path] = {}
        for variant in VARIANT_IDS:
            k = int(variant.removeprefix("B2_K"))
            path = tmp_path / seed / f"{seed}__{variant}"
            _write_selection_artifact(
                monkeypatch,
                path,
                top_k=k,
                validation_monitor=validation[variant] + offset,
            )
            runs[variant] = path
        suites[seed] = runs

    summary_2026 = p3_b2_summary_path(suites["seed2026"]["B2_K4"])
    summary_2027 = p3_b2_summary_path(suites["seed2027"]["B2_K4"])
    assert summary_2026 != summary_2027

    write_p3_b2_k_selection(
        summary_2026,
        suites["seed2026"],
        suite_run_id="seed2026",
    )
    write_p3_b2_k_selection(
        summary_2027,
        suites["seed2027"],
        suite_run_id="seed2027",
    )

    payload_2026 = json.loads(summary_2026.read_text(encoding="utf-8"))
    payload_2027 = json.loads(summary_2027.read_text(encoding="utf-8"))
    assert payload_2026["suite_run_id"] == "seed2026"
    assert payload_2027["suite_run_id"] == "seed2027"
    assert payload_2026["provisional_best_k"] == 4
    assert payload_2027["provisional_best_k"] == 4
    assert payload_2026["runs"][3]["validation_monitor"] == pytest.approx(17.8)
    assert payload_2027["runs"][3]["validation_monitor"] == pytest.approx(27.8)


def test_b2_stdout_report_contains_selected_name_score_rank_k_and_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = _write_selection_artifact(monkeypatch, tmp_path / "B2_K2", top_k=2)
    RUNNER.print_p3_b2_selection_report("B2_K2", artifact)
    output = capsys.readouterr().out
    assert "P3-B2 B2_K2" in output
    assert "K = 2" in output
    assert "best.pt" in output
    assert "Wspd.level" in output
    assert "score=" in output
    assert "rank=" in output
    assert "Base-variable scores" in output
    assert "Operator scores" in output
