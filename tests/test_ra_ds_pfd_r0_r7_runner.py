from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from cli.train import _check_checkpoint_compatibility, _write_model_invocation_artifacts
from models.ra_ds_pfd_crossformer.r0_r7_suite import (
    VARIANT_IDS,
    resolve_r0_r7_variant,
    resolve_r0_r7_variants,
)
from runtime.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ra_ds_pfd_r0_r7.py"


def _load_runner():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("ra_ds_pfd_r0_r7_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _parse(*values: str):
    return RUNNER.build_parser().parse_args(list(values))


def _dry_run(*values: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *values, "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_single_multiple_and_all_selection_are_strict_and_stable() -> None:
    assert RUNNER.selected_variants(_parse("--variant", "R3")) == ("R3",)
    assert RUNNER.selected_variants(_parse("--variants", "R1,R3,R4")) == (
        "R1",
        "R3",
        "R4",
    )
    assert RUNNER.selected_variants(_parse("--all")) == VARIANT_IDS

    for values in (
        ("--variant", "R8"),
        ("--variant", "r1"),
        ("--variants", "R1,R1"),
        ("--variants", "R1,,R2"),
        ("--all", "--variant", "R1"),
    ):
        with pytest.raises(SystemExit):
            _parse(*values)


def test_runner_exposes_no_public_protocol_overrides() -> None:
    destinations = {action.dest for action in RUNNER.build_parser()._actions}
    assert destinations.isdisjoint(
        {
            "batch_size",
            "epochs",
            "seed",
            "amp",
            "loss",
            "learning_rate",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "spatial_edge_chunk_size",
            "smoke_epochs",
            "smoke_max_train_updates",
            "smoke_max_eval_batches",
        }
    )


def test_smoke_parser_is_a_single_boolean_switch() -> None:
    assert _parse("--variant", "R1", "--smoke").smoke is True


def test_plan_uses_resolver_output_and_unique_variant_identities() -> None:
    args = _parse("--all", "--run-id", "suite-seed2026", "--device", "cuda")
    plan = RUNNER.build_plan(args)
    resolved = resolve_r0_r7_variants(RUNNER.SUITE_PATH, project_root=ROOT)
    runtime = RUNNER._base_runtime_document()

    assert tuple(item["variant"] for item in plan) == VARIANT_IDS
    assert len({item["planned_run_identity"] for item in plan}) == len(VARIANT_IDS)
    for item in plan:
        variant = item["variant"]
        document = RUNNER.resolved_model_document(variant, resolved, runtime)
        assert document["model"] == resolve_r0_r7_variant(
            RUNNER.SUITE_PATH,
            variant,
            project_root=ROOT,
        )
        assert document["runtime"] == runtime
        assert item["planned_run_identity"] == f"suite-seed2026__{variant}"
        assert item["public_experiment_config"] == str(ROOT / "configs" / "experiment.yaml")


def test_r0_and_spatial_execution_plans_report_the_active_chunk_only() -> None:
    plan = RUNNER.build_plan(_parse("--all", "--run-id", "execution-audit"))
    r0, *spatial = plan
    assert r0["variant"] == "R0"
    assert r0["spatial_disabled"] is True
    assert r0["execution_mode"] == "node_shared_microbatch"
    assert r0["node_shared_chunk_size"] == 32
    assert r0["spatial_edge_chunk_size"] == "not applicable"
    for item in spatial:
        assert item["spatial_disabled"] is False
        assert item["execution_mode"] == "full_spatiotemporal"
        assert item["node_shared_chunk_size"] == "not applicable"
        assert item["spatial_edge_chunk_size"] == 512


def test_dry_run_writes_no_results_and_prints_public_invocation(tmp_path: Path) -> None:
    output_root = tmp_path / "formal-results-must-not-exist"
    result = _dry_run(
        "--variants",
        "R1,R4",
        "--run-id",
        "dry-run-audit",
        "--output-root",
        str(output_root),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["smoke"] is False
    assert [item["variant"] for item in payload["variants"]] == ["R1", "R4"]
    assert not output_root.exists()
    for item in payload["variants"]:
        command = item["planned_command"]
        assert command[2:5] == ["train", "--model", "ra_ds_pfd_crossformer"]
        assert "--fail-fast" in command
        assert "--smoke" not in command


def test_smoke_dry_run_only_adds_public_smoke_forwarding(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke-results-must-not-exist"
    normal = RUNNER.build_plan(
        _parse("--all", "--run-id", "smoke-forwarding", "--device", "cuda")
    )
    smoke = RUNNER.build_plan(
        _parse("--all", "--run-id", "smoke-forwarding", "--device", "cuda", "--smoke")
    )
    assert [item["variant"] for item in smoke] == list(VARIANT_IDS)
    for normal_item, smoke_item in zip(normal, smoke, strict=True):
        normal_command = normal_item.pop("planned_command")
        smoke_command = smoke_item.pop("planned_command")
        assert normal_item == smoke_item
        assert "--smoke" not in normal_command
        assert smoke_command == [*normal_command, "--smoke"]

    result = _dry_run(
        "--all",
        "--run-id",
        "smoke-forwarding",
        "--device",
        "cuda",
        "--smoke",
        "--output-root",
        str(output_root),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["smoke"] is True
    assert [item["variant"] for item in payload["variants"]] == list(VARIANT_IDS)
    assert all("--smoke" in item["planned_command"] for item in payload["variants"])
    assert not output_root.exists()


def test_command_provenance_preserves_invocation_and_stable_replay_config(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "result"
    original_model_path = tmp_path / "temporary-R1.yaml"
    resolved = resolve_r0_r7_variants(RUNNER.SUITE_PATH, project_root=ROOT)
    document = RUNNER.resolved_model_document(
        "R1", resolved, RUNNER._base_runtime_document()
    )
    argv = ["train", "--model-config", str(original_model_path), "--smoke"]
    _write_model_invocation_artifacts(
        output_dir=output_dir,
        command_argv=argv,
        command_name="train",
        model_name="ra_ds_pfd_crossformer",
        run_name="provenance__R1",
        config_file=ROOT / "configs" / "experiment.yaml",
        model_file=original_model_path,
        model_document=document,
        effective_overrides={},
    )

    command = json.loads((output_dir / "command.json").read_text(encoding="utf-8"))
    replay_path = Path(command["replay_model_config_path"])
    assert command["argv"] == argv
    assert command["model_config_path"] == str(original_model_path)
    assert replay_path == output_dir / "model_config.yaml"
    assert replay_path.is_file()
    assert yaml.safe_load(replay_path.read_text(encoding="utf-8")) == document


def test_execution_generates_exact_temporary_yaml_and_fails_fast(monkeypatch) -> None:
    args = _parse("--variants", "R1,R3,R4", "--run-id", "mock-execution")
    plan = RUNNER.build_plan(args)
    expected = resolve_r0_r7_variants(RUNNER.SUITE_PATH, project_root=ROOT)
    captured: list[tuple[str, dict]] = []
    temporary_paths: list[Path] = []

    def fake_run(command, *, cwd, check):
        del cwd, check
        variant = command[command.index("--run-id") + 1].rsplit("__", 1)[-1]
        path = Path(command[command.index("--model-config") + 1])
        temporary_paths.append(path)
        captured.append((variant, yaml.safe_load(path.read_text(encoding="utf-8"))))
        return SimpleNamespace(returncode=7 if variant == "R3" else 0)

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    assert RUNNER.execute_plan(args, plan) == 7
    assert [variant for variant, _ in captured] == ["R1", "R3"]
    for variant, document in captured:
        assert document["model"] == expected[variant]
        assert document["runtime"] == RUNNER._base_runtime_document()
    assert all(not path.exists() for path in temporary_paths)
    assert not any((ROOT / "configs" / "models").glob("ra_ds_pfd_r[0-7].yaml"))


def test_resume_identity_uses_model_content_not_temporary_path() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    resolved = resolve_r0_r7_variants(RUNNER.SUITE_PATH, project_root=ROOT)
    manifest = {
        "model": "ra_ds_pfd_crossformer",
        "epoch": 0,
        "resolved_config": config.copy_values(),
        "model_config": resolved["R3"],
        "runtime_state": {"rng": {}, "dataloader_generators": {}},
    }
    plan = SimpleNamespace(uses_node_microbatch=False)
    _check_checkpoint_compatibility(
        manifest,
        config,
        resolved["R3"],
        Path("temporary-R3-a.yaml"),
        model_name="ra_ds_pfd_crossformer",
        execution_plan=plan,
    )
    with pytest.raises(ValueError, match="incompatible with the current model config"):
        _check_checkpoint_compatibility(
            manifest,
            config,
            resolved["R4"],
            Path("different-temporary-R4-b.yaml"),
            model_name="ra_ds_pfd_crossformer",
            execution_plan=plan,
        )


def test_resume_requires_explicit_base_identity() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--variant", "R3", "--resume", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--resume requires an explicit --run-id" in result.stderr
