from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cli.command_schema import build_parser
from runtime.losses import LOSS_NAMES


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, str(ROOT / "scripts" / "run.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_all_public_help_commands_succeed() -> None:
    commands = ("--help", "train", "evaluate", "check", "preflight", "repeatability")
    for command in commands:
        result = _run_cli(command) if command == "--help" else _run_cli(command, "--help")
        assert result.returncode == 0, result.stderr


def test_public_override_defaults_are_none() -> None:
    args = build_parser().parse_args(["train", "--model", "lstm"])
    assert args.model == ["lstm"]
    assert args.lookback is None
    assert args.batch_size is None
    assert args.eval_batch_size is None
    assert args.epochs is None
    assert args.learning_rate is None
    assert args.eval_horizons is None
    assert args.feature_columns is None
    assert args.amp is None


def test_environment_preflight_only_is_train_diagnostic_flag() -> None:
    args = build_parser().parse_args(
        ["train", "--model", "lstm", "--environment-preflight-only"]
    )
    assert args.environment_preflight_only is True


def test_environment_preflight_only_is_available_for_diagnostic_commands() -> None:
    parser = build_parser()
    for command in ("train", "check", "preflight", "repeatability"):
        args = parser.parse_args(
            [command, "--model", "lstm", "--environment-preflight-only"]
        )
        assert args.environment_preflight_only is True


def test_model_accepts_one_or_many_and_legacy_batch_is_gone() -> None:
    args = build_parser().parse_args(["train", "--model", "lstm", "dlinear"])
    assert args.model == ["lstm", "dlinear"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--model", "lstm", "--models", "dlinear"])


def test_run_modes_are_mutually_exclusive_and_loss_choices_are_registry_backed() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--model", "lstm", "--resume", "--overwrite"])
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    train = subparsers.choices["train"]
    loss_action = next(action for action in train._actions if action.dest == "loss")
    assert tuple(loss_action.choices) == LOSS_NAMES


def test_help_dispatch_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            PYTHON,
            "-c",
            "import sys; sys.path.insert(0, 'src'); import cli.main; assert 'torch' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_old_user_entrypoints_are_removed_and_business_modules_have_no_parser() -> None:
    old_names = (
        "run_model.py",
        "run_all_models.py",
        "check_model.py",
        "preflight.py",
        "evaluate.py",
        "compare_repeated_runs.py",
    )
    scripts = ROOT / "scripts"
    assert all(not (scripts / name).exists() for name in old_names)
    assert {path.name for path in scripts.glob("*.py")} == {
        "_bootstrap.py",
        "run.py",
        "run_ra_ds_pfd_r0_r7.py",
        "run_ra_ds_pfd_p3.py",
        "run_ra_ds_pfd_p3_b1.py",
        "run_ra_ds_pfd_p3_b2.py",
        "run_ra_ds_pfd_p3_ia1.py",
        "generate_command_reference.py",
        "build_ra_ds_pfd_relation.py",
    }

    cli_dir = ROOT / "src" / "cli"
    assert not (cli_dir / "batch.py").exists()
    for path in cli_dir.glob("*.py"):
        if path.name in {"command_schema.py", "main.py", "__init__.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert "ArgumentParser(" not in source
        assert "add_argument(" not in source
        assert "parse_args(" not in source
