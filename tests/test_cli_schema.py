from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cli.command_schema import build_parser


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
    commands = ("--help", "train", "evaluate", "check", "preflight", "repeatability", "batch")
    for command in commands:
        result = _run_cli(command) if command == "--help" else _run_cli(command, "--help")
        assert result.returncode == 0, result.stderr


def test_public_override_defaults_are_none() -> None:
    args = build_parser().parse_args(["train", "--model", "node_shared_lstm"])
    assert args.lookback is None
    assert args.batch_size is None
    assert args.epochs is None
    assert args.learning_rate is None
    assert args.eval_horizons is None
    assert args.feature_columns is None
    assert args.amp is None


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
        "generate_command_reference.py",
    }

    cli_dir = ROOT / "src" / "cli"
    for path in cli_dir.glob("*.py"):
        if path.name in {"command_schema.py", "main.py", "__init__.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert "ArgumentParser(" not in source
        assert "add_argument(" not in source
        assert "parse_args(" not in source
