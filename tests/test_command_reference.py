from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generated_command_reference_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_command_reference.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    document = (ROOT / "docs" / "COMMAND_REFERENCE.md").read_text(encoding="utf-8")
    for command in ("train", "evaluate", "check", "preflight", "repeatability", "batch"):
        assert f"## `{command}`" in document
    assert "内置结构默认 < configs/experiment.yaml < 命令行显式覆盖" in document
    assert "`training.train_batch_size`" in document
    assert "`data.eval_horizons`" in document
