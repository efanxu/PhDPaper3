from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ra_ds_pfd_p3.py"


def _load_runner():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("ra_ds_pfd_p3_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_p3_parser_has_only_execution_controls() -> None:
    destinations = {action.dest for action in RUNNER.build_parser()._actions}
    assert destinations.issuperset(
        {
            "run_id",
            "device",
            "output_root",
            "smoke",
            "resume",
            "overwrite",
            "id_suffix",
            "dry_run",
        }
    )
    assert destinations.isdisjoint(
        {
            "batch_size",
            "epochs",
            "seed",
            "loss",
            "learning_rate",
            "top_k",
            "candidate_features",
        }
    )


def test_p3_dry_run_reports_foundation_plan_without_results(tmp_path: Path) -> None:
    output_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--device",
            "cpu",
            "--run-id",
            "p3-a-foundation-dryrun",
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
    assert payload["base_variant"] == "R2"
    assert payload["pfd_mode"] == "pfd3_global_topk"
    assert payload["top_k"] == 2
    assert payload["candidate_count"] == 26
    assert payload["execution_mode"] == "full_spatiotemporal"
    assert payload["candidate_names"][0] == "Wspd.level"
    assert payload["candidate_names"][-1] == "Patv_clean_for_input.diff1"
    assert not output_root.exists()
    command = payload["planned_public_command"]
    assert command[2:5] == ["train", "--model", "ra_ds_pfd_crossformer"]
    assert "--fail-fast" in command


def test_resume_requires_an_explicit_run_id() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--resume", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--resume requires an explicit --run-id" in result.stderr
