from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ra_ds_pfd_p3_ia1.py"


def _load_runner():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("ra_ds_pfd_p3_ia1_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_ia1_parser_exposes_only_variant_and_execution_controls() -> None:
    destinations = {action.dest for action in RUNNER.build_parser()._actions}
    assert destinations.issuperset(
        {
            "variant",
            "variants",
            "all",
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
            "selected_candidates",
        }
    )


def test_ia1_dry_run_reports_both_resolved_fixed_arms_without_results(tmp_path: Path) -> None:
    output_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--all",
            "--device",
            "cpu",
            "--run-id",
            "ia1-foundation-dryrun",
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
    assert payload["suite"] == "ra_ds_pfd_p3_ia1"
    assert payload["pfd_mode"] == "pfd3_ia_fixed"
    assert [item["variant"] for item in payload["variants"]] == [
        "IA1_R2_PAIR",
        "IA1_AUTO_K2_PAIR",
    ]
    assert payload["variants"][0]["selected_candidates"] == [
        "Wspd.level",
        "Wspd.diff1",
    ]
    assert payload["variants"][1]["selected_candidates"] == [
        "Wspd.level",
        "Patv_clean_for_input.diff1",
    ]
    assert all(item["effective_candidate_count"] == 2 for item in payload["variants"])
    assert all(item["candidate_bank_count"] == 26 for item in payload["variants"])
    assert not output_root.exists()


def test_ia1_resume_requires_an_explicit_run_id() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--all", "--resume", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--resume requires an explicit --run-id" in result.stderr
