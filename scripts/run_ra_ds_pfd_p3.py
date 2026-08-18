"""Launch the P3-A foundation through the existing public training CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import yaml

from _bootstrap import ROOT  # noqa: F401
from models.ra_ds_pfd_crossformer.p3_suite import (
    BASE_VARIANT,
    DEFAULT_SUITE_PATH,
    load_p3_suite,
    resolve_p3_model_config,
)
from models.ra_ds_pfd_crossformer.p3_selector import SELECTOR_TYPE
from runtime.config import load_experiment_config, load_model_config_document
from runtime.paths import effective_run_id, validate_run_id


MODEL_NAME = "ra_ds_pfd_crossformer"
PUBLIC_CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
SUITE_PATH = ROOT / DEFAULT_SUITE_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the P3-A RA-DS-PFD propagation foundation through the "
            "public orchestrator/worker/Trainer path."
        )
    )
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--overwrite", action="store_true")
    modes.add_argument("--id-suffix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the plan without GPU loading, training, or result writes",
    )
    return parser


def base_run_id(value: str | None) -> str:
    return validate_run_id(value or datetime.now().strftime("ra-ds-pfd-p3-%Y%m%d-%H%M%S"))


def public_command(
    *,
    run_id: str,
    device: str,
    output_root: Path | None,
    model_config_path: Path | None,
    resume: bool,
    overwrite: bool,
    id_suffix: str | None,
    smoke: bool,
) -> list[str]:
    config_display = (
        str(model_config_path)
        if model_config_path is not None
        else "<temporary-resolved-model-yaml:P3>"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run.py"),
        "train",
        "--model",
        MODEL_NAME,
        "--config",
        str(PUBLIC_CONFIG_PATH),
        "--model-config",
        config_display,
        "--run-id",
        run_id,
        "--device",
        device,
        "--fail-fast",
    ]
    if output_root is not None:
        command.extend(("--output-root", str(output_root)))
    if smoke:
        command.append("--smoke")
    if resume:
        command.append("--resume")
    elif overwrite:
        command.append("--overwrite")
    elif id_suffix is not None:
        command.extend(("--id-suffix", id_suffix))
    return command


def _candidate_names(p3_config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        f"{feature}.{transform}"
        for feature in p3_config["candidate_features"]
        for transform in p3_config["candidate_transforms"]
    )


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite = load_p3_suite(SUITE_PATH)
    resolved = resolve_p3_model_config(SUITE_PATH, project_root=ROOT)
    load_experiment_config(PUBLIC_CONFIG_PATH)
    run_id = base_run_id(args.run_id)
    p3_config = dict(resolved["p3"])
    candidate_names = _candidate_names(p3_config)
    planned_command = public_command(
        run_id=run_id,
        device=args.device,
        output_root=args.output_root,
        model_config_path=None,
        resume=args.resume,
        overwrite=args.overwrite,
        id_suffix=args.id_suffix,
        smoke=args.smoke,
    )
    item = {
        "suite": suite["suite"],
        "base_variant": BASE_VARIANT,
        "pfd_mode": resolved["pfd_mode"],
        "selector_type": SELECTOR_TYPE,
        "top_k": int(p3_config["top_k"]),
        "selector_temperature": float(p3_config["selector_temperature"]),
        "selector_bisection_iterations": int(
            p3_config["selector_bisection_iterations"]
        ),
        "candidate_base_features": list(p3_config["candidate_features"]),
        "candidate_transforms": list(p3_config["candidate_transforms"]),
        "candidate_count": len(candidate_names),
        "candidate_names": list(candidate_names),
        "execution_mode": "full_spatiotemporal",
        "relation_resource": dict(resolved["relation_resource"]),
        "spatial_edge_chunk_size": resolved["spatial_edge_chunk_size"],
        "architecture_axes": {
            field: resolved[field]
            for field in (
                "spatial_query_mode",
                "propagation_encoder_mode",
                "turbine_embedding_mode",
                "bias_scaling_mode",
            )
        },
        "resolved_model_config_source": (
            f"{SUITE_PATH} -> frozen R2 -> pfd3_global_topk"
        ),
        "temporary_model_yaml": "generated only during execution",
        "public_experiment_config": str(PUBLIC_CONFIG_PATH),
        "node_shared_chunk_size": "not applicable",
        "planned_run_identity": effective_run_id(run_id, args.id_suffix),
        "planned_public_command": planned_command,
        "planned_command": planned_command,
    }
    return [item]


def _base_runtime_document() -> dict[str, Any]:
    suite = load_p3_suite(SUITE_PATH)
    base_suite_path = ROOT / suite["base"]["suite_file"]
    base_suite = yaml.safe_load(base_suite_path.read_text(encoding="utf-8"))
    model_file = ROOT / base_suite["base_model_config"]["file"]
    document = load_model_config_document(model_file)
    return dict(document["runtime"])


def resolved_model_document(
    resolved: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Wrap the P3 resolver output in the existing model YAML document shape."""

    return {"runtime": dict(runtime), "model": dict(resolved)}


def execute_plan(args: argparse.Namespace, plan: list[dict[str, Any]]) -> int:
    resolved = resolve_p3_model_config(SUITE_PATH, project_root=ROOT)
    runtime = _base_runtime_document()
    temporary_root = Path(tempfile.mkdtemp(prefix="ra_ds_pfd_p3_"))
    model_path = temporary_root / "P3.yaml"
    try:
        model_path.write_text(
            yaml.safe_dump(
                resolved_model_document(resolved, runtime),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        item = plan[0]
        run_id = str(item["planned_run_identity"])
        if args.id_suffix:
            run_id = run_id.removesuffix(f"__{args.id_suffix}")
        command = public_command(
            run_id=run_id,
            device=args.device,
            output_root=args.output_root,
            model_config_path=model_path,
            resume=args.resume,
            overwrite=args.overwrite,
            id_suffix=args.id_suffix,
            smoke=args.smoke,
        )
        print(f"Launching P3-A: {json.dumps(command, ensure_ascii=False)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        return int(completed.returncode or 0)
    finally:
        model_path.unlink(missing_ok=True)
        temporary_root.rmdir()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")
    if args.id_suffix is not None:
        validate_run_id(args.id_suffix, label="id-suffix")
    plan = build_plan(args)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "smoke": bool(args.smoke),
                "suite": plan[0]["suite"],
                "base_variant": plan[0]["base_variant"],
                "pfd_mode": plan[0]["pfd_mode"],
                "selector_type": plan[0]["selector_type"],
                "top_k": plan[0]["top_k"],
                "candidate_transforms": plan[0]["candidate_transforms"],
                "candidate_count": plan[0]["candidate_count"],
                "candidate_names": plan[0]["candidate_names"],
                "execution_mode": plan[0]["execution_mode"],
                "relation_resource": plan[0]["relation_resource"],
                "planned_public_command": plan[0]["planned_public_command"],
                "planned_command": plan[0]["planned_command"],
                "plan": plan,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    return execute_plan(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
