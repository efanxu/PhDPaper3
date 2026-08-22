"""Launch IA-1.1 temporal-closure arms through the existing public CLI."""

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
from models.ra_ds_pfd_crossformer.p3_ia11_suite import (
    BASE_VARIANT,
    DEFAULT_SUITE_PATH,
    VARIANT_IDS,
    load_p3_ia11_suite,
    resolve_p3_ia11_variants,
)
from models.ra_ds_pfd_crossformer.r0_r7_suite import load_r0_r7_suite
from runtime.config import load_experiment_config, load_model_config_document
from runtime.paths import effective_run_id, validate_run_id


MODEL_NAME = "ra_ds_pfd_crossformer"
PUBLIC_CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
SUITE_PATH = ROOT / DEFAULT_SUITE_PATH


def _variant_list(value: str) -> tuple[str, ...]:
    parts = value.split(",")
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--variants must not contain an empty variant")
    unknown = [part for part in parts if part not in VARIANT_IDS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported variant: {unknown[0]}")
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            raise argparse.ArgumentTypeError(f"duplicate variant: {part}")
        seen.add(part)
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run IA-1.1 selected-only RA-DS-PFD temporal-closure arms through "
            "the public orchestrator/worker/Trainer path."
        )
    )
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--variant", choices=VARIANT_IDS, help="run one exact arm")
    selectors.add_argument(
        "--variants",
        type=_variant_list,
        help="run a comma-separated, duplicate-free arm list",
    )
    selectors.add_argument("--all", action="store_true", help="run both IA-1.1 arms in order")
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


def selected_variants(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all:
        return VARIANT_IDS
    if args.variant is not None:
        return (args.variant,)
    assert args.variants is not None
    return tuple(args.variants)


def base_run_id(value: str | None) -> str:
    return validate_run_id(
        value or datetime.now().strftime("ra-ds-pfd-p3-ia11-%Y%m%d-%H%M%S")
    )


def variant_run_id(base: str, variant: str, id_suffix: str | None) -> str:
    return effective_run_id(f"{base}__{variant}", id_suffix)


def public_command(
    *,
    variant: str,
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
        else f"<temporary-resolved-model-yaml:{variant}>"
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


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite = load_p3_ia11_suite(SUITE_PATH)
    resolved = resolve_p3_ia11_variants(SUITE_PATH, project_root=ROOT)
    load_experiment_config(PUBLIC_CONFIG_PATH)
    run_base = base_run_id(args.run_id)
    plan: list[dict[str, Any]] = []
    for variant in selected_variants(args):
        config = resolved[variant]
        temporal_config = config["p3_ia_temporal"]
        run_id = variant_run_id(run_base, variant, args.id_suffix)
        command = public_command(
            variant=variant,
            run_id=f"{run_base}__{variant}",
            device=args.device,
            output_root=args.output_root,
            model_config_path=None,
            resume=args.resume,
            overwrite=args.overwrite,
            id_suffix=args.id_suffix,
            smoke=args.smoke,
        )
        plan.append(
            {
                "variant": variant,
                "base_variant": BASE_VARIANT,
                "pfd_mode": config["pfd_mode"],
                "selection_mode": temporal_config["selection_mode"],
                "temporal_encoder_mode": temporal_config["temporal_encoder_mode"],
                "selected_candidates": list(temporal_config["selected_candidates"]),
                "effective_candidate_count": len(temporal_config["selected_candidates"]),
                "candidate_bank_count": 26,
                "execution_mode": "full_spatiotemporal",
                "relation_resource": dict(config["relation_resource"]),
                "spatial_edge_chunk_size": config["spatial_edge_chunk_size"],
                "architecture_axes": {
                    field: config[field]
                    for field in (
                        "spatial_query_mode",
                        "propagation_encoder_mode",
                        "turbine_embedding_mode",
                        "bias_scaling_mode",
                    )
                },
                "resolved_model_config_source": (
                    f"{SUITE_PATH} -> frozen R2 -> pfd3_ia_temporal({variant})"
                ),
                "temporary_model_yaml": "generated only during execution",
                "public_experiment_config": str(PUBLIC_CONFIG_PATH),
                "node_shared_chunk_size": "not applicable",
                "planned_run_identity": run_id,
                "planned_public_command": command,
                "planned_command": command,
            }
        )
    return plan


def _base_runtime_document() -> dict[str, Any]:
    suite = load_p3_ia11_suite(SUITE_PATH)
    base_suite_path = ROOT / suite["base"]["suite_file"]
    base_suite = load_r0_r7_suite(base_suite_path)
    model_file = ROOT / base_suite["base_model_config"]["file"]
    document = load_model_config_document(model_file)
    return dict(document["runtime"])


def resolved_model_document(
    variant: str,
    resolved: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if variant not in VARIANT_IDS:
        raise ValueError(f"unsupported RA-DS-PFD IA-1.1 variant: {variant}")
    return {"runtime": dict(runtime), "model": dict(resolved[variant])}


def execute_plan(args: argparse.Namespace, plan: list[dict[str, Any]]) -> int:
    resolved = resolve_p3_ia11_variants(SUITE_PATH, project_root=ROOT)
    runtime = _base_runtime_document()
    temporary_root = Path(tempfile.mkdtemp(prefix="ra_ds_pfd_p3_ia11_"))
    try:
        for item in plan:
            variant = str(item["variant"])
            model_path = temporary_root / f"{variant}.yaml"
            try:
                model_path.write_text(
                    yaml.safe_dump(
                        resolved_model_document(variant, resolved, runtime),
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )
                run_id = str(item["planned_run_identity"])
                if args.id_suffix:
                    run_id = run_id.removesuffix(f"__{args.id_suffix}")
                command = public_command(
                    variant=variant,
                    run_id=run_id,
                    device=args.device,
                    output_root=args.output_root,
                    model_config_path=model_path,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    id_suffix=args.id_suffix,
                    smoke=args.smoke,
                )
                print(
                    f"Launching {variant} with temporal mode "
                    f"{item['temporal_encoder_mode']}: {json.dumps(command, ensure_ascii=False)}",
                    flush=True,
                )
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode != 0:
                    print(
                        f"RA-DS-PFD IA-1.1 suite stopped after {variant} "
                        f"with exit code {completed.returncode}",
                        file=sys.stderr,
                    )
                    return int(completed.returncode or 1)
            finally:
                model_path.unlink(missing_ok=True)
        return 0
    finally:
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
                "suite": "ra_ds_pfd_p3_ia11",
                "base_variant": BASE_VARIANT,
                "pfd_mode": "pfd3_ia_temporal",
                "variants": plan,
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
