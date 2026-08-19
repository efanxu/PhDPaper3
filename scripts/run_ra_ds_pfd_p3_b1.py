"""Launch P3-B1 arms through the existing public training CLI."""

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
from models.ra_ds_pfd_crossformer.p3_b1_suite import (
    DEFAULT_SUITE_PATH,
    VARIANT_IDS,
    load_p3_b1_suite,
    resolve_p3_b1_variants,
)
from models.ra_ds_pfd_crossformer.p3_selection import write_p3_selection_best
from models.ra_ds_pfd_crossformer.p3_suite import load_p3_suite
from models.ra_ds_pfd_crossformer.p3_selector import SELECTOR_TYPE
from runtime.config import load_experiment_config, load_model_config_document
from runtime.paths import effective_run_id, resolve_output_root, run_directory, validate_run_id


MODEL_NAME = "ra_ds_pfd_crossformer"
PUBLIC_CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
SUITE_PATH = ROOT / DEFAULT_SUITE_PATH
ARCHITECTURE_AXES = (
    "spatial_query_mode",
    "propagation_encoder_mode",
    "turbine_embedding_mode",
    "bias_scaling_mode",
)


def _variant_list(value: str) -> tuple[str, ...]:
    parts = value.split(",")
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--variants must not contain an empty variant")
    unknown = [part for part in parts if part not in VARIANT_IDS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported variant: {unknown[0]}")
    if len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError("--variants must not contain duplicates")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run P3-B1 Default-K discovery arms through the public "
            "orchestrator/worker/Trainer path."
        )
    )
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--variant", choices=VARIANT_IDS)
    selectors.add_argument("--variants", type=_variant_list)
    selectors.add_argument("--all", action="store_true")
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
    return validate_run_id(value or datetime.now().strftime("ra-ds-pfd-p3-b1-%Y%m%d-%H%M%S"))


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


def _architecture_identity(resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "base": "canonical P3 / frozen R2",
        "pfd_mode": resolved["pfd_mode"],
        "execution_mode": "full_spatiotemporal",
        "architecture_axes": {field: resolved[field] for field in ARCHITECTURE_AXES},
        "d_model": resolved["d_model"],
        "n_heads": resolved["n_heads"],
        "d_ff": resolved["d_ff"],
        "e_layers": resolved["e_layers"],
        "dropout": resolved["dropout"],
        "factor": resolved["factor"],
        "seg_len": resolved["seg_len"],
        "win_size": resolved["win_size"],
        "spatial_edge_chunk_size": resolved["spatial_edge_chunk_size"],
        "relation_resource": dict(resolved["relation_resource"]),
    }


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite = load_p3_b1_suite(SUITE_PATH)
    resolved_variants = resolve_p3_b1_variants(SUITE_PATH, project_root=ROOT)
    public_config = load_experiment_config(PUBLIC_CONFIG_PATH)
    run_base = base_run_id(args.run_id)
    plan: list[dict[str, Any]] = []
    for variant in selected_variants(args):
        resolved = resolved_variants[variant]
        p3_config = dict(resolved["p3"])
        candidate_names = [
            f"{feature}.{transform}"
            for feature in p3_config["candidate_features"]
            for transform in p3_config["candidate_transforms"]
        ]
        plan.append(
            {
                "suite": suite["suite"],
                "variant": variant,
                "base": "canonical P3 / frozen R2",
                "base_suite": suite["base"]["suite_file"],
                "pfd_mode": resolved["pfd_mode"],
                "selector_type": SELECTOR_TYPE,
                "top_k": int(p3_config["top_k"]),
                "selector_temperature": float(p3_config["selector_temperature"]),
                "selector_bisection_iterations": int(
                    p3_config["selector_bisection_iterations"]
                ),
                "candidate_features": list(p3_config["candidate_features"]),
                "candidate_transforms": list(p3_config["candidate_transforms"]),
                "candidate_count": len(candidate_names),
                "candidate_names": candidate_names,
                "unique_experiment_variable": "p3.candidate_transforms",
                "resolved_model_architecture_identity": _architecture_identity(resolved),
                "resolved_model_config_source": (
                    f"{SUITE_PATH} -> canonical P3 -> frozen R2 -> {variant}"
                ),
                "temporary_model_yaml": "generated only during execution",
                "public_experiment_config": str(PUBLIC_CONFIG_PATH),
                "node_shared_chunk_size": "not applicable",
                "planned_run_identity": variant_run_id(run_base, variant, args.id_suffix),
                "planned_public_command": public_command(
                    variant=variant,
                    run_id=f"{run_base}__{variant}",
                    device=args.device,
                    output_root=args.output_root,
                    model_config_path=None,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    id_suffix=args.id_suffix,
                    smoke=args.smoke,
                ),
            }
        )
    # The public config is loaded above for the same preflight/config boundary
    # as the existing suite runners; B1 never overrides any of its fields.
    del public_config
    return plan


def _base_runtime_document() -> dict[str, Any]:
    b1_suite = load_p3_b1_suite(SUITE_PATH)
    canonical_suite = load_p3_suite(ROOT / b1_suite["base"]["suite_file"])
    model_file = ROOT / canonical_suite["base"]["suite_file"]
    # The canonical P3 suite intentionally inherits the runtime document from
    # the frozen R0-R7 base model YAML.
    document = load_model_config_document(model_file)
    return dict(document["runtime"])


def resolved_model_document(resolved: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return {"runtime": dict(runtime), "model": dict(resolved)}


def execute_plan(args: argparse.Namespace, plan: list[dict[str, Any]]) -> int:
    resolved_variants = resolve_p3_b1_variants(SUITE_PATH, project_root=ROOT)
    runtime = _base_runtime_document()
    temporary_root = Path(tempfile.mkdtemp(prefix="ra_ds_pfd_p3_b1_"))
    try:
        for item in plan:
            variant = str(item["variant"])
            model_path = temporary_root / f"{variant}.yaml"
            try:
                model_path.write_text(
                    yaml.safe_dump(
                        resolved_model_document(resolved_variants[variant], runtime),
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )
                run_id = str(item["planned_run_identity"]).removesuffix(
                    f"__{args.id_suffix}" if args.id_suffix else ""
                )
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
                print(f"Launching {variant}: {json.dumps(command, ensure_ascii=False)}", flush=True)
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode != 0:
                    print(
                        f"P3-B1 suite stopped after {variant} with exit code {completed.returncode}",
                        file=sys.stderr,
                    )
                    return int(completed.returncode or 1)
            finally:
                model_path.unlink(missing_ok=True)

            result_dir = run_directory(
                ROOT,
                resolve_output_root(ROOT, args.output_root),
                MODEL_NAME,
                str(item["planned_run_identity"]),
            )
            artifact = write_p3_selection_best(
                result_dir,
                variant=variant,
                project_root=ROOT,
            )
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "selection_artifact": str(result_dir / "p3_selection_best.json"),
                        "checkpoint_source": artifact["checkpoint_source"],
                        "best_epoch": artifact["best_epoch"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
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
                "suite": plan[0]["suite"],
                "base": plan[0]["base"],
                "variants": [item["variant"] for item in plan],
                "arms": plan,
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
