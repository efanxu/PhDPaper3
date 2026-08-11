"""Launch frozen RA-DS-PFD R0-R7 variants through the public training CLI."""

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

from _bootstrap import ROOT  # noqa: F401  (bootstraps src/ on direct execution)
from models.ra_ds_pfd_crossformer.r0_r7_suite import (
    DEFAULT_SUITE_PATH,
    VARIANT_IDS,
    load_r0_r7_suite,
    resolve_r0_r7_variants,
)
from runtime.config import load_experiment_config, load_model_config_document
from runtime.paths import effective_run_id, validate_run_id


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
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            raise argparse.ArgumentTypeError(f"duplicate variant: {part}")
        seen.add(part)
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen RA-DS-PFD R0-R7 variants through the public "
            "orchestrator/worker/Trainer path."
        )
    )
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--variant", choices=VARIANT_IDS, help="run one exact variant")
    selectors.add_argument(
        "--variants",
        type=_variant_list,
        help="run a comma-separated, duplicate-free variant list",
    )
    selectors.add_argument("--all", action="store_true", help="run R0 through R7 in order")
    parser.add_argument(
        "--run-id",
        help="base run identity; each variant appends __R0 through __R7",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="forward the public train smoke profile unchanged",
    )
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
        value or datetime.now().strftime("ra-ds-pfd-r0-r7-%Y%m%d-%H%M%S")
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
    variants = selected_variants(args)
    resolved = resolve_r0_r7_variants(SUITE_PATH, project_root=ROOT)
    public_config = load_experiment_config(PUBLIC_CONFIG_PATH)
    run_base = base_run_id(args.run_id)
    plan: list[dict[str, Any]] = []
    for variant in variants:
        model_config = resolved[variant]
        spatial_disabled = bool(model_config["spatial_disabled"])
        run_id = variant_run_id(run_base, variant, args.id_suffix)
        plan.append(
            {
                "variant": variant,
                "architecture_axes": (
                    {axis: "not applicable (canonical P1)" for axis in ARCHITECTURE_AXES}
                    if spatial_disabled
                    else {axis: model_config[axis] for axis in ARCHITECTURE_AXES}
                ),
                "spatial_disabled": spatial_disabled,
                "execution_mode": (
                    "node_shared_microbatch" if spatial_disabled else "full_spatiotemporal"
                ),
                "node_shared_chunk_size": (
                    int(public_config.runtime["node_shared_chunk_size"])
                    if spatial_disabled
                    else "not applicable"
                ),
                "spatial_edge_chunk_size": (
                    "not applicable" if spatial_disabled else model_config["spatial_edge_chunk_size"]
                ),
                "resolved_model_config_source": (
                    f"{SUITE_PATH} -> resolve_r0_r7_variant({variant})"
                ),
                "temporary_model_yaml": "generated only during execution",
                "public_experiment_config": str(PUBLIC_CONFIG_PATH),
                "planned_run_identity": run_id,
                "planned_command": public_command(
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
    return plan


def _base_runtime_document() -> dict[str, Any]:
    suite = load_r0_r7_suite(SUITE_PATH)
    relative = Path(suite["base_model_config"]["file"])
    document = load_model_config_document(ROOT / relative)
    return dict(document["runtime"])


def resolved_model_document(
    variant: str,
    resolved: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Wrap the resolver output in the existing model YAML document shape."""

    if variant not in VARIANT_IDS:
        raise ValueError(f"unsupported RA-DS-PFD R0-R7 variant: {variant}")
    return {"runtime": dict(runtime), "model": dict(resolved[variant])}


def execute_plan(args: argparse.Namespace, plan: list[dict[str, Any]]) -> int:
    resolved = resolve_r0_r7_variants(SUITE_PATH, project_root=ROOT)
    runtime = _base_runtime_document()
    temporary_root = Path(tempfile.mkdtemp(prefix="ra_ds_pfd_r0_r7_"))
    try:
        for item in plan:
            variant = str(item["variant"])
            model_path = temporary_root / f"{variant}.yaml"
            try:
                document = resolved_model_document(variant, resolved, runtime)
                model_path.write_text(
                    yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                command = public_command(
                    variant=variant,
                    run_id=str(item["planned_run_identity"]).removesuffix(
                        f"__{args.id_suffix}" if args.id_suffix else ""
                    ),
                    device=args.device,
                    output_root=args.output_root,
                    model_config_path=model_path,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    id_suffix=args.id_suffix,
                    smoke=args.smoke,
                )
                print(
                    f"Launching {variant}: {json.dumps(command, ensure_ascii=False)}",
                    flush=True,
                )
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode != 0:
                    print(
                        f"RA-DS-PFD R0-R7 suite stopped after {variant} "
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
            {"dry_run": bool(args.dry_run), "smoke": bool(args.smoke), "variants": plan},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    return execute_plan(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
