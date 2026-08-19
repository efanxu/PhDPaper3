"""Launch P3-B2 cardinality arms through the existing public training CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence, TextIO

import yaml

from _bootstrap import ROOT  # noqa: F401
from models.ra_ds_pfd_crossformer.p3_b2_suite import (
    CANDIDATE_COUNT,
    DEFAULT_SUITE_PATH,
    K_GRID,
    MODEL_NAME,
    VARIANT_IDS,
    aggregate_p3_b2_k_selection,
    load_p3_b2_suite,
    p3_b2_summary_path,
    resolve_p3_b2_variants,
    write_p3_b2_k_selection,
)
from models.ra_ds_pfd_crossformer.p3_selection import write_p3_selection_best
from models.ra_ds_pfd_crossformer.p3_suite import load_p3_suite
from models.ra_ds_pfd_crossformer.p3_selector import SELECTOR_TYPE
from runtime.config import load_experiment_config, load_model_config_document
from runtime.paths import effective_run_id, resolve_output_root, run_directory, validate_run_id


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
            "Run P3-B2 propagation-cardinality arms through the public "
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
    return validate_run_id(value or datetime.now().strftime("ra-ds-pfd-p3-b2-%Y%m%d-%H%M%S"))


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
        "architecture_axes": {
            field: resolved[field] for field in ARCHITECTURE_AXES
        },
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
    suite = load_p3_b2_suite(SUITE_PATH)
    resolved_variants = resolve_p3_b2_variants(SUITE_PATH, project_root=ROOT)
    public_config = load_experiment_config(PUBLIC_CONFIG_PATH)
    selection = public_config.evaluation["checkpoint_selection"]
    run_base = base_run_id(args.run_id)
    suite_run_id = effective_run_id(run_base, args.id_suffix)
    plan: list[dict[str, Any]] = []
    for variant in selected_variants(args):
        resolved = resolved_variants[variant]
        p3_config = dict(resolved["p3"])
        candidate_names = [
            f"{feature}.{transform}"
            for feature in p3_config["candidate_features"]
            for transform in p3_config["candidate_transforms"]
        ]
        if len(candidate_names) != CANDIDATE_COUNT:
            raise ValueError(
                f"P3-B2 {variant} resolved candidate_count={len(candidate_names)}, "
                f"expected {CANDIDATE_COUNT}"
            )
        top_k = int(p3_config["top_k"])
        plan.append(
            {
                "suite": suite["suite"],
                "variant": variant,
                "base": "canonical P3 / frozen R2",
                "base_suite": suite["base"]["suite_file"],
                "pfd_mode": resolved["pfd_mode"],
                "selector_type": SELECTOR_TYPE,
                "top_k": top_k,
                "k": top_k,
                "candidate_features": list(p3_config["candidate_features"]),
                "candidate_transforms": list(p3_config["candidate_transforms"]),
                "candidate_count": len(candidate_names),
                "candidate_names": candidate_names,
                "unique_experiment_variable": "p3.top_k",
                "resolved_model_architecture_identity": _architecture_identity(resolved),
                "resolved_model_config_source": (
                    f"{SUITE_PATH} -> canonical P3 -> frozen R2 -> {variant}"
                ),
                "temporary_model_yaml": "generated only during execution",
                "public_experiment_config": str(PUBLIC_CONFIG_PATH),
                "selection_metric": selection["metric"],
                "selection_split": selection["split"],
                "lower_is_better": selection["mode"] == "min",
                "node_shared_chunk_size": "not applicable",
                "planned_run_identity": variant_run_id(run_base, variant, args.id_suffix),
                "suite_run_id": suite_run_id,
                "dry_run_summary": f"{variant} K={top_k} M={len(candidate_names)}",
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
    return plan


def _base_runtime_document() -> dict[str, Any]:
    """Resolve B2 -> canonical P3 -> R0-R7 -> model YAML runtime."""

    b2_suite = load_p3_b2_suite(SUITE_PATH)
    canonical_suite = load_p3_suite(ROOT / b2_suite["base"]["suite_file"])
    base_suite_path = ROOT / canonical_suite["base"]["suite_file"]
    base_suite = yaml.safe_load(base_suite_path.read_text(encoding="utf-8"))
    model_file = ROOT / base_suite["base_model_config"]["file"]
    document = load_model_config_document(model_file)
    return dict(document["runtime"])


def resolved_model_document(
    resolved: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Wrap one resolved B2 model mapping in the public model YAML shape."""

    return {"runtime": dict(runtime), "model": dict(resolved)}


def print_p3_b2_selection_report(
    variant: str,
    artifact: dict[str, Any],
    *,
    stream: TextIO | None = None,
) -> None:
    """Print the compact human-readable propagation readout for one arm."""

    output = stream or sys.stdout
    selected = sorted(
        (
            item
            for item in artifact["propagation_feature_scores"]
            if item["selected"]
        ),
        key=lambda item: int(item["rank"]),
    )
    ranking = sorted(
        artifact["propagation_feature_scores"],
        key=lambda item: int(item["rank"]),
    )[: min(10, int(artifact["candidate_count"]))]
    base_scores = sorted(
        artifact["base_variable_scores"],
        key=lambda item: int(item["rank"]),
    )[:10]
    operator_scores = artifact["operator_scores"]

    print("=" * 50, file=output)
    print(f"P3-B2 {variant}", file=output)
    print(f"K = {artifact['top_k']}", file=output)
    print(f"checkpoint = {artifact['checkpoint_source']}", file=output)
    print(f"best_epoch = {artifact['best_epoch']}", file=output)
    print("=" * 50, file=output)
    print("Selected propagation features:", file=output)
    for item in selected:
        print(
            f"{item['rank']}. {item['candidate_name']} "
            f"score={float(item['score']):.6f} rank={item['rank']}",
            file=output,
        )
    print("Top propagation ranking:", file=output)
    for item in ranking:
        print(
            f"{item['rank']}. {item['candidate_name']} "
            f"score={float(item['score']):.6f} rank={item['rank']}",
            file=output,
        )
    print("Base-variable scores:", file=output)
    for item in base_scores:
        print(
            f"{item['rank']}. {item['base_feature']} "
            f"score={float(item['score']):.6f} rank={item['rank']}",
            file=output,
        )
    print(
        f"Operator scores: level = {float(operator_scores['level_score']):.6f}; "
        f"diff1 = {float(operator_scores['diff1_score']):.6f}",
        file=output,
    )

    # These are normalized mixture weights within each run.  K-selection must
    # use validation performance, not compare absolute scores across K arms.


def execute_plan(args: argparse.Namespace, plan: list[dict[str, Any]]) -> int:
    resolved_variants = resolve_p3_b2_variants(SUITE_PATH, project_root=ROOT)
    runtime = _base_runtime_document()
    output_root = resolve_output_root(ROOT, args.output_root)
    run_directories: dict[str, Path] = {}
    temporary_root = Path(tempfile.mkdtemp(prefix="ra_ds_pfd_p3_b2_"))
    try:
        for item in plan:
            variant = str(item["variant"])
            model_path = temporary_root / f"{variant}.yaml"
            try:
                document = resolved_model_document(resolved_variants[variant], runtime)
                model_path.write_text(
                    yaml.safe_dump(
                        document,
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )
                # Validate the exact temporary document before invoking the
                # public CLI; this remains a CPU-only config boundary.
                load_model_config_document(model_path)
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
                print(
                    f"Launching {variant}: {json.dumps(command, ensure_ascii=False)}",
                    flush=True,
                )
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode != 0:
                    print(
                        f"P3-B2 suite stopped after {variant} with exit code "
                        f"{completed.returncode}",
                        file=sys.stderr,
                    )
                    return int(completed.returncode or 1)
            finally:
                model_path.unlink(missing_ok=True)

            result_dir = run_directory(
                ROOT,
                output_root,
                MODEL_NAME,
                str(item["planned_run_identity"]),
            )
            artifact = write_p3_selection_best(
                result_dir,
                variant=variant,
                project_root=ROOT,
            )
            run_directories[variant] = result_dir
            print_p3_b2_selection_report(variant, artifact)

        # Smoke is an execution/readout gate only.  A formal K summary is
        # emitted only after the complete non-smoke grid is present, and it
        # lives beside the provisional-best (or deterministic ambiguous)
        # arm so that it cannot be shared by unrelated suite runs.
        if not args.smoke and set(run_directories) == set(VARIANT_IDS):
            first = plan[0]
            summary = aggregate_p3_b2_k_selection(
                run_directories,
                selection_metric=str(first["selection_metric"]),
                lower_is_better=bool(first["lower_is_better"]),
                strict=True,
                suite_run_id=str(first["suite_run_id"]),
            )
            host_variant = summary.get("provisional_best_variant")
            if host_variant is None:
                ambiguous = summary.get("ambiguous_variants")
                if not isinstance(ambiguous, list) or not ambiguous:
                    raise ValueError(
                        "P3-B2 complete grid produced no summary host variant"
                    )
                host_variant = ambiguous[0]
            if not isinstance(host_variant, str) or host_variant not in run_directories:
                raise ValueError("P3-B2 summary host variant is not a completed arm")
            write_p3_b2_k_selection(
                p3_b2_summary_path(run_directories[host_variant]),
                run_directories,
                selection_metric=str(first["selection_metric"]),
                lower_is_better=bool(first["lower_is_better"]),
                strict=True,
                suite_run_id=str(first["suite_run_id"]),
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
                "operator_basis": plan[0]["candidate_transforms"],
                "candidate_count": CANDIDATE_COUNT,
                "k_grid": list(K_GRID),
                "variants": [item["variant"] for item in plan],
                "dry_run_summary": [item["dry_run_summary"] for item in plan],
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
