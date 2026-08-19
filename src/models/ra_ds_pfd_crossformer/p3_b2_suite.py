"""Strict P3-B2 resolver and validation-only K-selection readout.

P3-B2 is deliberately narrower than P3-B1: the canonical Level+Diff1
candidate basis is frozen and only ``p3.top_k`` may vary.  The summary helper
at the end of this module reads completed run artifacts; it never reads test
metrics and never changes the canonical P3 configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import yaml

from engine.checkpoint import read_checkpoint_manifest
from runtime.run_info import write_json

from .model import _validate_config
from .p3_feature_bank import P3_BASE_FEATURES, validate_p3_model_config
from .p3_suite import resolve_p3_model_config


DEFAULT_SUITE_PATH = Path("configs/experiments/ra_ds_pfd_p3_b2.yaml")
FROZEN_BASE_SUITE_PATH = "configs/experiments/ra_ds_pfd_p3.yaml"
MODEL_NAME = "ra_ds_pfd_crossformer"
SUITE_NAME = "ra_ds_pfd_p3_b2"
VARIANT_IDS = ("B2_K1", "B2_K2", "B2_K3", "B2_K4", "B2_K6", "B2_K8")
B2_VARIANT_IDS = VARIANT_IDS
K_GRID = (1, 2, 3, 4, 6, 8)
P3_B2_K_GRID = K_GRID
FROZEN_OPERATOR_BASIS = ("level", "diff1")
CANDIDATE_COUNT = len(P3_BASE_FEATURES) * len(FROZEN_OPERATOR_BASIS)
SUMMARY_ARTIFACT_NAME = "p3_b2_k_selection.json"
B2_SUMMARY_ARTIFACT_NAME = SUMMARY_ARTIFACT_NAME
SELECTION_METRIC_DEFAULT = "SDWPF Official Score"
LOWER_IS_BETTER_DEFAULT = True

P3_B2_SUITE_FIELDS = frozenset({"suite", "model", "base", "variants"})
P3_B2_BASE_FIELDS = frozenset({"suite_file"})
P3_B2_VARIANT_FIELDS = frozenset({"top_k"})


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.Node,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RA-DS-PFD P3-B2 suite {field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"RA-DS-PFD P3-B2 suite {field} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    field: str,
) -> None:
    unexpected = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unexpected:
        raise ValueError(
            f"RA-DS-PFD P3-B2 suite {field} has unsupported field: {unexpected[0]}"
        )
    if missing:
        raise ValueError(
            f"RA-DS-PFD P3-B2 suite {field} is missing field: {missing[0]}"
        )


def _relative_file(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"RA-DS-PFD P3-B2 suite {field} must be a non-empty project-relative path"
        )
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(
            f"RA-DS-PFD P3-B2 suite {field} must stay within the project root"
        )
    return value


def _validate_top_k(value: Any, *, variant_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"RA-DS-PFD P3-B2 {variant_id}.top_k must be an integer")
    if not 1 <= value <= CANDIDATE_COUNT:
        raise ValueError(
            f"RA-DS-PFD P3-B2 {variant_id}.top_k must satisfy 1 <= K <= {CANDIDATE_COUNT}"
        )
    return int(value)


def _validate_definition(value: Any) -> dict[str, Any]:
    suite = dict(_mapping(value, field="root"))
    _exact_keys(suite, P3_B2_SUITE_FIELDS, field="root")
    if suite["suite"] != SUITE_NAME:
        raise ValueError("RA-DS-PFD P3-B2 suite has the wrong suite name")
    if suite["model"] != MODEL_NAME:
        raise ValueError("RA-DS-PFD P3-B2 suite must use ra_ds_pfd_crossformer")

    base = _mapping(suite["base"], field="base")
    _exact_keys(base, P3_B2_BASE_FIELDS, field="base")
    suite_file = _relative_file(base["suite_file"], field="base.suite_file")
    if suite_file.replace("\\", "/") != FROZEN_BASE_SUITE_PATH:
        raise ValueError(
            "RA-DS-PFD P3-B2 suite base.suite_file must be the canonical P3 suite"
        )

    variants = _mapping(suite["variants"], field="variants")
    if set(variants) != set(VARIANT_IDS):
        missing = sorted(set(VARIANT_IDS) - set(variants))
        unexpected = sorted(set(variants) - set(VARIANT_IDS))
        if missing:
            raise ValueError(f"RA-DS-PFD P3-B2 suite is missing variant: {missing[0]}")
        raise ValueError(
            f"RA-DS-PFD P3-B2 suite has unsupported variant: {unexpected[0]}"
        )
    for variant_id in VARIANT_IDS:
        variant = _mapping(variants[variant_id], field=f"variants.{variant_id}")
        _exact_keys(
            variant,
            P3_B2_VARIANT_FIELDS,
            field=f"variants.{variant_id}",
        )
        _validate_top_k(variant["top_k"], variant_id=variant_id)
    return deepcopy(suite)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RA-DS-PFD P3-B2 suite file does not exist: {path}")
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid RA-DS-PFD P3-B2 suite YAML: {path}: {exc}") from exc
    return _validate_definition(loaded)


def load_p3_b2_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    """Load and validate the machine-readable P3-B2 arm definition."""

    return _load_yaml(Path(path).resolve())


def _suite_and_root(
    suite_or_path: Mapping[str, Any] | str | Path,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(suite_or_path, (str, Path)):
        suite_path = Path(suite_or_path).resolve()
        suite = load_p3_b2_suite(suite_path)
        if (
            project_root is None
            and suite_path.parent.name == "experiments"
            and suite_path.parent.parent.name == "configs"
        ):
            root = suite_path.parents[2]
        else:
            root = Path(project_root or Path.cwd()).resolve()
    else:
        suite = _validate_definition(suite_or_path)
        root = Path(project_root or Path.cwd()).resolve()
    return suite, root


def _resolve_project_file(value: str, *, project_root: Path, field: str) -> Path:
    relative = Path(_relative_file(value, field=field))
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"RA-DS-PFD P3-B2 suite {field} resolves outside the project root"
        ) from exc
    return resolved


def _difference_paths(
    left: Any,
    right: Any,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add((*prefix, str(key)))
            else:
                differences.update(
                    _difference_paths(left[key], right[key], (*prefix, str(key)))
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix}
        differences: set[tuple[str, ...]] = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.update(
                _difference_paths(left_item, right_item, (*prefix, str(index)))
            )
        return differences
    return set() if left == right else {prefix}


def _validate_isolation(
    canonical: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    variant_id: str,
) -> None:
    differences = _difference_paths(canonical, resolved)
    allowed = {("p3", "top_k")}
    unexpected = differences - allowed
    if unexpected:
        field = ".".join(sorted(unexpected)[0])
        raise ValueError(
            f"RA-DS-PFD P3-B2 {variant_id} changed a frozen field: {field}"
        )


def _validate_resolved_matrix(
    canonical: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(resolved) != set(VARIANT_IDS):
        raise ValueError("RA-DS-PFD P3-B2 resolver did not produce the exact K grid")

    for variant_id in VARIANT_IDS:
        p3 = resolved[variant_id].get("p3")
        if not isinstance(p3, Mapping):
            raise ValueError(f"RA-DS-PFD P3-B2 {variant_id} has no p3 mapping")
        if list(p3["candidate_features"]) != list(P3_BASE_FEATURES):
            raise ValueError(
                f"RA-DS-PFD P3-B2 {variant_id} changed the frozen candidate features"
            )
        if tuple(p3["candidate_transforms"]) != FROZEN_OPERATOR_BASIS:
            raise ValueError(
                f"RA-DS-PFD P3-B2 {variant_id} changed the frozen Level+Diff1 operator basis"
            )
        if len(p3["candidate_features"]) * len(p3["candidate_transforms"]) != CANDIDATE_COUNT:
            raise ValueError(
                f"RA-DS-PFD P3-B2 {variant_id} must resolve candidate_count={CANDIDATE_COUNT}"
            )
        if p3["top_k"] != K_GRID[VARIANT_IDS.index(variant_id)]:
            raise ValueError(
                f"RA-DS-PFD P3-B2 {variant_id} resolved to the wrong top_k"
            )
        _validate_isolation(canonical, resolved[variant_id], variant_id=variant_id)

    baseline = resolved["B2_K2"]
    for variant_id in VARIANT_IDS:
        current = resolved[variant_id]
        for field in ("candidate_features", "candidate_transforms", "selector_temperature", "selector_bisection_iterations"):
            if current["p3"][field] != baseline["p3"][field]:
                raise ValueError(
                    f"RA-DS-PFD P3-B2 {variant_id} does not preserve p3.{field}"
                )
        if _difference_paths(baseline, current) - {("p3", "top_k")}:
            raise ValueError(
                f"RA-DS-PFD P3-B2 {variant_id} does not preserve the R2 architecture"
            )


def resolve_p3_b2_variants(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve all B2 arms from canonical P3 with top_k-only isolation."""

    suite, root = _suite_and_root(suite_or_path, project_root)
    base = _mapping(suite["base"], field="base")
    canonical_path = _resolve_project_file(
        base["suite_file"],
        project_root=root,
        field="base.suite_file",
    )
    canonical = deepcopy(resolve_p3_model_config(canonical_path, project_root=root))
    canonical_p3 = canonical.get("p3")
    if not isinstance(canonical_p3, Mapping):
        raise ValueError("RA-DS-PFD P3-B2 canonical P3 has no p3 mapping")
    if tuple(canonical_p3.get("candidate_transforms", ())) != FROZEN_OPERATOR_BASIS:
        raise ValueError(
            "P3-B2 currently requires the frozen Level+Diff1 operator basis."
        )
    if (
        list(canonical_p3.get("candidate_features", ())) != list(P3_BASE_FEATURES)
        or len(canonical_p3["candidate_features"])
        * len(canonical_p3["candidate_transforms"])
        != CANDIDATE_COUNT
    ):
        raise ValueError(
            f"RA-DS-PFD P3-B2 canonical P3 must resolve candidate_count={CANDIDATE_COUNT}"
        )

    variants = _mapping(suite["variants"], field="variants")
    resolved: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        config = deepcopy(canonical)
        config["p3"]["top_k"] = _validate_top_k(
            variants[variant_id]["top_k"],
            variant_id=variant_id,
        )
        validate_p3_model_config(config["p3"])
        _validate_config(config)
        _validate_isolation(canonical, config, variant_id=variant_id)
        resolved[variant_id] = config
    _validate_resolved_matrix(canonical, resolved)
    return resolved


def resolve_p3_b2_variant(
    suite_or_path: Mapping[str, Any] | str | Path,
    variant_id: str,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one exact P3-B2 cardinality arm."""

    if variant_id not in VARIANT_IDS:
        raise ValueError(f"unsupported RA-DS-PFD P3-B2 variant: {variant_id}")
    return deepcopy(
        resolve_p3_b2_variants(suite_or_path, project_root=project_root)[variant_id]
    )


def validate_p3_b2_suite(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Fail closed unless the frozen Level+Diff1 B2 matrix resolves cleanly."""

    resolve_p3_b2_variants(suite_or_path, project_root=project_root)


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"could not read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return dict(value)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"P3-B2 {field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"P3-B2 {field} must be finite")
    return number


def _selected_features(artifact: Mapping[str, Any], *, variant_id: str) -> list[dict[str, Any]]:
    if artifact.get("variant") not in {None, variant_id}:
        raise ValueError(
            f"P3-B2 {variant_id} selection artifact has the wrong variant"
        )
    if artifact.get("checkpoint_source") != "best.pt":
        raise ValueError(
            f"P3-B2 {variant_id} selection artifact must use checkpoint_source=best.pt"
        )
    if artifact.get("candidate_count") != CANDIDATE_COUNT:
        raise ValueError(
            f"P3-B2 {variant_id} selection artifact must contain {CANDIDATE_COUNT} candidates"
        )
    if list(artifact.get("candidate_features", ())) != list(P3_BASE_FEATURES):
        raise ValueError(f"P3-B2 {variant_id} selection artifact changed candidate features")
    if tuple(artifact.get("candidate_transforms", ())) != FROZEN_OPERATOR_BASIS:
        raise ValueError(
            f"P3-B2 {variant_id} selection artifact changed the frozen operator basis"
        )
    top_k = _validate_top_k(artifact.get("top_k"), variant_id=variant_id)
    rows = artifact.get("propagation_feature_scores")
    if not isinstance(rows, list) or len(rows) != CANDIDATE_COUNT:
        raise ValueError(f"P3-B2 {variant_id} selection artifact has an invalid ranking")
    selected: list[dict[str, Any]] = []
    ranks: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"P3-B2 {variant_id} selection ranking entries must be mappings")
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError(f"P3-B2 {variant_id} selection ranks must be integers")
        ranks.append(int(rank))
        if row.get("selected"):
            name = row.get("candidate_name")
            if not isinstance(name, str):
                raise ValueError(
                    f"P3-B2 {variant_id} selected candidate must have a name"
                )
            selected.append(
                {
                    "candidate_name": name,
                    "score": _finite_number(
                        row.get("score"),
                        field=f"{variant_id}.selected_feature.score",
                    ),
                    "rank": int(rank),
                }
            )
    if sorted(ranks) != list(range(1, CANDIDATE_COUNT + 1)):
        raise ValueError(f"P3-B2 {variant_id} selection ranks are not 1..26")
    if len(selected) != top_k:
        raise ValueError(
            f"P3-B2 {variant_id} selected count {len(selected)} does not equal K={top_k}"
        )
    return sorted(selected, key=lambda item: int(item["rank"]))


def _validation_monitor(
    run_path: Path,
    *,
    selection_metric: str,
    lower_is_better: bool,
) -> tuple[float, str]:
    validation = _read_json_mapping(
        run_path / "metrics_validation.json",
        label="metrics_validation.json",
    )
    value = _finite_number(validation.get("monitor"), field="validation monitor")
    source = "metrics_validation.json:monitor"

    # The public train path reloads best.pt before writing metrics_validation.
    # When the checkpoint is available, validate that provenance and the
    # public checkpoint-selection semantics agree; never inspect test files.
    best_path = run_path / "best.pt"
    if best_path.is_file():
        manifest = read_checkpoint_manifest(best_path)
        checkpoint_value = manifest.get("monitor")
        if checkpoint_value is not None:
            checkpoint_number = _finite_number(
                checkpoint_value,
                field="best.pt validation monitor",
            )
            if not math.isclose(value, checkpoint_number, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(
                    "P3-B2 validation monitor does not match best.pt checkpoint monitor"
                )
            source = "best.pt:manifest.monitor"
        monitor_name = manifest.get("monitor_name")
        if monitor_name is not None and monitor_name != selection_metric:
            raise ValueError(
                "P3-B2 best.pt monitor_name does not match the public validation metric"
            )
        checkpoint_selection = manifest.get("checkpoint_selection")
        if isinstance(checkpoint_selection, Mapping):
            if checkpoint_selection.get("split") not in {"validation", "val"}:
                raise ValueError("P3-B2 checkpoint selection must use validation")
            if checkpoint_selection.get("metric") != selection_metric:
                raise ValueError(
                    "P3-B2 checkpoint selection metric does not match the public metric"
                )
            expected_mode = "min" if lower_is_better else "max"
            if checkpoint_selection.get("mode") != expected_mode:
                raise ValueError(
                    "P3-B2 checkpoint selection mode does not match the public metric direction"
                )
    return value, source


def _incomplete_summary(
    *,
    runs: list[dict[str, Any]],
    missing_variants: list[str],
    selection_metric: str,
    lower_is_better: bool,
    suite_run_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": SUITE_NAME,
        "suite_run_id": suite_run_id,
        "operator_basis": list(FROZEN_OPERATOR_BASIS),
        "candidate_count": CANDIDATE_COUNT,
        "selection_metric": selection_metric,
        "selection_uses_test": False,
        "lower_is_better": bool(lower_is_better),
        "validation_selection_mode": "min" if lower_is_better else "max",
        "selection_status": "INCOMPLETE",
        "missing_variants": list(missing_variants),
        "runs": runs,
        "provisional_best_k": None,
        "provisional_best_variant": None,
        "runner_up_k": None,
        "runner_up_variant": None,
        "validation_gap": None,
        "selected_k": None,
        "selected_variant": None,
    }


def _resolved_suite_run_id(
    suite_run_id: str | None,
    entries: list[dict[str, Any]],
) -> str | None:
    if suite_run_id is not None:
        if not isinstance(suite_run_id, str) or not suite_run_id:
            raise ValueError("P3-B2 suite_run_id must be a non-empty string")
        return suite_run_id

    prefixes: list[str] = []
    for entry in entries:
        run_id = entry.get("run_id")
        if not isinstance(run_id, str):
            return None
        variant_marker = next(
            (
                f"__{variant_id}"
                for variant_id in VARIANT_IDS
                if f"__{variant_id}" in run_id
            ),
            None,
        )
        if variant_marker is None:
            return None
        base, suffix = run_id.split(variant_marker, 1)
        prefixes.append(f"{base}{suffix}")
    if prefixes and len(set(prefixes)) == 1 and prefixes[0]:
        return prefixes[0]
    return None


def aggregate_p3_b2_k_selection(
    run_directories: Mapping[str, str | Path],
    *,
    selection_metric: str = SELECTION_METRIC_DEFAULT,
    lower_is_better: bool = LOWER_IS_BETTER_DEFAULT,
    strict: bool = True,
    suite_run_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate B2 validation monitors without using test metrics.

    ``strict=True`` fails closed for a partial K grid.  ``strict=False``
    returns an explicit ``INCOMPLETE`` payload without declaring a winner.
    """

    if not isinstance(selection_metric, str) or not selection_metric:
        raise ValueError("P3-B2 selection_metric must be a non-empty string")
    expected_mode = "min" if lower_is_better else "max"
    unknown = sorted(set(run_directories) - set(VARIANT_IDS))
    if unknown:
        raise ValueError(f"P3-B2 summary received unsupported variant: {unknown[0]}")

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for variant_id in VARIANT_IDS:
        raw_path = run_directories.get(variant_id)
        if raw_path is None:
            missing.append(variant_id)
            continue
        run_path = Path(raw_path).resolve()
        if not run_path.is_dir():
            missing.append(variant_id)
            continue
        try:
            artifact = _read_json_mapping(
                run_path / "p3_selection_best.json",
                label="p3_selection_best.json",
            )
            selected = _selected_features(artifact, variant_id=variant_id)
            top_k = _validate_top_k(artifact.get("top_k"), variant_id=variant_id)
            expected_k = K_GRID[VARIANT_IDS.index(variant_id)]
            if top_k != expected_k:
                raise ValueError(
                    f"P3-B2 {variant_id} selection artifact has K={top_k}, expected {expected_k}"
                )
            validation_monitor, source = _validation_monitor(
                run_path,
                selection_metric=selection_metric,
                lower_is_better=lower_is_better,
            )
            best_epoch = artifact.get("best_epoch")
            if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 1:
                raise ValueError(f"P3-B2 {variant_id} has no valid best_epoch")
            run_id = artifact.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                run_id = run_path.name
            entries.append(
                {
                    "variant": variant_id,
                    "k": expected_k,
                    "run_id": run_id,
                    "validation_monitor": validation_monitor,
                    "validation_monitor_name": selection_metric,
                    "validation_monitor_source": source,
                    "validation_metrics_file": "metrics_validation.json",
                    "best_epoch": int(best_epoch),
                    "selected_features": selected,
                }
            )
        except (FileNotFoundError, OSError) as exc:
            if strict:
                raise ValueError(
                    f"P3-B2 {variant_id} is incomplete: {exc}"
                ) from exc
            missing.append(variant_id)

    if missing:
        if strict:
            raise ValueError(
                "P3-B2 K grid is incomplete; cannot select a winner: "
                + ", ".join(missing)
            )
        return _incomplete_summary(
            runs=entries,
            missing_variants=missing,
            selection_metric=selection_metric,
            lower_is_better=lower_is_better,
            suite_run_id=_resolved_suite_run_id(suite_run_id, entries),
        )

    entries.sort(key=lambda item: VARIANT_IDS.index(str(item["variant"])))
    resolved_suite_run_id = _resolved_suite_run_id(suite_run_id, entries)
    ordered = sorted(
        entries,
        key=lambda item: (
            float(item["validation_monitor"])
            if lower_is_better
            else -float(item["validation_monitor"]),
            int(item["k"]),
        ),
    )
    best = ordered[0]
    second = ordered[1]
    if float(best["validation_monitor"]) == float(second["validation_monitor"]):
        return {
            "schema_version": 1,
            "suite": SUITE_NAME,
            "suite_run_id": resolved_suite_run_id,
            "operator_basis": list(FROZEN_OPERATOR_BASIS),
            "candidate_count": CANDIDATE_COUNT,
            "selection_metric": selection_metric,
            "selection_uses_test": False,
            "lower_is_better": bool(lower_is_better),
            "validation_selection_mode": expected_mode,
            "selection_status": "AMBIGUOUS",
            "ambiguous_variants": [
                item["variant"]
                for item in ordered
                if float(item["validation_monitor"])
                == float(best["validation_monitor"])
            ],
            "ambiguous_reason": (
                "exact validation-monitor tie; confirmation seed is required"
            ),
            "runs": entries,
            "provisional_best_k": None,
            "provisional_best_variant": None,
            "runner_up_k": None,
            "runner_up_variant": None,
            "validation_gap": 0.0,
            "selected_k": None,
            "selected_variant": None,
        }
    return {
        "schema_version": 1,
        "suite": SUITE_NAME,
        "suite_run_id": resolved_suite_run_id,
        "operator_basis": list(FROZEN_OPERATOR_BASIS),
        "candidate_count": CANDIDATE_COUNT,
        "selection_metric": selection_metric,
        "selection_uses_test": False,
        "lower_is_better": bool(lower_is_better),
        "validation_selection_mode": expected_mode,
        "selection_status": "PROVISIONAL",
        "runs": entries,
        "provisional_best_k": int(best["k"]),
        "provisional_best_variant": str(best["variant"]),
        "runner_up_k": int(second["k"]),
        "runner_up_variant": str(second["variant"]),
        "validation_gap": abs(
            float(second["validation_monitor"])
            - float(best["validation_monitor"])
        ),
        "selected_k": None,
        "selected_variant": None,
    }


def p3_b2_summary_path(run_directory: str | Path) -> Path:
    """Return the summary location inside one concrete B2 run directory."""

    return Path(run_directory).resolve() / SUMMARY_ARTIFACT_NAME


def write_p3_b2_k_selection(
    summary_path: str | Path,
    run_directories: Mapping[str, str | Path],
    *,
    selection_metric: str = SELECTION_METRIC_DEFAULT,
    lower_is_better: bool = LOWER_IS_BETTER_DEFAULT,
    strict: bool = True,
    suite_run_id: str | None = None,
) -> dict[str, Any]:
    """Write the small B2 suite artifact and return its payload."""

    summary = aggregate_p3_b2_k_selection(
        run_directories,
        selection_metric=selection_metric,
        lower_is_better=lower_is_better,
        strict=strict,
        suite_run_id=suite_run_id,
    )
    write_json(Path(summary_path).resolve(), summary)
    return summary


# Short aliases follow the existing P3/B1 resolver vocabulary.
load_b2_suite = load_p3_b2_suite
resolve_b2_variants = resolve_p3_b2_variants
resolve_b2_variant = resolve_p3_b2_variant


__all__ = [
    "B2_SUMMARY_ARTIFACT_NAME",
    "B2_VARIANT_IDS",
    "CANDIDATE_COUNT",
    "DEFAULT_SUITE_PATH",
    "FROZEN_BASE_SUITE_PATH",
    "FROZEN_OPERATOR_BASIS",
    "K_GRID",
    "MODEL_NAME",
    "P3_B2_K_GRID",
    "SELECTION_METRIC_DEFAULT",
    "SUMMARY_ARTIFACT_NAME",
    "SUITE_NAME",
    "VARIANT_IDS",
    "aggregate_p3_b2_k_selection",
    "load_b2_suite",
    "load_p3_b2_suite",
    "p3_b2_summary_path",
    "resolve_b2_variant",
    "resolve_b2_variants",
    "resolve_p3_b2_variant",
    "resolve_p3_b2_variants",
    "validate_p3_b2_suite",
    "write_p3_b2_k_selection",
]
