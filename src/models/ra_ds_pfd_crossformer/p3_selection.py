"""Read-only P3 selection extraction from a completed best checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from engine.checkpoint import load_checkpoint
from models.base import DataInfoView, ForecastModel
from models.loader import build_model
from runtime.config import load_model_config_document
from runtime.run_info import write_json

from .p3_feature_bank import validate_p3_model_config


MODEL_NAME = "ra_ds_pfd_crossformer"
SELECTION_SCHEMA_VERSION = 1
SELECTION_ARTIFACT_NAME = "p3_selection_best.json"


def _read_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return dict(value)


def _data_info_from_run(run_directory: Path, *, project_root: Path) -> DataInfoView:
    document = _read_yaml_mapping(
        run_directory / "resolved_config.yaml",
        label="resolved_config.yaml",
    )
    resolved = document.get("resolved")
    if not isinstance(resolved, Mapping):
        raise ValueError("resolved_config.yaml has no resolved mapping")
    raw_info = resolved.get("data_info")
    if not isinstance(raw_info, Mapping):
        raise ValueError("resolved_config.yaml has no resolved.data_info mapping")
    data_info = dict(raw_info)
    data_info["project_root"] = str(project_root)
    if "graph_config" not in data_info and "graph" in data_info:
        data_info["graph_config"] = data_info["graph"]
    return DataInfoView.from_object(data_info)


def _candidate_names(features: Sequence[str], transforms: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{feature}.{transform}" for feature in features for transform in transforms)


def _finite_score(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"P3 selection report {field} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise FloatingPointError(f"P3 selection report {field} is non-finite")
    if score < 0.0:
        raise ValueError(f"P3 selection report {field} must be non-negative")
    return score


def _validate_and_enrich_report(
    report: Any,
    *,
    features: tuple[str, ...],
    transforms: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    expected_names = _candidate_names(features, transforms)
    if not isinstance(report, list) or len(report) != len(expected_names):
        raise ValueError(
            "P3 selection report candidate count does not match the resolved candidate bank"
        )
    expected_set = set(expected_names)
    seen: set[str] = set()
    enriched: list[dict[str, Any]] = []
    ranks: list[int] = []
    for item in report:
        if not isinstance(item, Mapping):
            raise ValueError("P3 selection report entries must be mappings")
        name = item.get("candidate_name")
        if not isinstance(name, str) or name not in expected_set or name in seen:
            raise ValueError("P3 selection report contains an unexpected or duplicate candidate")
        seen.add(name)
        try:
            base_feature, operator = name.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(f"invalid P3 candidate name: {name!r}") from exc
        if base_feature not in features or operator not in transforms:
            raise ValueError(f"P3 selection report candidate is outside the resolved bank: {name}")
        rank = item.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("P3 selection report ranks must be integers")
        ranks.append(int(rank))
        selected = item.get("selected")
        if not isinstance(selected, bool):
            raise ValueError("P3 selection report selected values must be booleans")
        enriched.append(
            {
                "candidate_name": name,
                "base_feature": base_feature,
                "operator": operator,
                "score": _finite_score(item.get("score"), field=f"{name}.score"),
                "rank": int(rank),
                "selected": selected,
            }
        )
    if seen != expected_set:
        raise ValueError("P3 selection report is missing a candidate")
    if sorted(ranks) != list(range(1, len(expected_names) + 1)):
        raise ValueError("P3 selection report ranks are not a unique 1-based ranking")
    selected_count = sum(bool(item["selected"]) for item in enriched)
    if selected_count != int(top_k):
        raise ValueError(
            f"P3 selection report selected count {selected_count} does not equal top_k {top_k}"
        )
    score_sum = sum(float(item["score"]) for item in enriched)
    if not math.isclose(score_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"P3 selection report scores must sum to 1, got {score_sum}")
    return enriched


def _summarize_scores(
    report: list[dict[str, Any]],
    *,
    features: tuple[str, ...],
    transforms: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_scores = {feature: 0.0 for feature in features}
    first_candidate_index = {feature: index for index, feature in enumerate(features)}
    operator_scores = {operator: 0.0 for operator in transforms}
    selected_by_operator = {operator: 0 for operator in transforms}
    for item in report:
        base_scores[item["base_feature"]] += float(item["score"])
        operator = item["operator"]
        operator_scores[operator] += float(item["score"])
        if item["selected"]:
            selected_by_operator[operator] += 1

    ranked_features = sorted(
        features,
        key=lambda feature: (-base_scores[feature], first_candidate_index[feature]),
    )
    base_report = [
        {
            "base_feature": feature,
            "score": float(base_scores[feature]),
            "rank": rank,
        }
        for rank, feature in enumerate(ranked_features, start=1)
    ]
    base_sum = sum(item["score"] for item in base_report)
    if not math.isclose(base_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"P3 base-variable scores must sum to 1, got {base_sum}")

    result: dict[str, Any] = {
        "level_score": float(operator_scores["level"]),
        "selected_level_count": int(selected_by_operator["level"]),
    }
    if "diff1" in transforms:
        result.update(
            {
                "diff1_score": float(operator_scores["diff1"]),
                "selected_diff1_count": int(selected_by_operator["diff1"]),
            }
        )
        operator_sum = sum(float(operator_scores[operator]) for operator in transforms)
        if not math.isclose(operator_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"P3 operator scores must sum to 1, got {operator_sum}")
    else:
        result.update(
            {
                "diff1_score": None,
                "selected_diff1_count": None,
                "diff1": {"not_applicable": True},
            }
        )
        if not math.isclose(result["level_score"], 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"P3 level-only operator score must equal 1, got {result['level_score']}"
            )
    return base_report, result


def extract_p3_selection_best(
    run_directory: str | Path,
    *,
    variant: str | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load exactly ``best.pt`` and return the P3 selection artifact payload."""

    run_path = Path(run_directory).resolve()
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]
    model_document = load_model_config_document(run_path / "model_config.yaml")
    model_config = dict(model_document["model"])
    if model_config.get("pfd_mode") != "pfd3_global_topk":
        raise ValueError("P3 selection extraction requires pfd_mode=pfd3_global_topk")
    p3_config = model_config.get("p3")
    if not isinstance(p3_config, Mapping):
        raise ValueError("P3 model config has no p3 mapping")
    validate_p3_model_config(p3_config)
    features = tuple(str(value) for value in p3_config["candidate_features"])
    transforms = tuple(str(value) for value in p3_config["candidate_transforms"])
    top_k = int(p3_config["top_k"])

    info = _data_info_from_run(run_path, project_root=root)
    model = build_model(MODEL_NAME, model_config, info)
    if not isinstance(model, ForecastModel):
        raise TypeError("P3 selection extraction built a non-ForecastModel")
    report_method = getattr(model, "propagation_selection_report", None)
    if not callable(report_method):
        report_method = getattr(model, "selection_report", None)
    if not callable(report_method):
        raise TypeError("P3 model does not expose a read-only selection report")

    best_path = run_path / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"P3 selection extraction requires best.pt: {best_path}")
    # This is intentionally the only checkpoint path used by the formal
    # discovery readout. In particular, last.pt is never consulted here.
    checkpoint_manifest = load_checkpoint(best_path, model, device="cpu")
    saved_model_config = checkpoint_manifest.get("model_config")
    if saved_model_config is None:
        saved_model_config = checkpoint_manifest.get("model_config_identity")
    if not isinstance(saved_model_config, Mapping):
        raise ValueError(
            "best.pt checkpoint manifest has no model_config; refusing to generate "
            "propagation feature selection artifact"
        )
    if dict(saved_model_config) != model_config:
        raise ValueError(
            "best.pt model_config does not match run/model_config.yaml; refusing to "
            "generate propagation feature selection artifact"
        )
    best_epoch = checkpoint_manifest.get("epoch")
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or best_epoch < 1:
        raise ValueError("best.pt checkpoint manifest must contain a positive epoch")
    model.eval()
    enriched = _validate_and_enrich_report(
        report_method(),
        features=features,
        transforms=transforms,
        top_k=top_k,
    )
    base_report, operator_report = _summarize_scores(
        enriched,
        features=features,
        transforms=transforms,
    )

    run_info = _read_yaml_mapping(run_path / "run_info.json", label="run_info.json")
    run_id = run_info.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = run_path.name
    artifact: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "run_id": run_id,
        "variant": variant,
        "checkpoint_source": "best.pt",
        "best_epoch": int(best_epoch),
        "top_k": top_k,
        "candidate_count": len(enriched),
        "candidate_features": list(features),
        "candidate_transforms": list(transforms),
        "propagation_feature_scores": enriched,
        "base_variable_scores": base_report,
        "operator_scores": operator_report,
        "checkpoint_provenance": {
            "path": "best.pt",
            "epoch": int(best_epoch),
            "is_last": checkpoint_manifest.get("is_last"),
            "state_dict_hash": checkpoint_manifest.get("state_dict_hash"),
        },
    }
    return artifact


def write_p3_selection_best(
    run_directory: str | Path,
    *,
    variant: str | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the best-checkpoint selection artifact beside the run files."""

    run_path = Path(run_directory).resolve()
    artifact = extract_p3_selection_best(
        run_path,
        variant=variant,
        project_root=project_root,
    )
    write_json(run_path / SELECTION_ARTIFACT_NAME, artifact)
    return artifact


__all__ = [
    "MODEL_NAME",
    "SELECTION_ARTIFACT_NAME",
    "SELECTION_SCHEMA_VERSION",
    "extract_p3_selection_best",
    "write_p3_selection_best",
]
