"""Small path helpers shared by CLI entry points."""

from __future__ import annotations

from pathlib import Path
import re
from datetime import datetime, timezone
from uuid import uuid4


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FORMAL_RESULT_ARTIFACTS = (
    "best.pt",
    "last.pt",
    "resolved_config.yaml",
    "run_info.json",
    "performance.json",
)


def validate_run_id(value: str, *, label: str = "run-id") -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' or '-'")
    return value


def effective_run_id(run_id: str | None, id_suffix: str | None) -> str:
    base = validate_run_id(run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    if id_suffix is None:
        return base
    suffix = validate_run_id(id_suffix, label="id-suffix")
    return f"{base}__{suffix}"


def project_root_from_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.parent.name == "configs":
        return path.parent.parent
    return Path.cwd().resolve()


def resolve_data_root(project_root: Path, data_root: str | Path) -> Path:
    candidate = Path(data_root)
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def resolve_output_root(project_root: Path, output_root: str | Path | None) -> Path:
    if output_root is None:
        return (project_root / "results").resolve()
    candidate = Path(output_root)
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def run_directory(
    project_root: Path,
    output_root: str | Path | None,
    model_name: str,
    run_id: str,
) -> Path:
    validate_run_id(model_name, label="model")
    validate_run_id(run_id)
    return resolve_output_root(project_root, output_root) / model_name / run_id


def formal_result_exists(path: Path) -> bool:
    return any((path / name).exists() for name in FORMAL_RESULT_ARTIFACTS)


def is_completed_run(path: Path) -> bool:
    required = (
        path / "best.pt",
        path / "run_info.json",
        path / "performance.json",
    )
    if not all(item.is_file() for item in required):
        return False
    if not any(path.glob("metrics_test_*.json")):
        return False
    try:
        import json

        info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return info.get("status") == "COMPLETED"


def archive_directory(path: Path, archive_root: Path, *, label: str) -> Path:
    """Move one exact result directory into an archive without deleting it."""

    source = path.resolve()
    if not source.is_dir():
        raise ValueError(f"archive source is not a directory: {source}")
    archive_base = archive_root.resolve()
    archive_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = archive_base / f"{label}__{stamp}"
    if destination.exists():
        destination = archive_base / f"{label}__{stamp}__{uuid4().hex[:8]}"
    source.rename(destination)
    return destination
