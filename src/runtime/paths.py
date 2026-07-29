"""Small path helpers shared by CLI entry points."""

from __future__ import annotations

from pathlib import Path


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
    return resolve_output_root(project_root, output_root) / model_name / run_id
