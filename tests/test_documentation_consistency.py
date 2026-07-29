from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MARKDOWN = {
    "README.md",
    "MODEL_INTEGRATION_INDEX.md",
    "HANDOFF.md",
    "docs/COMMAND_REFERENCE.md",
}


def _tracked_markdown() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.replace("\\", "/") for line in result.stdout.splitlines() if line}


def test_only_the_four_project_markdown_documents_are_tracked() -> None:
    assert _tracked_markdown() == ALLOWED_MARKDOWN
    assert not (ROOT / "data").exists()
    assert not (ROOT / "docs" / "MIGRATION_REPORT.md").exists()
    assert (ROOT / "dataset" / ".gitkeep").is_file()


def test_documentation_uses_dataset_and_current_entrypoint() -> None:
    documents = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in ALLOWED_MARKDOWN)
    assert "data/" not in documents.replace("src/data/", "")
    assert "MIGRATION_REPORT.md" not in documents
    assert "run_all_models.py" not in documents
    assert "compare_repeated_runs.py" not in documents
    assert "check_model.py" not in documents
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts\\run.py batch" not in readme
    assert "--models" not in readme
    assert "scripts\\run.py" in readme


def test_model_index_core_paths_exist() -> None:
    index = (ROOT / "MODEL_INTEGRATION_INDEX.md").read_text(encoding="utf-8")
    paths = re.findall(r"`([^`]+)`", index)
    required = [
        path
        for path in paths
        if path.startswith(("configs/", "src/", "tests/")) and "<" not in path
    ]
    assert required
    for relative in required:
        assert (ROOT / relative).exists(), relative


def test_environment_contract_and_model_parameter_documentation_rules() -> None:
    environment_path = ROOT / "configs" / "environments.yaml"
    loaded = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    assert loaded["default_environment"] == "tslib"
    assert set(loaded["environments"]) >= {"tslib", "tsl"}
    index = (ROOT / "MODEL_INTEGRATION_INDEX.md").read_text(encoding="utf-8")
    assert "runtime.environment" in index
    assert "ordinary model" in index
    public_documents = "\n".join(
        (ROOT / path).read_text(encoding="utf-8") for path in ALLOWED_MARKDOWN
    )
    for model_parameter in ("hidden_dim", "e_layers", "diffusion_steps", "node_embedding_dim"):
        assert model_parameter not in public_documents


def test_command_reference_is_generated_from_current_parser() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts\\generate_command_reference.py",
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
