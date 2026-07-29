from __future__ import annotations

from pathlib import Path
import sys

from integrations.time_series_library import load_time_series_library_model


ROOT = Path(__file__).resolve().parents[1]


def test_upstream_model_is_loaded_by_explicit_path_without_replacing_project_models(tmp_path: Path) -> None:
    models = tmp_path / "models"
    layers = tmp_path / "layers"
    models.mkdir()
    layers.mkdir()
    (layers / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    (models / "Tiny.py").write_text(
        "from layers.helper import VALUE\n\nclass Model:\n    value = VALUE\n",
        encoding="utf-8",
    )

    import models as project_models
    import models.base as project_base

    loaded = load_time_series_library_model("Tiny", source_root=tmp_path)
    assert loaded.Model.value == 7
    assert sys.modules["models"] is project_models
    assert sys.modules["models.base"] is project_base
    assert loaded.__name__.startswith("_phdpaper3_time_series_library.")
    sys.modules.pop("layers.helper", None)
    sys.modules.pop("layers", None)


def test_real_time_series_library_model_can_load_without_models_shadowing() -> None:
    import models.base as project_base

    loaded = load_time_series_library_model("DLinear", source_root=ROOT / "Time-Series-Library")
    assert loaded.__name__.startswith("_phdpaper3_time_series_library.")
    assert sys.modules["models.base"] is project_base
