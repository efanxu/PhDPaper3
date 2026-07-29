"""Load selected Time-Series-Library modules without shadowing ``src/models``.

The upstream repository is treated as a read-only source tree.  Modules are
loaded from explicit file paths under a private namespace, while its ordinary
``layers`` imports are available through a short-lived, controlled source-root
entry.  The project's ``models`` package remains first on ``sys.path`` and is
never replaced in ``sys.modules``.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from collections.abc import Iterator


_NAMESPACE = "_phdpaper3_time_series_library"


def resolve_time_series_library_model(
    model_name: str,
    *,
    source_root: str | Path,
) -> Path:
    """Resolve ``<source_root>/models/<model_name>.py`` by exact path."""

    if not model_name or Path(model_name).name != model_name or "/" in model_name or "\\" in model_name:
        raise ValueError(f"invalid Time-Series-Library model name: {model_name!r}")
    root = Path(source_root).resolve()
    path = root / "models" / f"{model_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Time-Series-Library model does not exist: {path}")
    return path


@contextmanager
def _controlled_source_root(source_root: Path) -> Iterator[None]:
    """Expose upstream ``layers`` after the project's source root."""

    original = list(sys.path)
    root_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    # The project source path is expected to be first.  Inserting at index 1
    # keeps it ahead of the upstream root even when callers use a bare Python
    # interpreter instead of scripts/run.py.
    insert_at = 1 if sys.path else 0
    sys.path.insert(insert_at, root_text)
    try:
        yield
    finally:
        sys.path[:] = original


def load_time_series_library_model(
    model_name: str,
    *,
    source_root: str | Path,
    module_name: str | None = None,
) -> ModuleType:
    """Load one upstream model module under a unique internal module name."""

    path = resolve_time_series_library_model(model_name, source_root=source_root)
    root = Path(source_root).resolve()
    safe_name = model_name.replace("-", "_")
    qualified_name = module_name or f"{_NAMESPACE}.models.{safe_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for upstream model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        with _controlled_source_root(root):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def load_time_series_library_model_class(
    model_name: str,
    *,
    source_root: str | Path,
    class_name: str = "Model",
) -> type:
    """Load and return the conventional upstream ``Model`` class."""

    module = load_time_series_library_model(model_name, source_root=source_root)
    value = getattr(module, class_name, None)
    if not isinstance(value, type):
        raise TypeError(f"upstream model {model_name!r} does not define class {class_name!r}")
    return value
