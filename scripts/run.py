"""Unified command entry point for PhDPaper3."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401  (bootstraps src/ on direct execution)
from cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
