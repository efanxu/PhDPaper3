"""Unified command entry point for PhDPaper3."""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401  (bootstraps src/ on direct execution)
import sys


if len(sys.argv) > 1 and sys.argv[1] == "_worker":
    from cli.worker import main

    raise SystemExit(main(sys.argv[2:]))

from cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
