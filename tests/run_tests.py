#!/usr/bin/env python3
"""Canonical CPU regression-suite entry point.

Keeping a hand-maintained filename list previously made this command report
missing, deleted tests and bypassed pytest's import setup.  Discovery is now
delegated to pytest so this entry point and ``pytest -q tests`` are identical.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parent.parent
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        str(project / "tests"),
    ]
    return subprocess.run(command, cwd=project, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
