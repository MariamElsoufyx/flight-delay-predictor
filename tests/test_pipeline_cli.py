"""CLI smoke tests (no raw data required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_pipeline_dry_run(project_root: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "run_pipeline.py", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "[dry-run]" in proc.stdout
