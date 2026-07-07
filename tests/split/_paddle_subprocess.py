"""Helpers for Paddle split tests that must run outside the main pytest process."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest


def run_paddle_subprocess(code: str, *, timeout: int = 180) -> None:
    """Run a Paddle split scenario in a backend-isolated subprocess."""

    if find_spec("paddle") is None:
        pytest.skip("'paddle' is not installed.")
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        "Paddle split subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
