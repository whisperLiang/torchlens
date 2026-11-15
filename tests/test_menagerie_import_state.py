"""Model catalog imports must preserve the caller's numeric environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_classic_import_preserves_matmul_precision() -> None:
    """Importing Clay cannot change linear replay or benchmark precision globally."""
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    script = """
import torch

torch.set_float32_matmul_precision("highest")
before = torch.get_float32_matmul_precision()
import menagerie.classics.rs_L752_clay
after = torch.get_float32_matmul_precision()
assert after == before, (before, after)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
