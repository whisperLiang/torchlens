"""Helpers for Paddle split tests that must run outside the main pytest process."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest


def run_paddle_subprocess(code: str, *, timeout: int = 180, require_cuda: bool = False) -> None:
    """Run a Paddle split scenario in a backend-isolated subprocess."""

    if find_spec("paddle") is None:
        pytest.skip("'paddle' is not installed.")
    env = os.environ.copy()
    helper_path = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = helper_path + os.pathsep + env.get("PYTHONPATH", "")
    source = "from v2_helpers import split_request\n" + textwrap.dedent(code)
    cuda_skip_marker = "__torchlens_paddle_cuda_unavailable__"
    if require_cuda:
        source = (
            "import sys\nimport paddle\n"
            "if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:\n"
            f"    print({cuda_skip_marker!r})\n    sys.exit(75)\n"
        ) + source
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if require_cuda and result.returncode == 75 and cuda_skip_marker in result.stdout.splitlines():
        pytest.skip("A CUDA-enabled Paddle runtime and GPU are required.")
    assert result.returncode == 0, (
        f"Paddle split subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
