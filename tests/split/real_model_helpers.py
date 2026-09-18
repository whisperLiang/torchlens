"""Shared enablement and subprocess isolation for real-model split tests."""

from __future__ import annotations

import gc
import os
import site
import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

_SUBPROCESS_UNAVAILABLE_EXIT_CODE = 75


def _enabled() -> bool:
    """Return whether real-model tests are enabled."""

    return os.environ.get("TORCHLENS_REAL_MODEL_TESTS") != "0"


def _strict() -> bool:
    """Return whether missing real-model environments are hard failures."""

    return os.environ.get("TORCHLENS_REAL_MODEL_STRICT") == "1"


def _skip_unless_enabled() -> None:
    """Skip real-model split tests only when explicitly disabled."""

    if not _enabled():
        pytest.skip("TORCHLENS_REAL_MODEL_TESTS=0 disables real-model split tests.")


def _skip_if_module_missing(module_name: str) -> None:
    """Skip when an optional backend/model dependency is unavailable."""

    if find_spec(module_name) is None:
        if _strict():
            pytest.fail(f"strict real-model mode requires {module_name!r}.")
        pytest.skip(f"{module_name!r} is not installed.")


def _run_backend_subprocess(
    code: str,
    *,
    timeout: int = 180,
    env_overrides: dict[str, str] | None = None,
) -> None:
    """Run a backend-isolated real-model split scenario."""

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["DEBUG"] = "0"
    cuda_library_dirs = sorted(
        {
            str(path)
            for site_dir in site.getsitepackages()
            for path in (Path(site_dir) / "nvidia").glob("*/lib")
            if path.is_dir()
        }
    )
    if cuda_library_dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*cuda_library_dirs, env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    libcuda = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
    if libcuda.exists():
        env["LD_PRELOAD"] = os.pathsep.join([str(libcuda), env.get("LD_PRELOAD", "")]).rstrip(
            os.pathsep
        )
    helper_path = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = helper_path + os.pathsep + env.get("PYTHONPATH", "")
    if env_overrides:
        env.update(env_overrides)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from v2_helpers import split_request\n" + textwrap.dedent(code),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode == _SUBPROCESS_UNAVAILABLE_EXIT_CODE:
        reason = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        pytest.fail(reason or "real-model backend subprocess was unavailable.")
    assert result.returncode == 0, (
        f"real-model backend subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
