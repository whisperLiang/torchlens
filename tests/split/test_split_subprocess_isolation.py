"""The backend subprocess harness must preserve its GPU memory isolation."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest
import real_model_helpers


@pytest.mark.parametrize("cuda_initialized", [False, True])
def test_backend_subprocess_only_cleans_an_existing_cuda_context(
    cuda_initialized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release existing parent caches without allocating a new CUDA context."""

    events: list[str] = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(real_model_helpers.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(real_model_helpers.torch.cuda, "is_initialized", lambda: cuda_initialized)

    def available() -> bool:
        """Reject a capability probe in place of the parent-state check."""

        pytest.fail("cleanup must inspect existing CUDA state, not device availability")

    monkeypatch.setattr(real_model_helpers.torch.cuda, "is_available", available)
    monkeypatch.setattr(
        real_model_helpers.torch.cuda, "empty_cache", lambda: events.append("empty_cache")
    )
    monkeypatch.setattr(
        real_model_helpers.torch.cuda, "ipc_collect", lambda: events.append("ipc_collect")
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Inspect the actual child launch without requiring a GPU in this test."""

        events.append("subprocess")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert "assert True" in command[-1]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(real_model_helpers.subprocess, "run", run)
    real_model_helpers._run_backend_subprocess("assert True")

    cleanup = ["empty_cache", "ipc_collect"] if cuda_initialized else []
    assert events == ["gc", *cleanup, "subprocess"]
