"""Rescue-rerun mechanism tests (stage-2 safety net).

The outcome corpus (``test_detached_reference_capture_outcomes.py``) pins
per-escape-class outcomes; THIS module pins the driver mechanics: the net is
never armed on a clean primary capture, ineligible captures skip the rescue,
RNG is restored to capture entry so stochastic re-runs replay the primary
draw, and the opt-in escape-detector diagnostic also triggers the rescue.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

pytestmark = pytest.mark.smoke


@pytest.fixture()
def raw_cos() -> Any:
    """A pristine pre-wrap ``torch.cos`` reference, rewrapping afterwards."""
    unwrap_torch()
    raw = torch.cos
    assert not is_decorated_function(raw)
    try:
        yield raw
    finally:
        wrap_torch()


def _stale_closure_model(raw: Callable[..., Any]) -> nn.Module:
    def invoke(v: torch.Tensor) -> torch.Tensor:
        return raw(v)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(invoke(torch.sigmoid(v)))

    return Model()


def test_clean_capture_is_never_rescued() -> None:
    """No escape signal -> exactly one mode-free run, no disclosure."""

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(v))

    trace = tl.trace(Model(), torch.randn(3, 4))
    assert trace.rescue_rerun is None
    assert trace.capture_verification_reason != "mode_rescue_rerun"


def test_rescue_restores_rng_to_capture_entry() -> None:
    """The rescue run sees EXACTLY the capture-entry RNG state.

    Driven directly against the driver: a capture stub draws from torch RNG
    on every invocation and reports an escape signal on the first. The
    driver must restore RNG before the re-run, so both invocations observe
    the identical draw (the rescue replays the primary's randomness).
    """

    from torchlens.backends.torch.rescue import capture_with_rescue

    draws: list[float] = []

    def run_capture() -> Any:
        draws.append(torch.rand(1).item())
        return types.SimpleNamespace(
            escape_diagnostics=[],
            _had_unattributed_tensor_args=len(draws) == 1,
            ops=[],
        )

    capture_with_rescue(run_capture)
    assert len(draws) == 2
    assert draws[0] == draws[1]


def test_streaming_capture_skips_rescue(raw_cos: Any, tmp_path: Any) -> None:
    """A disk-streamed capture is not re-runnable: escape reported, no rescue."""

    wrap_torch()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(
            _stale_closure_model(raw_cos),
            torch.tensor([0.25, 0.5]),
            storage=tl.to_disk(str(tmp_path / "run.tlspec")),
        )
    assert "cos" not in [op.func_name for op in trace.ops]
    assert trace.rescue_rerun is None


def test_escape_detector_diagnostic_triggers_rescue(raw_cos: Any) -> None:
    """The opt-in shadow detector's diagnostics are a rescue trigger too."""

    from torchlens._errors import TorchLensCaptureGapWarning

    try:
        wrap_torch(escape_detector="shadow")
        with pytest.warns(TorchLensCaptureGapWarning):
            trace = tl.trace(_stale_closure_model(raw_cos), torch.tensor([0.25, 0.5]))
        info = trace.rescue_rerun
        assert info is not None and info["recovered"] is True
        assert trace.capture_verification_reason == "mode_rescue_rerun"
        assert info["primary_escape_diagnostics"]
        assert "cos" in [op.func_name for op in trace.ops]
    finally:
        unwrap_torch()
        wrap_torch()
