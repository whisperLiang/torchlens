"""Forward op timings remain measurable when the wall clock has coarse resolution."""

import time

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


def test_forward_duration_does_not_depend_on_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wall-clock tick shared by short ops must not erase their durations."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)
    monkeypatch.setattr(time, "time", lambda: 1_234_567_890.0)
    trace = tl.trace(model, x)
    try:
        measured = [op for op in trace.ops if op.func_name in {"linear", "relu"}]
        assert len(measured) == 2
        assert all(float(op.func_duration) > 0 for op in measured)
    finally:
        trace.cleanup()
