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


def _stub_trace(
    op_names: list[str],
    *,
    signal: bool = False,
    modules: list[Any] | None = None,
    **extra: Any,
) -> Any:
    """Build a minimal trace stub for direct driver tests."""

    fields: dict[str, Any] = {
        "escape_diagnostics": [],
        "_had_unattributed_tensor_args": signal,
        "ops": [types.SimpleNamespace(func_name=name) for name in op_names],
        "modules": modules or [],
        "capture_verification_reason": None,
    }
    fields.update(extra)
    return types.SimpleNamespace(**fields)


def test_defusion_is_not_counted_as_recovery() -> None:
    """R16-1: a rescued op multiset that LOSES ops is mode perturbation.

    The one-sided ``Counter.__sub__`` oracle read eval-MHA de-fusion (one
    fused op replaced by many small ops) as pure gains and swapped the
    canonical fused primary for the mode-perturbed rescue. The two-sided
    oracle keeps the mode-free primary and discloses both deltas.
    """

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["mha_fused", "relu"], signal=True)
    rescued = _stub_trace(["matmul", "softmax", "matmul", "relu"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert result is primary
    assert result.capture_verification_reason == "escape_rescue_unrecovered"
    info = result.rescue_rerun
    assert info["recovered"] is False
    assert "mha_fused" in info["lost_ops"]
    assert "matmul" in info["recovered_ops"]


def test_strict_superset_rescue_still_recovers() -> None:
    """R16-1 control: gains with zero losses stay a genuine recovery."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(["relu", "cos"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert result is rescued
    assert result.capture_verification_reason == "mode_rescue_rerun"
    assert result.rescue_rerun["recovered_ops"] == ("cos",)


def test_buffer_writing_primary_refuses_the_rescue_rerun() -> None:
    """R16-2: a primary that WROTE buffers would double-apply them; refuse."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    primary.ops.append(
        types.SimpleNamespace(
            func_name="none", buffer_write_kind="inplace", label_raw="buffer_bn1"
        )
    )

    def run_capture() -> Any:
        runs.append("run")
        return primary

    with pytest.warns(UserWarning, match="double-apply"):
        result = capture_with_rescue(run_capture)
    assert len(runs) == 1, "the forward must run exactly once for a buffer-writing primary"
    assert result is primary
    info = result.rescue_rerun
    assert info["skipped_reason"] == "buffer_writes_double_forward"
    assert info["forward_runs"] == 1
    assert result.capture_verification_reason == "escape_rescue_unrecovered"


def test_train_mode_batchnorm_capture_skips_rescue_end_to_end(raw_cos: Any) -> None:
    """R16-2 end-to-end: a train-mode BN model's counter increments ONCE."""

    wrap_torch()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(2)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(raw_cos(self.bn(v)))

    model = Model()
    model.train()
    with pytest.warns(UserWarning):
        trace = tl.trace(model, torch.randn(4, 2))
    assert model.bn.num_batches_tracked.item() == 1, "the forward must not run twice"
    info = trace.rescue_rerun
    assert info is not None
    assert info["skipped_reason"] == "buffer_writes_double_forward"
    assert "cos" not in [op.func_name for op in trace.ops]


def test_stateless_primary_still_rescues() -> None:
    """R16-2 control: no buffer writes -> the re-run proceeds."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(["relu", "cos"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert len(runs) == 2
    assert result is rescued


def test_recovered_rescue_never_clobbers_dynamo_verdict() -> None:
    """R16-3: dynamo_region_not_logged outranks mode_rescue_rerun in _mark."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(
        ["relu", "cos"],
        _raw_dynamo_region_detected=True,
        capture_verification_reason="dynamo_region_not_logged",
    )

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert result is rescued
    assert result.capture_verification_reason == "dynamo_region_not_logged"
    assert result.rescue_rerun is not None, "the rescue attempt must stay disclosed"


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
