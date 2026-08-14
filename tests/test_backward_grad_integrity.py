"""Backward/grad integrity regression tests (fix-bwgrad lane, grind-p3 T2).

Each test here is red-capable: it FAILS against the defect it pins and passes
only with the corresponding fix in place.
"""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class _TinyModel(nn.Module):
    """Small MLP for backward capture tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.fc2(torch.relu(self.fc1(x)))


def _armed_trace(save_mode: str = "copy") -> tl.Trace:
    """Capture a backward-ready trace of the tiny model."""

    torch.manual_seed(0)
    model = _TinyModel()
    x = torch.randn(2, 3, requires_grad=True)
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True, save_grads="all"),
        save_mode=save_mode,
    )


def _loss(trace: tl.Trace) -> torch.Tensor:
    """Return a scalar loss built from the trace's saved output."""

    return trace[trace.output_layers[0]].out.sum()


@pytest.mark.smoke
@pytest.mark.parametrize("save_mode", ["reference", "view"])
def test_second_backward_does_not_rewrite_recorded_grads(save_mode: str) -> None:
    """Recorded pass-1 grad payloads stay frozen when a second backward runs.

    Under save_mode="reference"/"view" the grad payload chokepoint used to
    store a live ALIAS of the observed gradient; AccumulateGrad steals that
    tensor as ``.grad`` and accumulates into it in place, so the pass-1
    record silently became the running sum (2x truth) after a second
    backward.
    """

    trace = _armed_trace(save_mode=save_mode)
    loss = _loss(trace)
    trace.log_backward(loss, retain_graph=True)

    param_payloads = [record.grad for param in trace.param_logs.values() for record in param.grads]
    op_payloads = [
        record.out if hasattr(record, "out") else record.grad
        for layer in trace.layer_list
        for record in layer.grads
    ]
    payloads = [p for p in param_payloads + op_payloads if isinstance(p, torch.Tensor)]
    assert payloads, "expected recorded pass-1 gradient payloads"
    snapshots = [p.clone() for p in payloads]

    trace.log_backward(loss)

    for payload, snapshot in zip(payloads, snapshots):
        assert torch.equal(payload, snapshot), (
            "pass-1 gradient record was rewritten by a second backward: "
            f"{snapshot.flatten()[:4]} -> {payload.flatten()[:4]}"
        )


@pytest.mark.smoke
def test_reentered_recording_backward_restores_tensor_backward() -> None:
    """Re-entering one RecordingBackward context leaves no persistent wrapper.

    A second ``__enter__`` on the same context object used to capture the
    first entry's wrapper as the "original", so the outer exit could never
    match its own wrapper and permanently leaked a TorchLens wrapper on the
    process-global ``torch.Tensor.backward``.
    """

    trace = _armed_trace()
    original_backward = torch.Tensor.backward
    context = trace.recording_backward()
    try:
        with context:
            with context:
                _loss(trace).backward()
    finally:
        if torch.Tensor.backward is not original_backward:
            torch.Tensor.backward = original_backward  # type: ignore[method-assign]
            pytest.fail("re-entered recording_backward() leaked a wrapper on torch.Tensor.backward")
    # The backward inside the nested block is still recorded exactly once.
    assert trace.num_backward_passes == 1
