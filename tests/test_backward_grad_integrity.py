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


@pytest.mark.smoke
def test_op_grad_payloads_charge_the_save_budget() -> None:
    """Retained op-gradient payloads charge the save-budget accountant.

    Frozen parameters isolate the op/output gradient path (no param grads
    exist): the committed footprint must grow when backward retains op grad
    payloads, otherwise the budget's committed-footprint claim is false
    after any backward.
    """

    torch.manual_seed(0)
    model = _TinyModel()
    for param in model.parameters():
        param.requires_grad_(False)
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True, save_grads="all"),
    )
    budget = trace._save_budget_accountant
    assert budget is not None
    committed_before = sum(ledger.committed_bytes for ledger in budget.ledgers.values())

    trace.log_backward(_loss(trace))

    grad_bytes = sum(
        int(record.grad.nelement() * record.grad.element_size())
        for layer in trace.layer_list
        for record in layer.grads
        if isinstance(record.grad, torch.Tensor)
    )
    assert grad_bytes > 0, "expected retained op gradient payloads"
    committed_after = sum(ledger.committed_bytes for ledger in budget.ledgers.values())
    assert committed_after >= committed_before + grad_bytes, (
        "op gradient payloads bypassed the save-budget accountant: "
        f"committed {committed_before} -> {committed_after}, grads {grad_bytes}"
    )


@pytest.mark.smoke
def test_selective_save_grads_consults_predicate_for_params() -> None:
    """Callable save_grads policies see parameter gradients, not a silent drop.

    A selector/callable policy used to bypass parameters entirely (no param
    context existed), so every param grad payload silently dropped no matter
    what the predicate asked for.
    """

    seen_kinds: set[str] = set()

    def keep_param_grads(ctx: object) -> bool:
        """Retain exactly the parameter gradients."""

        kind = getattr(ctx, "grad_kind", None)
        if isinstance(kind, str):
            seen_kinds.add(kind)
        return kind == "param_grad"

    trace = _armed_trace()
    trace.log_backward(_loss(trace), save_grads=keep_param_grads)

    param_payloads = [record.grad for param in trace.param_logs.values() for record in param.grads]
    assert param_payloads, "expected parameter gradient records"
    assert "param_grad" in seen_kinds, "predicate never saw a parameter context"
    assert all(isinstance(p, torch.Tensor) for p in param_payloads), (
        "selective save_grads silently dropped parameter gradients"
    )
    # The same predicate rejected op gradients, so none retain payloads.
    op_payloads = [
        record.out if hasattr(record, "out") else record.grad
        for layer in trace.layer_list
        for record in layer.grads
    ]
    assert all(not isinstance(p, torch.Tensor) for p in op_payloads)


@pytest.mark.smoke
def test_backward_finalize_drains_pending_cpu_async_copies() -> None:
    """The backward finalize seam fences pending cpu_async D2H copies.

    The R36-1 drain only ran at the FORWARD finalize seam, so a cpu_async
    grad payload recorded during log_backward stayed on the pending-event
    list and a host read could observe partial bytes from the unfinished
    non_blocking copy. A planted fence object stands in for an in-flight
    CUDA copy event (CPU-only hosts record no real events).
    """

    from torchlens.utils import tensor_utils

    class _FenceSpy:
        """Pending-copy stand-in recording whether it was synchronized."""

        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            """Mark the pending copy as fenced."""

            self.synchronized = True

    trace = _armed_trace(save_mode="cpu_async")
    fence = _FenceSpy()
    tensor_utils._CPU_ASYNC_PENDING_EVENTS.append(fence)
    try:
        trace.log_backward(_loss(trace))
        assert fence.synchronized, (
            "log_backward finished without draining pending cpu_async copy fences"
        )
        assert fence not in tensor_utils._CPU_ASYNC_PENDING_EVENTS
    finally:
        if fence in tensor_utils._CPU_ASYNC_PENDING_EVENTS:
            tensor_utils._CPU_ASYNC_PENDING_EVENTS.remove(fence)
