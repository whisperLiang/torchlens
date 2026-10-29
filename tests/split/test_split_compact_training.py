"""Compact training retains live state while discarding capture autograd history."""

from __future__ import annotations

import copy
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.split import PlacementPlan, SplitFeatures, SplitRequest, after, pipeline
from torchlens.split._torch_compact import compact_torch_graph
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.graph import split_graph_from_trace


class TiedBranches(torch.nn.Module):
    """Reuse one weight on both sides of a split, with nested outputs."""

    def __init__(self) -> None:
        """Create a shared affine and a nonlinearity."""

        super().__init__()
        self.shared = torch.nn.Linear(4, 4)
        self.middle = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Return a residual output and an auxiliary branch."""

        h = self.middle(self.shared(x))
        return {"prediction": self.shared(h) + x, "aux": (h,)}


def _loss(output: dict[str, Any], targets: torch.Tensor) -> torch.Tensor:
    """Differentiate both branches of the nested output."""

    return (output["prediction"] - targets).square().mean() + output["aux"][0].square().mean()


def _request() -> SplitRequest:
    """Opt into training compaction without changing default training policy."""

    return SplitRequest(after("middle"), features=SplitFeatures(training=True, retain_trace=False))


@pytest.mark.parametrize("optimizer_kind", ["sgd", "momentum", "adam"])
def test_compact_tied_training_recut_preserves_gradients_and_optimizer_state(
    optimizer_kind: str,
) -> None:
    """Repeated recuts preserve live identities and external optimizer continuity."""

    torch.manual_seed(71)
    reference = TiedBranches()
    model = copy.deepcopy(reference)
    runtime = tl.split.prepare(model, torch.randn(2, 4), _request())
    assert not runtime.retains_trace
    assert runtime.batch_validation["status"] == "passed"

    def optimizer(parameters: Any) -> torch.optim.Optimizer:
        """Construct the same optimizer for both models."""

        if optimizer_kind == "adam":
            return torch.optim.Adam(parameters, lr=0.003)
        return torch.optim.SGD(
            parameters, lr=0.003, momentum=0.8 if optimizer_kind == "momentum" else 0
        )

    expected_optimizer = optimizer(reference.parameters())
    actual_optimizer = optimizer(model.parameters())
    parameter_ids = tuple(id(value) for value in model.parameters())
    for step, batch in enumerate((1, 2, 4, 8)):
        expected_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(batch, 4, requires_grad=True)
        split_inputs = inputs.detach().clone().requires_grad_()
        targets = torch.randn(batch, 4)
        expected = reference(inputs)
        expected_loss = _loss(expected, targets)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(split_inputs)
        actual_loss, boundary_gradients = runtime.train_suffix(boundary, targets, loss_fn=_loss)
        runtime.backward_prefix(boundary, boundary_gradients)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(split_inputs.grad, inputs.grad)
        for actual, expected_param in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected_param.grad)
        expected_optimizer.step()
        actual_optimizer.step()
        for actual, expected_param in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected_param)
            torch.testing.assert_close(
                actual_optimizer.state[actual], expected_optimizer.state[expected_param]
            )
        compute = runtime.trace_graph.compute_nodes
        point = tl.split.before(compute[0].canonical_id) if step % 2 else after("middle")
        runtime = runtime.at(point)
        assert tuple(id(value) for value in model.parameters()) == parameter_ids
        assert not runtime.retains_trace


def test_compact_training_collects_capture_and_preserves_existing_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture cleanup releases values without clearing caller-owned parameter gradients."""

    original = pipeline.capture_model
    traces: list[weakref.ReferenceType[Any]] = []
    payloads: list[weakref.ReferenceType[Any]] = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        """Observe lifetimes through weak references only."""

        trace = original(*args, **kwargs)
        traces.append(weakref.ref(trace))
        payloads.extend(
            weakref.ref(op.out)
            for op in trace
            if not op.is_input and isinstance(op.out, torch.Tensor)
        )
        return trace

    monkeypatch.setattr(pipeline, "capture_model", observe)
    model = TiedBranches()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = tuple(parameter.grad for parameter in model.parameters())
    runtime = tl.split.prepare(model, torch.randn(2, 4, requires_grad=True), _request())
    gc.collect()
    assert traces and all(reference() is None for reference in traces)
    assert payloads and all(reference() is None for reference in payloads)
    assert all(
        parameter.grad is gradient
        for parameter, gradient in zip(model.parameters(), gradients, strict=True)
    )
    handles = {id(param.handle) for node in runtime.trace_graph.nodes for param in node.param_refs}
    assert handles == {id(parameter) for parameter in model.parameters()}


def test_compact_training_refuses_nonleaf_source_history() -> None:
    """A one-shot source history cannot silently become a disconnected constant."""

    model = TiedBranches()
    capture = pipeline.capture_model(model, (torch.randn(2, 4),), _request())
    try:
        graph = split_graph_from_trace(capture)
        nonleaf = torch.ones(2, 4, requires_grad=True) * 2
        source = replace(
            graph.compute_nodes[0], target=None, parents=(), op=SimpleNamespace(out=nonleaf)
        )
        with pytest.raises(SplitUnsupportedError) as error:
            compact_torch_graph(replace(graph, nodes=(source,)), training=True)
        assert error.value.context.reason == "compact_training_autograd_source"
    finally:
        capture.cleanup()


def test_compact_training_updates_live_batchnorm_buffers() -> None:
    """Repeated training uses current buffers and leaves capture-time state untouched."""

    reference = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    )
    model = copy.deepcopy(reference)
    inputs = torch.randn(3, 4)
    request = SplitRequest(
        after("0"), features=SplitFeatures(training=True, retain_trace=False, batch_axes={})
    )
    runtime = tl.split.prepare(model, inputs, request)
    torch.testing.assert_close(model.state_dict(), reference.state_dict())
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        inputs = torch.randn(3, 4)
        expected = reference(inputs)
        actual = runtime.run_suffix(runtime.run_training_prefix(inputs))
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(model.state_dict(), reference.state_dict())
        for parameter, expected_param in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter.grad, expected_param.grad)


def test_compact_training_preserves_rng_and_matches_diagnostic_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compaction retains per-call stochastic execution evidence without advancing user RNG."""

    capture = pipeline.capture_model

    def seeded_capture(*args: Any, **kwargs: Any) -> Any:
        """Give independent captures the same stochastic execution context."""

        return capture(*args, **{**kwargs, "random_seed": 1907})

    monkeypatch.setattr(pipeline, "capture_model", seeded_capture)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))
    diagnostic_model = copy.deepcopy(model)
    inputs = torch.randn(3, 4)
    request = SplitRequest(
        after("0"), features=SplitFeatures(training=True, retain_trace=False, batch_axes={})
    )
    rng = torch.random.get_rng_state().clone()
    runtime = tl.split.prepare(model, inputs, request)
    diagnostic = tl.split.prepare(
        diagnostic_model,
        inputs,
        replace(request, features=replace(request.features, retain_trace=True)),
    )
    try:
        assert torch.equal(torch.random.get_rng_state(), rng)
        for _ in range(2):
            actual = runtime.run_suffix(runtime.run_training_prefix(inputs))
            expected = diagnostic.run_suffix(diagnostic.run_training_prefix(inputs))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual.sum(), tuple(model.parameters())),
                torch.autograd.grad(expected.sum(), tuple(diagnostic_model.parameters())),
            )
            assert torch.equal(torch.random.get_rng_state(), rng)
    finally:
        diagnostic.trace.cleanup()


def test_compact_training_replays_cpu_autocast_gradients() -> None:
    """Native replay restores captured AMP for forward and ordinary autograd for backward."""

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))
    inputs = torch.randn(3, 4, requires_grad=True)
    request = SplitRequest(
        after("1"), features=SplitFeatures(training=True, retain_trace=False, batch_axes={})
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        runtime = tl.split.prepare(model, inputs, request)
        expected = model(inputs)
    actual = runtime.run_suffix(runtime.run_training_prefix(inputs))
    assert actual.dtype == torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.float().sum(), (inputs, *model.parameters())),
        torch.autograd.grad(expected.float().sum(), (inputs, *model.parameters())),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compact_training_cpu_cuda_gradient_handoff_and_update() -> None:
    """A CPU prefix and CUDA suffix preserve input gradients and independent state updates."""

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    reference = copy.deepcopy(model)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    request = SplitRequest(
        after("1"),
        features=SplitFeatures(training=True, retain_trace=False),
        placement=PlacementPlan.across("cpu", "cuda:0"),
    )
    runtime = tl.split.prepare(model, torch.randn(3, 4), request)
    runtime.prefix_parameters()
    runtime.suffix_parameters()
    # Bindings expose parameter collections without promising model traversal
    # order. Match replicas to original sources before comparing gradients.
    effective = {
        entry.source_id: entry.value
        for segment in (runtime.segments.training_prefix, runtime.segments.suffix)
        for entry in segment._state.entries()
    }
    parameters = [effective[id(parameter)] for parameter in model.parameters()]
    actual_optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    for _ in range(2):
        actual_optimizer.zero_grad(set_to_none=True)
        expected_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(3, 4, requires_grad=True)
        split_inputs = inputs.detach().clone().requires_grad_()
        targets = torch.randn(3, 2)
        expected_loss = (reference(inputs) - targets).square().mean()
        expected_loss.backward()
        boundary = runtime.run_training_prefix(split_inputs)
        actual_loss, gradients = runtime.train_suffix(boundary, targets.cuda())
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(actual_loss.cpu(), expected_loss)
        torch.testing.assert_close(split_inputs.grad, inputs.grad)
        for actual, expected in zip(parameters, reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad.cpu(), expected.grad)
        actual_optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(parameters, reference.parameters(), strict=True):
            torch.testing.assert_close(actual.cpu(), expected)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    # Only the prefix references the live CPU model; suffix updates remain in its replica.
    torch.testing.assert_close(model[2].weight, original["2.weight"])
    torch.testing.assert_close(model[2].bias, original["2.bias"])
