"""Boundary, target, device, and lifetime contracts for suffix microbatching."""

from __future__ import annotations

import copy
import weakref
from collections import namedtuple
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError
from torchlens.split.shape_program import DimExpr
from torchlens.split.training import train_suffix_result


class ContractMlp(nn.Module):
    """Small model with a fixed-width, batch-symbolic boundary."""

    def __init__(self) -> None:
        """Build separate trainable prefix and suffix layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 7)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(7, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer model."""

        return self.fc2(self.relu(self.fc1(x)))


class AxisAndSharedModel(nn.Module):
    """Cross a nonleading batch axis and a differentiable shared value."""

    def __init__(self) -> None:
        """Build trainable layers around the unusual boundary layout."""

        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        """Reuse one shared vector for every independent sample."""

        hidden = self.fc1(x).transpose(0, 1)
        common = shared.sin()
        return self.fc2(hidden.transpose(0, 1) + common)


class ResidualAndIndexModel(ContractMlp):
    """Keep activation, skip, and integer values alive across the boundary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Carry a skip and nondifferentiable index alongside the activation."""

        hidden = self.relu(self.fc1(x))
        index = x.argmax(dim=1)
        return self.fc2(hidden) + x[:, :3] + index.float().unsqueeze(1)


class CountingAdam(torch.optim.Adam):
    """Expose optimizer side effects for preflight-refusal assertions."""

    def __init__(self, parameters: Any) -> None:
        """Construct a lazy optimizer without allocating its moment buffers."""

        super().__init__(parameters, lr=0.01)
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """Count clearing once per logical batch."""

        self.zero_calls += 1
        super().zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Count updating once per logical batch."""

        self.step_calls += 1
        return super().step(*args, **kwargs)


def _prepare(model: nn.Module | None = None) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Prepare a deterministic seven-sample boundary fixture."""

    torch.manual_seed(314)
    model = ContractMlp() if model is None else model
    x, targets = torch.randn(7, 4), torch.randn(7, 3)
    return tl.split.prepare(model, x, split_request("after:relu", trainable=True)), x, targets


def _assert_gradients(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare complete detached gradients including nonleading axes."""

    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert not value.requires_grad and value.grad_fn is None
        torch.testing.assert_close(value.cpu(), expected[key].cpu(), atol=2e-6, rtol=2e-5)


def test_nonleading_batch_and_shared_value_match_full_prefix_backward() -> None:
    """Axis-one gradients concatenate while the shared vector gradient sums."""

    torch.manual_seed(123)
    model = AxisAndSharedModel()
    micro_model = copy.deepcopy(model)
    x = torch.randn(7, 4)
    shared = torch.randn(6, requires_grad=True)
    micro_shared = shared.detach().clone().requires_grad_(True)
    targets = torch.randn(7, 3)
    request = split_request("after:sin", trainable=True, batch_axes={"/args/0": 0})
    full = tl.split.prepare(model, (x, shared), request)
    micro = tl.split.prepare(micro_model, (x, micro_shared), request)
    boundary = full.run_training_prefix(x, shared)
    micro_boundary = micro.run_training_prefix(x, micro_shared)
    assert sorted(tuple(value.shape) for value in boundary.tensors.values()) == [(6,), (6, 7)]
    full_loss, full_grads = full.train_suffix(boundary, targets)
    micro_loss, micro_grads = micro.train_suffix(micro_boundary, targets, microbatch_size=3)
    torch.testing.assert_close(micro_loss, full_loss)
    _assert_gradients(micro_grads, full_grads)
    full.backward_prefix(boundary, full_grads)
    micro.backward_prefix(micro_boundary, micro_grads)
    torch.testing.assert_close(micro_shared.grad, shared.grad)
    for actual, expected in zip(micro_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_residual_and_integer_frontier_matches_full_and_stays_reusable() -> None:
    """Slice every batched crossing, exclude integer gradients, and preserve input."""

    runtime, x, targets = _prepare(ResidualAndIndexModel())
    runtime = runtime.at(tl.split.after("argmax"))
    boundary = runtime.run_prefix(x)
    snapshots = {key: value.clone() for key, value in boundary.tensors.items()}
    integer_keys = {key for key, value in boundary.tensors.items() if not value.is_floating_point()}
    assert len(snapshots) >= 3 and integer_keys
    full_loss, full_grads = runtime.train_suffix(boundary, targets)
    micro_loss, micro_grads = runtime.train_suffix(boundary, targets, microbatch_size=3)
    torch.testing.assert_close(micro_loss, full_loss)
    _assert_gradients(micro_grads, full_grads)
    assert integer_keys.isdisjoint(micro_grads)
    for key, value in boundary.tensors.items():
        torch.testing.assert_close(value, snapshots[key], rtol=0, atol=0)
    torch.testing.assert_close(runtime.run_suffix(boundary), runtime.model(x))


def test_integer_chunk_mutation_does_not_write_caller_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nondifferentiable inputs need storage isolation just like autograd roots."""

    runtime, x, targets = _prepare(ResidualAndIndexModel())
    runtime = runtime.at(tl.split.after("argmax"))
    boundary = runtime.run_prefix(x)
    snapshots = {key: value.clone() for key, value in boundary.tensors.items()}
    execute = runtime.segments.suffix._execute_nodes
    seen: list[int] = []

    def mutate_integer_input(overlay: dict[str, torch.Tensor]) -> None:
        """Model an in-place suffix operation on the chunk's integer crossing."""

        for value in overlay.values():
            if not value.is_floating_point():
                seen.append(len(value))
                value.add_(1)
        execute(overlay)

    monkeypatch.setattr(runtime.segments.suffix, "_execute_nodes", mutate_integer_input)
    runtime.train_suffix(boundary, targets, microbatch_size=3)
    assert seen == [3, 3, 1]
    for key, value in boundary.tensors.items():
        torch.testing.assert_close(value, snapshots[key], atol=0, rtol=0)


def test_nested_target_slicing_preserves_samples_and_namedtuples() -> None:
    """Generic mapping, tuple, namedtuple, and sample-list targets agree numerically."""

    runtime, x, targets = _prepare()
    pair = namedtuple("TargetPair", "values scale")
    values = MappingProxyType(
        {"pair": pair(targets, torch.tensor(1.0)), "samples": list(range(7)), "nested": (targets,)}
    )
    seen: list[list[int]] = []

    def loss_fn(output: torch.Tensor, chunk: Any) -> torch.Tensor:
        """Check container slicing before evaluating an additive objective."""

        assert isinstance(chunk["pair"], pair)
        assert isinstance(chunk["nested"], tuple)
        assert chunk["pair"].scale.ndim == 0
        assert len(chunk["samples"]) == output.shape[0]
        torch.testing.assert_close(chunk["pair"].values, chunk["nested"][0])
        seen.append(chunk["samples"])
        return torch.nn.functional.mse_loss(output, chunk["pair"].values)

    boundary = runtime.run_prefix(x)
    full_loss, full_grads = runtime.train_suffix(boundary, targets)
    loss, grads = runtime.train_suffix(boundary, values, loss_fn=loss_fn, microbatch_size=3)
    assert seen == [[0, 1, 2], [3, 4, 5], [6]]
    torch.testing.assert_close(loss, full_loss)
    _assert_gradients(grads, full_grads)


def test_target_slicer_can_override_nonleading_target_batch_axis() -> None:
    """The task-agnostic callback can override ambiguous automatic tensor slicing."""

    runtime, x, targets = _prepare()
    seen: list[tuple[int, int, int]] = []

    def slice_target(value: torch.Tensor, start: int, end: int, batch: int) -> torch.Tensor:
        """Return a leading-batch target from a transposed storage layout."""

        seen.append((start, end, batch))
        return value[:, start:end].transpose(0, 1)

    boundary = runtime.run_prefix(x)
    full_loss, full_grads = runtime.train_suffix(boundary, targets)
    loss, grads = runtime.train_suffix(
        boundary, targets.T, microbatch_size=3, target_slicer=slice_target
    )
    assert seen == [(0, 3, 7), (3, 6, 7), (6, 7, 7)]
    torch.testing.assert_close(loss, full_loss)
    _assert_gradients(grads, full_grads)


def test_sum_reduction_matches_full_batch_and_optimizer_is_lazy() -> None:
    """Explicit sum losses accumulate unscaled with exactly one lazy Adam update."""

    full, x, targets = _prepare()
    micro_model = ContractMlp()
    micro_model.load_state_dict(full.model.state_dict())
    micro = tl.split.prepare(micro_model, x, split_request("after:relu", trainable=True))
    full_optimizer = CountingAdam(full.model.fc2.parameters())
    micro_optimizer = CountingAdam(micro.model.fc2.parameters())
    boundary = full.run_prefix(x)
    micro_boundary = micro.run_prefix(x)
    assert not full_optimizer.state and not micro_optimizer.state

    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Use an explicitly additive scalar sum."""

        return torch.nn.functional.mse_loss(output, target, reduction="sum")

    full_loss, full_grads = full.train_suffix(
        boundary, targets, loss_fn=loss_fn, optimizer=full_optimizer
    )
    result = micro.train_suffix_result(
        micro_boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=micro_optimizer,
        microbatch_size=3,
        microbatch_reduction="sum",
    )
    assert result.optimizer_applied and result.loss.grad_fn is None
    assert (micro_optimizer.zero_calls, micro_optimizer.step_calls) == (1, 1)
    assert micro_optimizer.state
    torch.testing.assert_close(result.loss, full_loss)
    _assert_gradients(result.boundary_grads, full_grads)
    for actual, expected in zip(
        micro.model.fc2.parameters(), full.model.fc2.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("policy", ["compound", "multiple", "missing", "unresolved", "suffix"])
def test_unsupported_symbolic_shapes_refuse_before_optimizer_side_effects(policy: str) -> None:
    """Never guess slices for unsupported shape proofs or clear existing gradients."""

    runtime, x, targets = _prepare()
    boundary = runtime.run_prefix(x)
    program = runtime.trace_graph.shape_program
    key = next(iter(boundary.tensors))
    shapes = dict(program.value_shapes)
    unresolved = dict(program.unresolved)
    shape = shapes[key]
    if policy == "compound":
        shapes[key] = replace(
            shape,
            dims=(DimExpr("mul", args=(DimExpr.symbol("B"), DimExpr.const(1))), *shape.dims[1:]),
        )
    elif policy == "multiple":
        shapes[key] = replace(shape, dims=(DimExpr.symbol("B"), DimExpr.symbol("B")))
    elif policy == "missing":
        del shapes[key]
    elif policy == "unresolved":
        unresolved[key] = "unsupported boundary relation"
    else:
        suffix_key = next(iter(runtime.plan.suffix_node_ids))
        unresolved[suffix_key] = "unsupported suffix relation"
    runtime.trace_graph = replace(
        runtime.trace_graph,
        shape_program=replace(program, value_shapes=shapes, unresolved=unresolved),
    )
    optimizer = CountingAdam(runtime.model.fc2.parameters())
    sentinel = torch.ones_like(runtime.model.fc2.weight)
    runtime.model.fc2.weight.grad = sentinel
    with pytest.raises(SplitUnsupportedError) as exc:
        runtime.train_suffix(boundary, targets, optimizer=optimizer, microbatch_size=3)
    assert exc.value.context is not None
    assert exc.value.context.reason in {
        "microbatch_boundary_shape_unsupported",
        "microbatch_shape_unsupported",
    }
    assert optimizer.zero_calls == optimizer.step_calls == 0
    assert not optimizer.state
    assert runtime.model.fc2.weight.grad is sentinel


@pytest.mark.parametrize("size", [False, True, 0, -1, 1.5, "3"])
def test_invalid_microbatch_sizes_refuse_before_optimizer_changes(size: Any) -> None:
    """Reject booleans and noninteger sizes rather than truncating or interpreting them."""

    runtime, x, targets = _prepare()
    optimizer = CountingAdam(runtime.model.fc2.parameters())
    with pytest.raises(ValueError, match="positive integer"):
        runtime.train_suffix(
            runtime.run_prefix(x), targets, optimizer=optimizer, microbatch_size=size
        )
    assert optimizer.zero_calls == optimizer.step_calls == 0
    assert not optimizer.state


@pytest.mark.parametrize("backend", ["tf", "paddle", "jax", "tinygrad", "mlx"])
def test_preview_backends_explicitly_refuse_microbatch_policy(backend: str) -> None:
    """No preview engine receives a request it might silently execute full-batch."""

    runtime = SimpleNamespace(
        adapter=SimpleNamespace(name=backend), request=SimpleNamespace(boundary="cut")
    )
    with pytest.raises(SplitUnsupportedError) as exc:
        train_suffix_result(runtime, None, None, microbatch_size=3)
    assert exc.value.context.backend == backend
    assert "microbatch" in exc.value.context.reason


def test_stale_boundary_identity_refuses_before_optimizer_changes() -> None:
    """Microbatch execution retains the existing semantic boundary identity checks."""

    runtime, x, targets = _prepare()
    boundary = runtime.run_prefix(x)
    stale = replace(boundary, metadata={**boundary.metadata, "split_id": "stale"})
    optimizer = CountingAdam(runtime.model.fc2.parameters())
    with pytest.raises(SplitBoundaryError):
        runtime.train_suffix(stale, targets, optimizer=optimizer, microbatch_size=3)
    assert optimizer.zero_calls == optimizer.step_calls == 0


def test_saved_boundary_can_change_microbatch_policy_without_recapture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache identity, graph, and shape program remain independent of chunk policy."""

    runtime, x, targets = _prepare()
    runtime.save_boundary(runtime.run_prefix(x), tmp_path / "boundary")
    boundary = runtime.load_boundary(tmp_path / "boundary")
    graph = runtime.graph_ir
    program = runtime.trace_graph.shape_program
    identity = runtime.graph_identity, runtime.split_id, program.fingerprint
    spec = dict(boundary.spec)
    metadata = copy.deepcopy(boundary.metadata)

    def forbidden_capture(*args: Any, **kwargs: Any) -> Any:
        """Fail if training tries to trace or prepare again."""

        raise AssertionError("microbatch training must reuse capture")

    monkeypatch.setattr(tl.split, "prepare", forbidden_capture)
    full_loss, full_grads = runtime.train_suffix(boundary, targets)
    for size in (1, 3, 4, 8):
        result = runtime.train_suffix_result(boundary, targets, microbatch_size=size)
        torch.testing.assert_close(result.loss, full_loss)
        _assert_gradients(result.boundary_grads, full_grads)
        assert runtime.graph_ir is graph and runtime.trace_graph.shape_program is program
        assert (runtime.graph_identity, runtime.split_id, program.fingerprint) == identity
        assert boundary.spec == spec and boundary.metadata == metadata


def test_previous_suffix_graph_is_released_before_next_microbatch() -> None:
    """Only detached scalar losses and gradients escape each chunk's backward."""

    runtime, x, targets = _prepare()
    previous: list[weakref.ReferenceType[torch.Tensor]] = []
    seen: list[int] = []

    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Observe Python output/loss references at successive chunk boundaries."""

        assert all(reference() is None for reference in previous)
        loss = torch.nn.functional.mse_loss(output, target)
        previous[:] = [weakref.ref(output), weakref.ref(loss)]
        seen.append(output.shape[0])
        return loss

    result = runtime.train_suffix_result(
        runtime.run_prefix(x), targets, loss_fn=loss_fn, microbatch_size=3
    )
    assert seen == [3, 3, 1]
    assert all(reference() is None for reference in previous)
    assert result.loss.grad_fn is None and not result.loss.requires_grad


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for heterogeneous parity"
)
def test_cpu_prefix_cuda_suffix_microbatch_gradient_handoff() -> None:
    """Transport once to CUDA and return complete gradients for one CPU backward."""

    runtime, x, targets = _prepare()
    reference = ContractMlp()
    reference.load_state_dict(runtime.model.state_dict())
    placement = tl.split.PlacementPlan.across("cpu", "cuda:0")
    placed = runtime.with_placement(placement)
    boundary = placed.run_training_prefix(x)
    assert all(value.device.type == "cpu" for value in boundary.tensors.values())
    suffix_parameters = placed.suffix_parameters()
    optimizer = torch.optim.SGD(suffix_parameters, lr=0.03)
    prefix_parameters = placed.prefix_parameters() or list(runtime.model.fc1.parameters())
    prefix_optimizer = torch.optim.SGD(prefix_parameters, lr=0.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = torch.nn.functional.mse_loss(reference(x), targets)
    reference_loss.backward()
    reference_optimizer.step()
    loss, grads = placed.train_suffix(
        boundary, targets.cuda(), optimizer=optimizer, microbatch_size=3
    )
    assert all(value.device.type == "cuda" for value in grads.values())
    placed.backward_prefix(boundary, grads, optimizer=prefix_optimizer)
    torch.testing.assert_close(loss.cpu(), reference_loss, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(prefix_parameters, reference.fc1.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(suffix_parameters, reference.fc2.parameters(), strict=True):
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
