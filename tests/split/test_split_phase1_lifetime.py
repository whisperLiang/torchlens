"""Observe actual autograd and reference lifetimes in Phase 1 split execution."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan
from torchlens.split.boundary import ReplayBoundary
from torchlens.split.state import SegmentState


class LifetimeMlp(nn.Module):
    """Use saved-tensor operations in both independently executable segments."""

    def __init__(self) -> None:
        """Build an affine prefix and nonlinear suffix around a named cut."""

        super().__init__()
        self.stem = nn.Linear(4, 6)
        self.cut = nn.Sigmoid()
        self.head = nn.Linear(6, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate both segments with gradients enabled by the caller."""

        return self.head(self.cut(self.stem(value))).tanh()


@dataclass
class ObservedSegment:
    """Observe calls while preserving the segment's state-identity accessors."""

    segment: Any
    observer: Callable[[ReplayBoundary], Any]

    def __getattr__(self, name: str) -> Any:
        """Forward state and shape inspection to the original segment."""

        return getattr(self.segment, name)

    def __call__(self, boundary: ReplayBoundary) -> Any:
        """Run the observer around the original segment."""

        return self.observer(boundary)


@pytest.mark.parametrize("training", [False, True])
def test_prefix_grad_policy_controls_execution_and_saved_tensors(
    training: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detached execution saves nothing, while training preserves prefix derivatives."""

    model = LifetimeMlp()
    inputs = torch.randn(3, 4, requires_grad=True)
    runtime = tl.split.prepare(model, inputs, split_request("after:cut", trainable=True))
    segment = runtime.segments.training_prefix if training else runtime.segments.prefix
    execute = segment._execute_func
    grad_modes: list[bool] = []
    saved: list[weakref.ReferenceType[torch.Tensor]] = []

    def observe(node: Any, args: Any, kwargs: Any) -> Any:
        """Record grad mode at the actual replay-kernel boundary."""

        grad_modes.append(torch.is_grad_enabled())
        return execute(node, args, kwargs)

    def pack(value: torch.Tensor) -> torch.Tensor:
        """Count autograd saves without extending tensor lifetime in the observer."""

        saved.append(weakref.ref(value))
        return value

    monkeypatch.setattr(segment, "_execute_func", observe)
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        boundary = runtime.run_training_prefix(inputs) if training else runtime.run_prefix(inputs)
        assert torch.is_grad_enabled()
    assert grad_modes and all(mode is training for mode in grad_modes)
    if training:
        assert saved
        actual = torch.autograd.grad(
            sum(value.sum() for value in boundary.tensors.values()), inputs
        )
        expected = torch.autograd.grad(model.cut(model.stem(inputs)).sum(), inputs)
        torch.testing.assert_close(actual, expected)
    else:
        assert saved == []
        assert all(value.grad_fn is None for value in boundary.tensors.values())
        # A detached prefix remains a valid source for trainable suffix roots.
        loss, gradients = runtime.train_suffix(boundary, torch.zeros(3, 3))
        assert torch.isfinite(loss) and gradients
        assert model.head.weight.grad is not None
        assert model.stem.weight.grad is None


def test_public_suffix_preserves_reusable_boundary() -> None:
    """Two ordinary suffix calls neither consume nor alter their borrowed boundary."""

    model = LifetimeMlp()
    inputs = torch.randn(3, 4)
    runtime = tl.split.prepare(model, inputs, split_request("after:cut", trainable=True))
    boundary = runtime.run_prefix(inputs)
    tensors = dict(boundary.tensors)
    snapshots = {key: value.clone() for key, value in tensors.items()}
    versions = {key: value._version for key, value in tensors.items()}
    metadata = dict(boundary.metadata)
    expected = model(inputs)
    torch.testing.assert_close(runtime.run_suffix(boundary), expected)
    torch.testing.assert_close(runtime.run_suffix(boundary), expected)
    assert boundary.metadata == metadata
    assert boundary.tensors.keys() == tensors.keys()
    for key, value in boundary.tensors.items():
        assert value is tensors[key] and value._version == versions[key]
        torch.testing.assert_close(value, snapshots[key], atol=0, rtol=0)


def test_internal_replay_releases_source_boundary_before_transported_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU cloning simulates transport and exposes source ownership deterministically."""

    model = LifetimeMlp()
    inputs = torch.randn(3, 4)
    runtime = tl.split.prepare(model, inputs, split_request("after:cut", trainable=True))
    run_prefix = runtime._run_prefix
    suffix = runtime.segments.suffix
    source_refs: list[weakref.ReferenceType[Any]] = []
    checked = False

    def observe_prefix(*args: Any, **kwargs: Any) -> ReplayBoundary:
        """Track the owned boundary and each source tensor without retaining them."""

        result = run_prefix(*args, **kwargs)
        source_refs.append(weakref.ref(result))
        source_refs.extend(weakref.ref(value) for value in result.tensors.values())
        return result

    def copy_transport(boundary: ReplayBoundary, placement: Any) -> ReplayBoundary:
        """Allocate destination storage on CPU without borrowing source tensors."""

        del placement
        return ReplayBoundary(
            boundary.backend,
            {key: value.detach().clone() for key, value in boundary.tensors.items()},
            boundary.spec,
            dict(boundary.metadata),
        )

    def observe_suffix(boundary: ReplayBoundary) -> Any:
        """Source objects must already be dead when destination replay starts."""

        nonlocal checked
        assert len(source_refs) > 1 and all(ref() is None for ref in source_refs)
        checked = True
        return suffix(boundary)

    monkeypatch.setattr(runtime, "_run_prefix", observe_prefix)
    monkeypatch.setattr(runtime, "_transport_boundary", copy_transport)
    monkeypatch.setattr(
        runtime,
        "segments",
        replace(runtime.segments, suffix=ObservedSegment(suffix, observe_suffix)),
    )
    torch.testing.assert_close(runtime.replay(inputs), model(inputs))
    assert checked


@dataclass
class SavedPayload:
    """Weak-referenceable witness owned only by an autograd saved-tensor slot."""

    value: torch.Tensor


def test_microbatch_releases_graph_roots_and_saved_payloads_before_next_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each 3+3+1 chunk frees its graph and roots before the next suffix call."""

    model = LifetimeMlp()
    inputs = torch.randn(7, 4)
    runtime = tl.split.prepare(model, inputs, split_request("after:cut", trainable=True))
    boundary = runtime.run_training_prefix(inputs)
    suffix = runtime.segments.suffix
    tensor_refs: list[weakref.ReferenceType[Any]] = []
    saved_refs: list[weakref.ReferenceType[SavedPayload]] = []
    sizes: list[int] = []

    def pack(value: torch.Tensor) -> SavedPayload:
        """Give each saved autograd slot its own observable lifetime witness."""

        payload = SavedPayload(value)
        saved_refs.append(weakref.ref(payload))
        return payload

    def observe_suffix(chunk: ReplayBoundary) -> torch.Tensor:
        """The prior chunk must be collectible immediately, without cyclic GC."""

        assert all(ref() is None for ref in tensor_refs)
        assert all(ref() is None for ref in saved_refs)
        sizes.append(chunk.metadata["runtime_batch_size"])
        tensor_refs.append(weakref.ref(chunk))
        tensor_refs.extend(weakref.ref(value) for value in chunk.tensors.values())
        output = suffix(chunk)
        tensor_refs.append(weakref.ref(output))
        return output

    def loss_fn(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Observe the per-chunk scalar before it is consumed by backward."""

        loss = torch.nn.functional.mse_loss(output, targets)
        tensor_refs.append(weakref.ref(loss))
        return loss

    monkeypatch.setattr(
        runtime,
        "segments",
        replace(runtime.segments, suffix=ObservedSegment(suffix, observe_suffix)),
    )
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda payload: payload.value):
        result = runtime.train_suffix_result(
            boundary,
            torch.randn(7, 3),
            loss_fn=loss_fn,
            microbatch_size=3,
        )
    assert sizes == [3, 3, 1] and saved_refs
    assert all(ref() is None for ref in tensor_refs)
    assert all(ref() is None for ref in saved_refs)
    assert result.loss.grad_fn is None
    assert all(value.grad_fn is None for value in result.boundary_grads.values())
    runtime.backward_prefix(boundary, result.boundary_grads)
    assert model.stem.weight.grad is not None


class MultioutputBranches(nn.Module):
    """Hold two chunk views and a residual until their distinct final consumers."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Join independent branches before two additional allocating operations."""

        left, right = value.chunk(2, dim=1)
        residual = left.sin()
        branch = right.cos()
        return (residual + branch).square().sin()


def test_multioutput_and_residual_values_die_after_their_final_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A view survives its own consumer; the residual survives until the join."""

    model = MultioutputBranches()
    inputs = torch.randn(3, 8)
    seed = tl.split.prepare(model, inputs, split_request("50%"))
    runtime = seed.at(tl.split.before(seed.trace_graph.compute_nodes[0].canonical_id))
    segment = runtime.segments.suffix
    execute = segment._execute_func
    views: list[weakref.ReferenceType[torch.Tensor]] = []
    residual: list[weakref.ReferenceType[torch.Tensor]] = []
    checked: list[str] = []

    def observe(node: Any, args: Any, kwargs: Any) -> Any:
        """Check lifetime before each consuming kernel can create temporary references."""

        if node.op_type == "cos":
            assert views[0]() is None and views[1]() is not None
            assert residual[0]() is not None
            checked.append("one_view_released")
        elif node.op_type == "add":
            assert all(ref() is None for ref in views)
            assert residual[0]() is not None
            checked.append("residual_retained")
        elif node.op_type == "square":
            assert residual[0]() is None
            checked.append("residual_released")
        output = execute(node, args, kwargs)
        if node.op_type == "chunk":
            views.extend(weakref.ref(value) for value in output)
        elif node.op_type == "sin" and not residual:
            residual.append(weakref.ref(output))
        return output

    monkeypatch.setattr(segment, "_execute_func", observe)
    with torch.no_grad():
        torch.testing.assert_close(runtime.replay(inputs), model(inputs))
    assert checked == ["one_view_released", "residual_retained", "residual_released"]


def test_frozen_embedding_with_forward_mutation_keeps_independent_segment_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen embedding max-norm state is mutable despite not requiring gradients."""

    class FrozenEmbedding(nn.Module):
        """Reuse a renormalizing embedding on both sides of the boundary."""

        def __init__(self) -> None:
            """Freeze gradient ownership without disabling forward renormalization."""

            super().__init__()
            self.shared = nn.Embedding(5, 4, max_norm=1.0)
            self.shared.weight.requires_grad_(False)
            self.cut = nn.ReLU()

        def forward(self, indexes: torch.Tensor) -> torch.Tensor:
            """Consume the mutating weight before and after the cut."""

            first = self.cut(self.shared(indexes))
            return first + self.shared(indexes)

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = FrozenEmbedding()
    inputs = torch.tensor([[0, 1], [1, 2], [2, 3]])
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:cut", trainable=True, placement=PlacementPlan.on("cpu")),
    )
    # Let the production adapter decide eligibility before observing cached state.
    runtime.segments.prefix.bound_state_values()
    runtime.segments.suffix.bound_state_values()
    prefix_weight = runtime.segments.prefix._state.resolve(model.shared.weight)
    suffix_weight = runtime.segments.suffix._state.resolve(model.shared.weight)
    assert prefix_weight is not suffix_weight
    assert prefix_weight.data_ptr() != suffix_weight.data_ptr()
    original_prefix = prefix_weight.clone()
    suffix_weight.mul_(5)
    torch.testing.assert_close(prefix_weight, original_prefix, atol=0, rtol=0)
