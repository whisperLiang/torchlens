"""Ordered native writes need fresh local storage and ordinary tensor dispatch."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, SplitRequest, after, prepare
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.planner import _validate_partition_effects


class ResidualWrites(nn.Module):
    """Two fresh projections with native activation and residual writes."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True)
        self.second = nn.Linear(4, 4)
        self.final = nn.ReLU(inplace=True)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate only fresh local values, preserving the residual source."""
        left = self.activation(self.first(x))
        merged = self.second(left)
        merged += left
        return self.head(self.final(merged))


@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_ordered_fresh_writes_match_gradients_and_adam(activation: str) -> None:
    """A native write chain executes without rewriting the model's inplace flags."""
    torch.manual_seed(77)
    reference = ResidualWrites(activation)
    model = deepcopy(reference)
    runtime = prepare(
        model,
        torch.randn(2, 4),
        SplitRequest(
            after("activation"), features=SplitFeatures(training=True, retain_trace=False)
        ),
    )
    selected = frozenset(
        n.canonical_id
        for n in runtime.trace_graph.compute_nodes
        if n.module_path in {"first", "activation"}
    )
    partition = GraphPartition(selected)
    runtime = runtime.partition(partition)
    assert len(runtime.plan.ordered_mutation_node_ids) == 3
    assert model.activation.inplace and model.final.inplace
    actual_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
    for batch in (2, 4):
        x = torch.randn(batch, 4, requires_grad=True)
        actual_x = x.detach().clone().requires_grad_(True)
        target = torch.randn(batch, 2)
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        expected = reference(x)
        expected_loss = nn.functional.mse_loss(expected, target)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(actual_x)
        actual_loss, gradients = runtime.train_suffix(boundary, target)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_x.grad, x.grad)
        for actual, expected_parameter in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, expected_parameter.grad)
        actual_optimizer.step()
        expected_optimizer.step()
        for actual, expected_parameter in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected_parameter)
        runtime = runtime.partition(partition)


def test_boundary_between_fresh_producer_and_write_refuses() -> None:
    """A fresh allocation in another segment becomes a boundary root, not local storage."""
    runtime = prepare(ResidualWrites("relu"), torch.ones(2, 4), SplitRequest(after("activation")))
    selected = frozenset(
        n.canonical_id for n in runtime.trace_graph.compute_nodes if n.module_path == "first"
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.partition(GraphPartition(selected))
    assert caught.value.context.reason == "partition_effect_order_unproven"


def test_write_through_view_refuses_even_with_local_allocation() -> None:
    """Alias-producing view operations cannot serve as certified write producers."""

    class ViewWrite(nn.Module):
        """Write through a view of an allocated tensor."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Leave alias obligations outside the narrow fresh-producer proof."""
            fresh = x + 1
            return fresh.view_as(x).relu_()

    runtime = prepare(ViewWrite(), torch.ones(2, 4), SplitRequest(after("relu")))
    with pytest.raises(SplitUnsupportedError):
        runtime.partition(
            GraphPartition(frozenset(n.canonical_id for n in runtime.trace_graph.compute_nodes))
        )


@pytest.mark.parametrize("local_normalization", [False, True])
def test_permuted_write_requires_fresh_local_storage(local_normalization: bool) -> None:
    """RF-DETR's LayerNorm/permutation/SiLU chain has a provable local origin."""

    class PermutedWrite(nn.Module):
        """Mutate a permutation with either fresh normalized or caller storage."""

        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(4) if local_normalization else nn.Identity()
            self.activation = nn.SiLU(inplace=True)
            self.head = nn.Linear(12, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Keep the native activation and noncontiguous permutation intact."""
            return self.head(self.activation(self.norm(x).permute(0, 2, 1)).flatten(1))

    model = PermutedWrite()
    reference = deepcopy(model)
    runtime = prepare(
        model,
        torch.randn(2, 3, 4),
        SplitRequest(
            after("activation"), features=SplitFeatures(training=True, retain_trace=False)
        ),
    )
    partition = GraphPartition(
        frozenset(
            node.canonical_id
            for node in runtime.trace_graph.compute_nodes
            if node.canonical_id in runtime.plan.prefix_node_ids
        )
    )
    if not local_normalization:
        with pytest.raises(SplitUnsupportedError):
            runtime.partition(partition)
        return
    cut = runtime.partition(partition)
    x = torch.randn(2, 3, 4, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_(True)
    target = torch.randn(2, 2)
    expected_loss = nn.functional.mse_loss(reference(expected_x), target)
    expected_loss.backward()
    boundary = cut.run_training_prefix(x)
    loss, gradients = cut.train_suffix(boundary, target)
    cut.backward_prefix(boundary, gradients)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(x.grad, expected_x.grad)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_partition_cannot_reorder_independent_computation_across_write() -> None:
    """Data closure alone does not allow an independent node to cross a mutation barrier."""

    class Interleaved(nn.Module):
        """Keep an independent branch between allocation and mutation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Capture two fresh values before mutating the first."""
            left = x + 1
            right = x * 2
            return left.relu_() + right

    runtime = prepare(Interleaved(), torch.ones(2, 4), SplitRequest(after("relu")))
    nodes = runtime.trace_graph.compute_nodes
    selected = frozenset(n.canonical_id for n in nodes if n.op_type == "relu" or n is nodes[0])
    with pytest.raises(SplitUnsupportedError):
        runtime.partition(GraphPartition(selected))


def test_eval_batchnorm_fresh_output_can_be_mutated() -> None:
    """Native eval BatchNorm allocates output without updating its running buffers."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(inplace=True), nn.Conv2d(4, 2, 1)
    ).eval()
    x = torch.randn(2, 3, 4, 4)
    runtime = prepare(model, x, SplitRequest(after("2")))
    selected = frozenset(
        n.canonical_id
        for n in runtime.trace_graph.compute_nodes
        if n.canonical_id in runtime.plan.prefix_node_ids
    )
    cut = runtime.partition(GraphPartition(selected))
    torch.testing.assert_close(cut.replay(x), model(x))
    with pytest.raises(SplitUnsupportedError):
        _validate_partition_effects(cut.trace_graph, cut.request)


def test_certified_mutations_refuse_dispatch_modes_and_subclasses() -> None:
    """Runtime tensor overrides cannot invalidate the planner's allocation proof."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Passthrough(TorchDispatchMode):
        """Even a benign mode is outside certified ordinary tensor semantics."""

        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            return func(*args, **(kwargs or {}))

    class Subclass(torch.Tensor):
        """A tensor subclass is not a certified native allocation operand."""

    model = ResidualWrites("relu")
    x = torch.ones(2, 4)
    runtime = prepare(model, x, SplitRequest(after("activation")))
    runtime = runtime.partition(GraphPartition(frozenset()))
    with Passthrough(), pytest.raises(SplitUnsupportedError) as caught:
        runtime.replay(x)
    assert caught.value.context.reason == "partition_mutation_dispatch_unsupported"
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.replay(x.as_subclass(Subclass))
    assert caught.value.context.reason == "partition_mutation_dispatch_unsupported"
