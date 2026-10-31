"""Ordered native writes need fresh local storage and ordinary tensor dispatch."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any

import pytest
import torch
from torch import nn
from v2_helpers import (
    capture_fixture,
    prepare_fixture,
    request_fixture,
    select_node_prefix,
)

from torchlens.split import GraphPartition, SplitFeatures
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
    runtime = prepare_fixture(
        model,
        torch.randn(2, 4),
        request_fixture(
            partial(select_node_prefix, node_id="activation"),
            features=SplitFeatures(training=True, retain_trace=False),
        ),
    )
    selected = frozenset(
        n.canonical_id
        for n in runtime.trace_graph.compute_nodes
        if n.module_path in {"first", "activation"}
    )
    partition = GraphPartition(selected)
    runtime = runtime.repartition(partition)
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
        runtime = runtime.repartition(partition)


def test_boundary_between_fresh_producer_and_write_refuses() -> None:
    """A fresh allocation in another segment becomes a boundary root, not local storage."""
    runtime = prepare_fixture(
        ResidualWrites("relu"),
        torch.ones(2, 4),
        request_fixture(
            partial(select_node_prefix, node_id="activation"), features=SplitFeatures(training=True)
        ),
    )
    selected = frozenset(
        n.canonical_id for n in runtime.trace_graph.compute_nodes if n.module_path == "first"
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.repartition(GraphPartition(selected))
    assert caught.value.context.reason == "partition_effect_order_unproven"


def test_ordered_view_write_preserves_fresh_storage() -> None:
    """An ordered view write retains the proven fresh allocation and native alias behavior."""

    class ViewWrite(nn.Module):
        """Write through a view of an allocated tensor."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Leave alias obligations outside the narrow fresh-producer proof."""
            fresh = x + 1
            return fresh.view_as(x).relu_()

    runtime = prepare_fixture(
        ViewWrite(), torch.ones(2, 4), request_fixture(partial(select_node_prefix, node_id="relu"))
    )
    cut = runtime.repartition(
        GraphPartition(frozenset(n.canonical_id for n in runtime.trace_graph.compute_nodes))
    )
    x = torch.randn(2, 4)
    torch.testing.assert_close(cut.replay(x), ViewWrite()(x))


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
    if not local_normalization:
        graph = capture_fixture(
            model,
            torch.randn(2, 3, 4),
            request_fixture(
                partial(select_node_prefix, node_id="activation"),
                features=SplitFeatures(training=True, retain_trace=False),
            ),
        )
        with pytest.raises(SplitUnsupportedError):
            graph.analyze(
                GraphPartition(frozenset(n.canonical_id for n in graph.trace_graph.compute_nodes))
            )
        return
    runtime = prepare_fixture(
        model,
        torch.randn(2, 3, 4),
        request_fixture(
            partial(select_node_prefix, node_id="activation"),
            features=SplitFeatures(training=True, retain_trace=False),
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
            runtime.repartition(partition)
        return
    cut = runtime.repartition(partition)
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


def test_partition_reorders_only_data_independent_local_writes() -> None:
    """Fresh storage and independent reads license this branch reorder."""

    class Interleaved(nn.Module):
        """Keep an independent branch between allocation and mutation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Capture two fresh values before mutating the first."""
            left = x + 1
            right = x * 2
            return left.relu_() + right

    runtime = prepare_fixture(
        Interleaved(),
        torch.ones(2, 4),
        request_fixture(partial(select_node_prefix, node_id="relu")),
    )
    nodes = runtime.trace_graph.compute_nodes
    selected = frozenset(n.canonical_id for n in nodes if n.op_type == "relu" or n is nodes[0])
    cut = runtime.repartition(GraphPartition(selected))
    x = torch.randn(2, 4)
    torch.testing.assert_close(cut.replay(x), Interleaved()(x))


@pytest.mark.parametrize(
    ("independent_after_write", "selected_indexes", "supported"),
    [
        (False, (), True),
        (False, (1,), True),
        (False, (0,), False),
        (False, (0, 2), True),
        (False, (0, 1, 2), True),
        (True, (2,), True),
        (True, (0, 1), True),
    ],
)
def test_write_barrier_distinguishes_storage_dependencies(
    independent_after_write: bool, selected_indexes: tuple[int, ...], supported: bool
) -> None:
    """Independent storage may move; floating autograd boundary writes remain refused."""

    class Interleaved(nn.Module):
        """Place an independent read between fresh allocation and its write."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the written branch and the independent branch together."""
            left = x + 1
            if independent_after_write:
                written = left.relu_()
                return written + x * 2
            right = x * 2
            return left.relu_() + right

    model = Interleaved()
    runtime = prepare_fixture(
        model,
        torch.ones(2, 4),
        request_fixture(
            partial(select_node_prefix, node_id="relu"), features=SplitFeatures(training=True)
        ),
    )
    nodes = runtime.trace_graph.compute_nodes
    assert len(nodes) == 4
    partition = GraphPartition(frozenset(nodes[index].canonical_id for index in selected_indexes))
    if not supported:
        with pytest.raises(SplitUnsupportedError) as caught:
            runtime.repartition(partition)
        assert caught.value.context.reason == "partition_effect_order_unproven"
        return

    cut = runtime.repartition(partition)
    write_index = 1 if independent_after_write else 2
    assert cut.plan.ordered_mutation_node_ids == frozenset({nodes[write_index].canonical_id})
    x = torch.randn(2, 4)
    torch.testing.assert_close(cut.replay(x), model(x))


def test_eval_batchnorm_fresh_output_can_be_mutated() -> None:
    """Native eval BatchNorm allocates output without updating its running buffers."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(inplace=True), nn.Conv2d(4, 2, 1)
    ).eval()
    x = torch.randn(2, 3, 4, 4)
    runtime = prepare_fixture(model, x, request_fixture(partial(select_node_prefix, node_id="2")))
    selected = frozenset(
        n.canonical_id
        for n in runtime.trace_graph.compute_nodes
        if n.canonical_id in runtime.plan.prefix_node_ids
    )
    cut = runtime.repartition(GraphPartition(selected))
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
    runtime = prepare_fixture(
        model, x, request_fixture(partial(select_node_prefix, node_id="activation"))
    )
    runtime = runtime.repartition(GraphPartition(frozenset()))
    with Passthrough(), pytest.raises(SplitUnsupportedError) as caught:
        runtime.replay(x)
    assert caught.value.context.reason == "partition_mutation_dispatch_unsupported"
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.replay(x.as_subclass(Subclass))
    assert caught.value.context.reason == "partition_mutation_dispatch_unsupported"
