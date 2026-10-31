"""Frozen state crossing a DAG boundary must not become an autograd root."""

from __future__ import annotations

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from v2_helpers import (
    prepare_fixture,
    request_fixture,
    select_node_prefix,
)

from torchlens.split import GraphPartition, SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("microbatch", [None, 2])
@pytest.mark.parametrize("retain_trace", [False, True])
def test_eval_batchnorm_state_stays_frozen_across_partition_and_recut(
    device: str, microbatch: int | None, retain_trace: bool
) -> None:
    """Compare two momentum updates with BatchNorm buffers in the frontier."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for native cuDNN BatchNorm regression coverage")
    torch.manual_seed(902)
    model = (
        nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Linear(4, 2),
        )
        .to(device)
        .eval()
    )
    reference = deepcopy(model)
    runtime = prepare_fixture(
        model,
        torch.randn(4, 4, device=device),
        request_fixture(
            partial(select_node_prefix, node_id="2"),
            features=SplitFeatures(training=True, retain_trace=retain_trace),
        ),
    )
    selected = frozenset(
        node.canonical_id
        for node in runtime.trace_graph.compute_nodes
        if node.canonical_id in runtime.plan.prefix_node_ids
    )
    partition = GraphPartition(selected)
    runtime = runtime.repartition(partition)
    buffer_keys = {
        key
        for key, node_id in runtime.plan.boundary_bindings.items()
        if runtime.trace_graph.node_by_id[node_id].is_buffer
    }
    assert buffer_keys, "This regression requires frozen running statistics in the frontier"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    for _ in range(2):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        targets = torch.randn(4, 2, device=device)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        expected_loss = nn.functional.mse_loss(reference(expected_x), targets)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, targets, microbatch_size=microbatch)
        assert not buffer_keys.intersection(gradients)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, expected_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(model.buffers(), reference.buffers(), strict=True):
            torch.testing.assert_close(actual, expected)
        runtime = runtime.repartition(partition)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("microbatch", [None, 2])
@pytest.mark.parametrize("retain_trace", [False, True])
def test_training_batchnorm_updates_current_state_through_repartition(
    device: str,
    microbatch: int | None,
    retain_trace: bool,
) -> None:
    """Native suffix buffer updates persist for full and sequential chunk training."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("A real CUDA device is required")
    torch.manual_seed(41)
    model = (
        nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.BatchNorm2d(4), nn.Tanh())
        .to(device)
        .double()
        .train()
    )
    reference = deepcopy(model)
    graph = prepare_graph(
        model,
        torch.randn(4, 4, 2, 2, device=device, dtype=torch.float64),
        features=SplitFeatures(training=True, retain_trace=retain_trace),
    )
    first = GraphPartition(frozenset({graph.trace_graph.compute_nodes[0].canonical_id}))
    runtime = graph.materialize(graph.analyze(first))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    for step, partition in enumerate((first, GraphPartition(frozenset()), first), 1):
        runtime = runtime.repartition(partition)
        optimizer.zero_grad(set_to_none=True)
        expected_optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4, 4, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        targets = torch.randn_like(x)
        size = microbatch or 4
        expected_loss = sum(
            nn.functional.mse_loss(
                reference(expected_x[start : start + size]), targets[start : start + size]
            )
            * size
            / 4
            for start in range(0, 4, size)
        )
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, targets, microbatch_size=microbatch)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, expected_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(model.state_dict(), reference.state_dict())
        assert model[1].num_batches_tracked.item() == step * (4 // size)


def test_mutable_buffer_training_refuses_disconnected_boundary() -> None:
    """A detached buffer snapshot cannot silently discard authoritative state writes."""
    model = nn.Sequential(nn.BatchNorm2d(4), nn.Tanh()).train()
    x = torch.randn(2, 4, 2, 2)
    graph = prepare_graph(model, x, features=SplitFeatures(training=True, retain_trace=False))
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    before = deepcopy(model.state_dict())
    boundary = runtime.run_prefix(x)
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.train_suffix(boundary, torch.zeros_like(x))
    torch.testing.assert_close(model.state_dict(), before)


def test_readonly_buffer_operand_does_not_require_connected_training_boundary() -> None:
    """A local activation write may read frozen state without mutating that state."""

    class Offset(nn.Module):
        """Add a read-only offset into a fresh activation allocation."""

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("offset", torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            value = x.clone()
            value.add_(self.offset)
            return value.tanh()

    model = Offset()
    x = torch.randn(2, 4)
    graph = prepare_graph(model, x, features=SplitFeatures(training=True, retain_trace=False))
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    reference_x = x.detach().clone().requires_grad_(True)
    target = torch.zeros_like(x)
    expected_loss = nn.functional.mse_loss(model(reference_x), target)
    expected_loss.backward()
    boundary = runtime.run_prefix(x)
    loss, gradients = runtime.train_suffix(boundary, target)
    torch.testing.assert_close(loss, expected_loss)
    input_key = next(
        key
        for key, value in runtime.plan.boundary_bindings.items()
        if value in graph.trace_graph.input_node_ids
    )
    torch.testing.assert_close(gradients[input_key], reference_x.grad)
    torch.testing.assert_close(model.offset, torch.ones(4))
