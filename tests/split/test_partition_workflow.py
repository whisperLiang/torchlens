"""Capture first, then plan arbitrary partitions without executable dummy segments."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

import torchlens.split as split


def test_capture_first_then_choose_noncontiguous_partition() -> None:
    """Independent branches can execute across a noncontiguous cut after one capture."""

    class Branches(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            left = x.sin()
            right = x.cos()
            return left.relu() + right

    model = Branches()
    graph = split.prepare_graph(model, torch.ones(2, 4))
    nodes = graph.trace_graph.compute_nodes
    partition = split.GraphPartition(
        frozenset(n.canonical_id for n in nodes if n.op_type in {"sin", "relu"})
    )
    assert partition.prefix_node_ids != frozenset(n.canonical_id for n in nodes[:2])
    with patch.object(graph.adapter, "build_segments", wraps=graph.adapter.build_segments) as build:
        analysis = graph.analyze(partition)
        assert build.call_count == 0
        runtime = graph.materialize(analysis)
        assert build.call_count == 1
    assert runtime.request.partition == partition
    for batch in (1, 2, 4, 8):
        x = torch.randn(batch, 4)
        torch.testing.assert_close(runtime.replay(x), model(x))


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
def test_repartition_reuses_current_parameters_and_optimizer_history(optimizer_type: type) -> None:
    """A-B-A recuts preserve updates and caller-owned optimizer state by tensor identity."""
    torch.manual_seed(41)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    reference = deepcopy(model)
    graph = split.prepare_graph(
        model, torch.ones(2, 4), features=split.SplitFeatures(training=True, retain_trace=False)
    )
    ids = [n.canonical_id for n in graph.trace_graph.compute_nodes]
    partitions = [split.GraphPartition(frozenset(ids[:k])) for k in (1, 2, 1)]
    runtime = graph.materialize(graph.analyze(partitions[0]))
    optimizer = optimizer_type(model.parameters(), lr=0.01)
    oracle = optimizer_type(reference.parameters(), lr=0.01)
    for partition in partitions:
        runtime = runtime.repartition(partition)
        assert {
            id(value)
            for owners in runtime.parameter_bindings().values()
            for value in owners.values()
        } == set(map(id, model.parameters()))
        x = torch.randn(2, 4, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)
        target = torch.randn(2, 2)
        optimizer.zero_grad()
        oracle.zero_grad()
        expected_loss = nn.functional.mse_loss(reference(ref_x), target)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, grads = runtime.train_suffix(boundary, target)
        runtime.backward_prefix(boundary, grads)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, ref_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        oracle.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
            for key, value in optimizer.state[actual].items():
                torch.testing.assert_close(value, oracle.state[expected][key])
