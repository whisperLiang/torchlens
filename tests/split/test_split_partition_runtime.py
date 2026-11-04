"""DAG runtime integration and numerical training contracts."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from torchlens.split import SplitFeatures, prepare_graph

pytestmark = pytest.mark.usefixtures("fp64_functional")


class Branches(nn.Module):
    """Two independent branches captured in an interleaved order."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.left_end = nn.ReLU()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a diamond merge with a direct input skip."""
        left = self.left(x)
        right = self.right(x)
        return self.head(self.left_end(left) + right + x)


@pytest.mark.parametrize("optimizer_kind", ["sgd", "momentum", "adam"])
def test_noncontiguous_partition_training_and_recut(optimizer_kind: str) -> None:
    """Compare outputs, input/parameter gradients and continued optimizer state."""
    torch.manual_seed(23)
    reference = Branches()
    model = deepcopy(reference)
    sample = torch.randn(2, 4)
    graph = prepare_graph(
        model,
        sample,
        features=SplitFeatures(training=True, retain_trace=False),
    )
    dag = graph.compute_dag()
    partitions = [
        dag.to_partition(
            call.call_id
            for call in dag.nodes
            if graph.trace_graph.node_by_id[call.output_node_ids[0]].module_path in modules
        )
        for modules in ({"left", "left_end"}, {"right"}, set())
    ]
    assert len(set(partitions)) == 3
    assert partitions[0].prefix_node_ids != frozenset(
        n.canonical_id for n in graph.trace_graph.compute_nodes[:2]
    )
    with patch.object(graph.adapter, "build_segments", wraps=graph.adapter.build_segments) as build:
        analysis = graph.analyze(partitions[0])
        assert build.call_count == 0
        runtime = graph.materialize(analysis)
        assert build.call_count == 1

    def optimizer(parameters: object) -> torch.optim.Optimizer:
        """Construct equivalent stateful optimizers on independent models."""
        if optimizer_kind == "adam":
            return torch.optim.Adam(parameters, lr=0.01)
        return torch.optim.SGD(
            parameters, lr=0.01, momentum=0.9 if optimizer_kind == "momentum" else 0
        )

    expected_opt = optimizer(reference.parameters())
    actual_opt = optimizer(model.parameters())
    for batch, partition in zip((1, 2, 4, 8), (*partitions, partitions[0]), strict=True):
        runtime = runtime.repartition(partition)
        assert runtime.trace_graph is graph.trace_graph
        assert {
            id(v) for owners in runtime.parameter_bindings().values() for v in owners.values()
        } == set(map(id, model.parameters()))
        x = torch.randn(batch, 4, requires_grad=True)
        split_x = x.detach().clone().requires_grad_(True)
        target = torch.randn(batch, 2)
        expected_opt.zero_grad()
        actual_opt.zero_grad()
        expected_output = reference(x)
        expected_loss = nn.functional.mse_loss(expected_output, target)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(split_x)
        actual_loss, gradients = runtime.train_suffix(boundary, target)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(split_x.grad, x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        expected_opt.step()
        actual_opt.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_opt.state[actual], expected_opt.state[expected])
        torch.testing.assert_close(runtime.replay(split_x.detach()), reference(x.detach()))
