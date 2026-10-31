"""DAG runtime integration and numerical training contracts."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch import nn
from v2_helpers import (
    analyze_fixture,
    prepare_fixture,
    request_fixture,
    select_node_prefix,
)

from torchlens.split import GraphPartition, SplitFeatures


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
    runtime = prepare_fixture(
        model,
        sample,
        request_fixture(
            partial(select_node_prefix, node_id="left_end"), features=SplitFeatures(training=True)
        ),
    )
    partition = GraphPartition(
        frozenset(
            node.canonical_id
            for node in runtime.trace_graph.compute_nodes
            if node.module_path in {"left", "left_end"}
        )
    )
    assert partition.prefix_node_ids
    with patch.object(
        runtime.adapter, "build_segments", wraps=runtime.adapter.build_segments
    ) as build:
        analysis = analyze_fixture(runtime, partition)
        assert build.call_count == 0
        runtime = runtime.materialize(analysis)
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
    for batch in (1, 2, 4, 8):
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
        runtime = runtime.repartition(partition)
        torch.testing.assert_close(runtime.replay(split_x.detach()), reference(x.detach()))


def test_partition_analysis_rejects_foreign_capture() -> None:
    """Equivalent graph shapes do not authorize cross-client state sharing."""
    x = torch.ones(2, 4)
    first = prepare_fixture(
        Branches(), x, request_fixture(partial(select_node_prefix, node_id="left_end"))
    )
    second = prepare_fixture(
        Branches(), x, request_fixture(partial(select_node_prefix, node_id="left_end"))
    )
    analysis = analyze_fixture(first, GraphPartition(frozenset()))
    with pytest.raises(ValueError, match="another captured graph"):
        second.materialize(analysis)
