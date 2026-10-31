"""Adversarial alias and positional-write coverage for native DAG partitions."""

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

from torchlens.split import GraphPartition, SplitFeatures
from torchlens.split.errors import SplitUnsupportedError


@pytest.mark.parametrize("retain_trace", [False, True])
def test_read_alias_created_before_write_keeps_forward_and_backward_truth(
    retain_trace: bool,
) -> None:
    """Transport both a preexisting permutation and its written allocation."""

    class AliasRead(nn.Module):
        """Read a permutation created before a certified local allocation write."""

        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(4, 4)
            self.activation = nn.ReLU(inplace=True)
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Make independent frontier values whose storage aliases before transport."""
            value = self.project(x)
            alias = value.permute(0, 2, 1)
            written = self.activation(value)
            return self.head(written + alias.permute(0, 2, 1))

    model = AliasRead()
    reference = deepcopy(model)
    runtime = prepare_fixture(
        model,
        torch.randn(2, 4, 4),
        request_fixture(
            partial(select_node_prefix, node_id="activation"),
            features=SplitFeatures(training=True, retain_trace=retain_trace),
        ),
    )
    selected = frozenset(
        node.canonical_id
        for node in runtime.trace_graph.compute_nodes
        if node.canonical_id in runtime.plan.prefix_node_ids
    )
    runtime = runtime.repartition(GraphPartition(selected))
    assert runtime.plan.ordered_mutation_node_ids
    x = torch.randn(2, 4, 4, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_(True)
    target = torch.randn(2, 4, 2)
    expected_loss = nn.functional.mse_loss(reference(expected_x), target)
    expected_loss.backward()
    boundary = runtime.run_training_prefix(x)
    loss, gradients = runtime.train_suffix(boundary, target)
    runtime.backward_prefix(boundary, gradients)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(x.grad, expected_x.grad)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("fresh", [False, True])
def test_positional_functional_inplace_argument_is_not_missed(fresh: bool) -> None:
    """Positional inplace=True must still require a certified local allocation."""

    class PositionalWrite(nn.Module):
        """Use the positional functional spelling of the activation write."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Write either fresh storage or the original caller-owned input."""
            value = x + 1 if fresh else x
            return nn.functional.relu(value, True) * 2

    model = PositionalWrite()
    runtime = prepare_fixture(
        model, torch.randn(2, 4), request_fixture(partial(select_node_prefix, node_id="relu"))
    )
    partition = GraphPartition(frozenset())
    if fresh:
        cut = runtime.repartition(partition)
        assert cut.plan.ordered_mutation_node_ids
        x = torch.randn(2, 4)
        torch.testing.assert_close(cut.replay(x), model(x))
    else:
        cut = runtime.repartition(partition)
        x = torch.randn(2, 4)
        torch.testing.assert_close(cut.replay(x.clone()), model(x.clone()))
        with pytest.raises(SplitUnsupportedError) as caught:
            cut.run_training_prefix(x)
        assert caught.value.context.reason == "partition_training_boundary_write"
