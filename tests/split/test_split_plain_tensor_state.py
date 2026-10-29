"""Plain Tensor attributes captured as state remain live after compaction."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, SplitRequest, after, prepare
from torchlens.split.errors import SplitUnsupportedError


class PlainTensorState(nn.Module):
    """Model the unregistered stride tensor used by detection heads."""

    def __init__(self) -> None:
        """Create named parameters and a plain, non-gradient state source."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.cut = nn.Tanh()
        self.right = nn.Linear(4, 2)
        self.stride = torch.tensor([2.0, 4.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the plain state through an indexing operation in the suffix."""

        return self.right(self.cut(self.left(x))) * self.stride[1]


@pytest.mark.parametrize("retain_trace", [False, True])
def test_plain_tensor_state_training_replacement_and_recut(retain_trace: bool) -> None:
    """Match gradients and Adam state while following plain attribute replacement."""

    torch.manual_seed(717)
    model = PlainTensorState()
    reference = deepcopy(model)
    runtime = prepare(
        model,
        torch.randn(2, 4),
        SplitRequest(
            after("cut"), features=SplitFeatures(training=True, retain_trace=retain_trace)
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.002)
    for step in range(3):
        if step:
            model.stride = torch.tensor([3.0, 5.0 + step])
            reference.stride = model.stride.clone()
        rows = {row.state_id: row for row in runtime.parameter_ownership()}
        assert rows["buffer:stride"].aliases == ("stride",)
        assert not rows["buffer:stride"].optimizer_managed
        x = torch.randn(4, 4, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        target = torch.randn(4, 2)
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        expected_loss = nn.functional.mse_loss(reference(expected_x), target)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, target)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, expected_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(optimizer.state[actual], expected_optimizer.state[expected])
        selected = (
            frozenset(node.canonical_id for node in runtime.trace_graph.compute_nodes)
            if step == 0
            else frozenset()
        )
        runtime = runtime.partition(GraphPartition(selected))


@pytest.mark.parametrize("retain_trace", [False, True])
def test_removed_plain_tensor_state_refuses_typed(retain_trace: bool) -> None:
    """Missing live state must never fall back to a stale captured value."""

    model = PlainTensorState()
    runtime = prepare(
        model,
        torch.randn(2, 4),
        SplitRequest(
            after("cut"), features=SplitFeatures(training=True, retain_trace=retain_trace)
        ),
    )
    del model.stride
    with pytest.raises(SplitUnsupportedError, match="live buffer value is unavailable"):
        boundary = runtime.run_training_prefix(torch.randn(2, 4))
        runtime.train_suffix(boundary, torch.randn(2, 2))
