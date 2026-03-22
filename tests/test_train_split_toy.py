"""Toy-model tests for replay training and prefix rebackward."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens import (
    backward_prefix_from_boundary,
    compile_execution_plan,
    enumerate_frontier_splits,
    train_partitioned,
    validate_gradient_equivalence,
)


class TrainToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = torch.relu(self.left(x))
        right = torch.sigmoid(self.right(x))
        return self.head(torch.cat([left, right], dim=-1))


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


@pytest.mark.smoke
def test_train_partitioned_and_backward_prefix_from_boundary_toy() -> None:
    model = TrainToy().train()
    example_inputs = torch.randn(4, 4)
    targets = torch.randn(4, 2)
    plan = compile_execution_plan(model, example_inputs)
    split = enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=8)[0]

    split_result = train_partitioned(
        plan,
        example_inputs,
        split=split,
        targets=targets,
        loss_fn=mse_loss,
        return_boundary=True,
        return_boundary_grad=True,
        step_optimizer=False,
    )
    prefix_result = backward_prefix_from_boundary(
        plan,
        example_inputs,
        split,
        split_result["boundary_grad"],
        step_optimizer=False,
        match_boundary=True,
        cached_boundary=split_result["boundary"],
    )

    assert "output" in split_result
    assert "loss" in split_result
    assert prefix_result["boundary"]["cut_id"] == split.split_id


@pytest.mark.smoke
def test_validate_gradient_equivalence_toy() -> None:
    model = TrainToy().train()
    example_inputs = torch.randn(4, 4)
    targets = torch.randn(4, 2)
    plan = compile_execution_plan(model, example_inputs)
    split = enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=8)[0]

    assert validate_gradient_equivalence(
        plan,
        split,
        model,
        example_inputs,
        targets=targets,
        loss_fn=mse_loss,
        atol=1e-5,
        rtol=1e-4,
    )
