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


class InplaceSuffixTrainToy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundary = x + 1
        boundary.add_(2)
        return boundary * 3


class NoGradPrefixTrainToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundary = x + 1
        return self.head(boundary)


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


@pytest.mark.smoke
def test_backward_prefix_matches_cached_boundary_after_inplace_suffix() -> None:
    model = InplaceSuffixTrainToy().train()
    example_inputs = torch.randn(4, 4, requires_grad=True)
    targets = torch.randn(4, 4)
    plan = compile_execution_plan(model, example_inputs)
    split = next(
        split
        for split in enumerate_frontier_splits(plan, max_frontier_size=1, max_splits=8)
        if split.boundary_labels == ["add_1_1"]
    )

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
    assert prefix_result["boundary"]["cut_id"] == split.split_id


@pytest.mark.smoke
def test_backward_prefix_accepts_missing_boundary_grad_for_nondiff_prefix() -> None:
    model = NoGradPrefixTrainToy().train()
    example_inputs = torch.randn(4, 4)
    targets = torch.randn(4, 4)
    plan = compile_execution_plan(model, example_inputs)
    split = next(
        split
        for split in enumerate_frontier_splits(plan, max_frontier_size=1, max_splits=8)
        if split.boundary_labels == ["add_1_1"]
    )

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

    assert split_result["boundary_grad"][split.boundary_labels[0]] is None
    assert prefix_result["boundary"]["cut_id"] == split.split_id
