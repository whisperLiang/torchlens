"""Toy-model tests for frontier split enumeration and partition replay."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens import (
    compile_execution_plan,
    enumerate_frontier_splits,
    replay_partitioned,
    validate_split_equivalence,
)


class BranchToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = torch.relu(self.left(x))
        right = torch.sigmoid(self.right(x))
        return self.head(torch.cat([left, right], dim=-1))


class WideToy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + 1
        b = x + 2
        c = x + 3
        d = a * 2
        e = b * 3
        f = c * 4
        return d + e + f


class IndependentPassthroughToy(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        boundary = torch.relu(x + 1)
        return boundary + (y * 2)


@pytest.mark.smoke
def test_enumerate_frontier_splits_toy() -> None:
    model = BranchToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs)

    splits = enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=8)

    assert splits
    assert all(split.boundary_indices for split in splits)
    assert all(
        idx not in plan.input_node_indices and idx not in plan.output_node_indices
        for split in splits
        for idx in split.boundary_indices
    )


@pytest.mark.smoke
def test_enumerate_frontier_splits_all_mode_toy() -> None:
    model = WideToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs)

    minimal_splits = enumerate_frontier_splits(plan, mode="minimal")
    all_splits = enumerate_frontier_splits(plan, mode="all")

    assert all_splits
    assert {tuple(split.boundary_indices) for split in minimal_splits}.issubset(
        {tuple(split.boundary_indices) for split in all_splits}
    )


@pytest.mark.smoke
def test_replay_partitioned_raw_and_boundary_toy() -> None:
    model = BranchToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs)
    split = enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=8)[0]

    raw_result = replay_partitioned(plan, example_inputs, split=split, return_boundary=True)
    boundary_output = replay_partitioned(
        plan,
        raw_result["boundary"],
        split=split,
        input_mode="boundary",
    )

    assert torch.allclose(raw_result["output"], boundary_output)
    assert validate_split_equivalence(plan, split, model, example_inputs)


@pytest.mark.smoke
def test_replay_partitioned_boundary_with_independent_passthrough_input() -> None:
    model = IndependentPassthroughToy().eval()
    x = torch.randn(2, 4)
    y = torch.randn(2, 4)
    plan = compile_execution_plan(model, (x, y))
    splits = enumerate_frontier_splits(plan, max_frontier_size=1, max_splits=8)
    split = next(split for split in splits if split.meta.get("passthrough_input_indices"))

    raw_result = replay_partitioned(plan, (x, y), split=split, return_boundary=True)
    boundary_output = replay_partitioned(
        plan,
        {"boundary": raw_result["boundary"], "raw_inputs": (x, y)},
        split=split,
        input_mode="boundary",
    )

    assert torch.allclose(raw_result["output"], boundary_output)
    with pytest.raises(ValueError):
        replay_partitioned(plan, raw_result["boundary"], split=split, input_mode="boundary")
