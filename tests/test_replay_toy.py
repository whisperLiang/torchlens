"""Toy-model tests for execution-plan compilation and replay."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from torchlens import compile_execution_plan, replay_forward, validate_replay_equivalence


@dataclass
class ToyOutput:
    logits: torch.Tensor
    aux: dict


class MultiOutputToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> ToyOutput:
        left = torch.relu(self.left(x))
        right = torch.sigmoid(self.right(x))
        merged = torch.cat([left, right], dim=-1)
        logits = self.head(merged)
        return ToyOutput(logits=logits, aux={"left": left, "right": right})


class BatchReshapeToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.reshape(x.size(0), -1)
        hidden = torch.relu(self.hidden(flattened))
        return self.head(hidden)


class BatchFactoryToy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_bias = torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)
        return x[:, :1] + batch_bias


@pytest.mark.smoke
def test_compile_execution_plan_and_replay_forward_toy() -> None:
    model = MultiOutputToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs)

    replay_output = replay_forward(plan, example_inputs)
    direct_output = model(example_inputs)

    assert torch.allclose(replay_output.logits, direct_output.logits)
    assert torch.allclose(replay_output.aux["left"], direct_output.aux["left"])
    assert torch.allclose(replay_output.aux["right"], direct_output.aux["right"])


@pytest.mark.smoke
def test_validate_replay_equivalence_toy() -> None:
    model = MultiOutputToy().eval()
    example_inputs = torch.randn(3, 4)
    plan = compile_execution_plan(model, example_inputs)

    assert validate_replay_equivalence(plan, model, example_inputs)


@pytest.mark.smoke
def test_replay_forward_allows_batch_size_change_for_reshape() -> None:
    model = BatchReshapeToy().eval()
    example_inputs = torch.randn(2, 2, 2)
    new_inputs = torch.randn(5, 2, 2)
    plan = compile_execution_plan(model, example_inputs)

    replay_output = replay_forward(plan, new_inputs)
    direct_output = model(new_inputs)

    assert torch.allclose(replay_output, direct_output)


@pytest.mark.smoke
def test_replay_forward_allows_batch_size_change_for_factory_sizes() -> None:
    model = BatchFactoryToy().eval()
    example_inputs = torch.randn(2, 3)
    new_inputs = torch.randn(5, 3)
    plan = compile_execution_plan(model, example_inputs)

    replay_output = replay_forward(plan, new_inputs)
    direct_output = model(new_inputs)

    assert torch.allclose(replay_output, direct_output)
