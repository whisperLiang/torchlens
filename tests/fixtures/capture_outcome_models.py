"""Shared tiny models for the capture-outcome test family.

``ThreeStageModel`` and the relu halt predicate were defined verbatim in four
capture-outcome test files (settlement, gates, characterization, matrix); this
module is the single definition. Models that differ per file (exploding, NaN,
parametrized-failure variants) stay local to their tests.
"""

import torch
from torch import nn


class ThreeStageModel(nn.Module):
    """Tiny linear -> relu -> sigmoid model with an operation after the halt target."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.relu(self.linear(x)))


def halt_on_relu(ctx: object) -> bool:
    """Halt predicate matching the relu op of ``ThreeStageModel``."""

    return ctx.kind == "op" and ctx.func_name == "relu"
