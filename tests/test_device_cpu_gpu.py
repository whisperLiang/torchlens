"""Device-placement tests for replay and partitioned replay."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens import (
    compile_execution_plan,
    enumerate_frontier_splits,
    replay_forward,
    replay_partitioned,
)


class DeviceToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.mark.smoke
@pytest.mark.parametrize("target_device", ["cpu", "cuda"])
def test_replay_device_selection(target_device: str) -> None:
    if target_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = DeviceToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs, device="cpu")
    split = enumerate_frontier_splits(plan, max_frontier_size=1, max_splits=4)[0]

    replay_output = replay_forward(plan, example_inputs, device=target_device)
    partitioned_output = replay_partitioned(
        plan,
        example_inputs,
        split=split,
        device=target_device,
    )

    assert str(replay_output.device).startswith(target_device)
    assert str(partitioned_output.device).startswith(target_device)
    assert torch.allclose(replay_output.cpu(), partitioned_output.cpu(), atol=1e-5, rtol=1e-4)
