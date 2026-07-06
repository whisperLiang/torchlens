"""Dynamic-batch split replay tests."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class DynamicShapeModel(nn.Module):
    """Toy model that uses common shape-changing Torch ops."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.view(batch, -1)
        flat = torch.reshape(flat, (batch, -1))
        flat = torch.flatten(flat, start_dim=1)
        return self.proj(self.relu(flat))


def test_dynamic_batch_replay_for_view_reshape_flatten() -> None:
    """A trace at batch 2 replays supported dynamic batches."""

    torch.manual_seed(0)
    model = DynamicShapeModel().eval()
    example = torch.randn(2, 3, 4)
    runtime = tl.prepare_split(
        model,
        example,
        tl.SplitSpec("50%", dynamic_batch=(1, 8)),
    )

    for batch in (1, 2, 4, 8):
        x = torch.randn(batch, 3, 4)
        assert torch.allclose(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)
