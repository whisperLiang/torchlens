"""Boundary cache tests."""

from __future__ import annotations

from v2_helpers import split_request

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.split.errors import SplitUnsupportedError


class CacheModel(nn.Module):
    """Small model for cache roundtrip tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_boundary_cache_roundtrip(tmp_path) -> None:
    """A saved boundary validates and drives suffix replay after load."""

    torch.manual_seed(0)
    model = CacheModel().eval()
    x = torch.randn(2, 3)
    runtime = tl.split.prepare(model, x, split_request("after:relu"))
    boundary = runtime.run_prefix(x)

    runtime.save_boundary(boundary, tmp_path / "boundary")
    loaded = runtime.load_boundary(tmp_path / "boundary")

    runtime.validate_boundary(loaded)
    assert torch.allclose(runtime.run_suffix(loaded), model(x), atol=1e-5, rtol=1e-4)


def test_training_boundary_cache_is_suffix_only(tmp_path) -> None:
    """Caching a graph-connected boundary strips prefix-backward metadata."""

    torch.manual_seed(0)
    model = CacheModel().train()
    x = torch.randn(2, 3)
    target = torch.randn(2, 2)
    runtime = tl.split.prepare(model, x, split_request("after:relu", trainable=True))
    boundary = runtime.run_training_prefix(x)

    runtime.save_boundary(boundary, tmp_path / "training_boundary")
    loaded = runtime.load_boundary(tmp_path / "training_boundary")

    assert loaded.metadata.get("supports_prefix_backward") is False
    assert "prefix_boundary_tensors" not in loaded.metadata
    _loss, grads = runtime.train_suffix(loaded, target)
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(loaded, grads)
