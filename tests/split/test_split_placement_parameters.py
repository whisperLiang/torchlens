"""Placed parameter enumeration must identify the leaves used by split training."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan
from torchlens.split.state import SegmentState


class PlacedMlp(nn.Module):
    """Small model whose trainable prefix can own a separate device replica."""

    def __init__(self) -> None:
        """Construct an affine prefix and suffix around a named activation."""

        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the unsplit model used as the update oracle."""

        return self.fc2(self.relu(self.fc1(x)))


@pytest.mark.parametrize("warm_inference", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_placed_prefix_optimizer_uses_training_state(
    warm_inference: bool, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enumerated replicas receive gradients and updates before any training forward."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for heterogeneous state replicas.")
    if device == "cpu":
        # Exercise replica ownership on CPU-only CI as well as real CUDA.
        monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = PlacedMlp()
    reference = copy.deepcopy(model)
    x = torch.randn(3, 4)
    y = torch.randn(3, 3)
    original_prefix = [value.detach().clone() for value in model.fc1.parameters()]
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:relu", trainable=True, placement=PlacementPlan.across(device, "cpu")),
    )
    if warm_inference:
        runtime.run_prefix(x)
    prefix_parameters = runtime.prefix_parameters()
    assert len(prefix_parameters) == len(original_prefix)
    assert all(
        value.is_leaf and value.device == torch.device(device) for value in prefix_parameters
    )
    prefix_optimizer = torch.optim.SGD(prefix_parameters, lr=0.05)
    suffix_optimizer = torch.optim.SGD(
        runtime.suffix_parameters() or list(model.fc2.parameters()), lr=0.05
    )
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    for _ in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = torch.nn.functional.mse_loss(reference(x), y)
        reference_loss.backward()
        reference_optimizer.step()

        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, y, optimizer=suffix_optimizer)
        runtime.backward_prefix(boundary, gradients, optimizer=prefix_optimizer)

        torch.testing.assert_close(loss.detach(), reference_loss.detach(), atol=1e-5, rtol=1e-4)
        assert all(value.grad is not None for value in prefix_parameters)
        assert [id(value) for value in runtime.prefix_parameters()] == [
            id(value) for value in prefix_parameters
        ]
        for actual, expected in zip(prefix_parameters, reference.fc1.parameters(), strict=True):
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-4)
        for actual, expected in zip(model.fc1.parameters(), original_prefix, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(runtime.replay(x), reference(x), atol=1e-5, rtol=1e-4)


def test_training_prefix_parameters_with_captured_inference_sources() -> None:
    """The training getter binds live replicas even when inference uses captured state."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for heterogeneous state replicas.")
    model = PlacedMlp()
    x = torch.randn(3, 4)
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:relu", placement=PlacementPlan.across("cuda:0", "cpu")),
    )
    runtime.run_prefix(x)
    parameters = runtime.prefix_parameters()
    assert len(parameters) == 2
    optimizer = torch.optim.SGD(parameters, lr=0.05)
    before = [value.detach().clone() for value in parameters]
    boundary = runtime.run_training_prefix(x)
    runtime.backward_prefix(
        boundary,
        {key: torch.ones_like(value) for key, value in boundary.tensors.items()},
        optimizer=optimizer,
    )
    assert all(value.grad is not None for value in parameters)
    assert any(
        not torch.equal(value, initial) for value, initial in zip(parameters, before, strict=True)
    )
