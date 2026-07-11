"""Torch split training tests."""

from __future__ import annotations

from v2_helpers import split_request

import copy

import torch
from torch import nn

import torchlens as tl


class TrainMlp(nn.Module):
    """Small model with an explicit split-friendly hidden layer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class NonDiffPassModel(nn.Module):
    """Model with an integer passthrough crossing the split boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        idx = torch.argmax(x, dim=1)
        return self.fc2(h) + idx.float().unsqueeze(1)


def _params(module: nn.Module) -> list[torch.Tensor]:
    """Return detached parameter clones."""

    return [param.detach().clone() for param in module.parameters()]


def test_suffix_only_training_parity() -> None:
    """Training the suffix from a boundary updates head params like full-model head training."""

    torch.manual_seed(0)
    model = TrainMlp()
    split_model = copy.deepcopy(model)
    x = torch.randn(4, 4)
    y = torch.randn(4, 3)
    for param in model.fc1.parameters():
        param.requires_grad_(False)
    for param in split_model.fc1.parameters():
        param.requires_grad_(False)

    runtime = tl.split.prepare(split_model, x, split_request("after:relu", trainable=True))
    boundary = runtime.run_prefix(x)
    opt = torch.optim.SGD(split_model.fc2.parameters(), lr=0.05)
    full_opt = torch.optim.SGD(model.fc2.parameters(), lr=0.05)

    full_opt.zero_grad(set_to_none=True)
    full_loss = torch.nn.functional.mse_loss(model(x), y)
    full_loss.backward()
    full_opt.step()
    split_loss, grads = runtime.train_suffix(boundary, y, optimizer=opt)

    assert grads
    assert torch.allclose(split_loss.detach(), full_loss.detach(), atol=1e-5, rtol=1e-4)
    for left, right in zip(_params(split_model.fc2), _params(model.fc2)):
        assert torch.allclose(left, right, atol=1e-5, rtol=1e-4)


def test_full_split_training_gradient_handoff() -> None:
    """Suffix gradients can be handed back through a graph-connected prefix."""

    torch.manual_seed(0)
    model = TrainMlp()
    split_model = copy.deepcopy(model)
    x = torch.randn(4, 4)
    y = torch.randn(4, 3)
    runtime = tl.split.prepare(split_model, x, split_request("after:relu", trainable=True))
    boundary = runtime.run_training_prefix(x)
    suffix_opt = torch.optim.SGD(split_model.fc2.parameters(), lr=0.05)
    prefix_opt = torch.optim.SGD(split_model.fc1.parameters(), lr=0.05)
    full_opt = torch.optim.SGD(model.parameters(), lr=0.05)

    full_opt.zero_grad(set_to_none=True)
    full_loss = torch.nn.functional.mse_loss(model(x), y)
    full_loss.backward()
    full_opt.step()
    loss, grads = runtime.train_suffix(boundary, y, optimizer=suffix_opt)
    runtime.backward_prefix(boundary, grads, optimizer=prefix_opt)

    assert torch.allclose(loss.detach(), full_loss.detach(), atol=1e-5, rtol=1e-4)
    for left, right in zip(_params(split_model), _params(model)):
        assert torch.allclose(left, right, atol=1e-5, rtol=1e-4)


def test_nondifferentiable_boundary_is_skipped() -> None:
    """Non-floating boundary values are skipped when collecting boundary gradients."""

    model = NonDiffPassModel()
    x = torch.randn(4, 4)
    y = torch.randn(4, 3)
    runtime = tl.split.prepare(model, x, split_request("before:float", trainable=True))
    boundary = runtime.run_training_prefix(x)
    int_keys = [
        key
        for key, value in boundary.tensors.items()
        if isinstance(value, torch.Tensor) and not (value.is_floating_point() or value.is_complex())
    ]

    _loss, grads = runtime.train_suffix(boundary, y)

    assert int_keys
    assert all(key not in grads for key in int_keys)
