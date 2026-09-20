"""Torch suffix microbatch training preserves logical-batch semantics."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl


class TinyMicrobatchMlp(nn.Module):
    """Small independently-sampleable suffix-training model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.fc2(self.relu(self.fc1(x)))


class CountingSgd(torch.optim.SGD):
    """Count logical optimizer calls without changing SGD semantics."""

    def __init__(self, params: object, **kwargs: object) -> None:
        super().__init__(params, **kwargs)
        self.zero_count = 0
        self.step_count = 0

    def zero_grad(self, *args: object, **kwargs: object) -> None:
        self.zero_count += 1
        super().zero_grad(*args, **kwargs)

    def step(self, *args: object, **kwargs: object) -> None:
        self.step_count += 1
        super().step(*args, **kwargs)


def test_uneven_microbatches_match_full_suffix_and_prefix_update() -> None:
    """Uneven 3+3+1 chunks match one full logical mean objective."""

    torch.manual_seed(17)
    full_model = TinyMicrobatchMlp()
    micro_model = copy.deepcopy(full_model)
    x = torch.randn(7, 4)
    targets = torch.randn(7, 3)
    full_runtime = tl.split.prepare(full_model, x, split_request("after:relu", trainable=True))
    micro_runtime = tl.split.prepare(micro_model, x, split_request("after:relu", trainable=True))
    full_suffix = CountingSgd(full_model.fc2.parameters(), lr=0.03)
    full_prefix = CountingSgd(full_model.fc1.parameters(), lr=0.03)
    micro_suffix = CountingSgd(micro_model.fc2.parameters(), lr=0.03)
    micro_prefix = CountingSgd(micro_model.fc1.parameters(), lr=0.03)

    full_boundary = full_runtime.run_training_prefix(x)
    full_loss, full_grads = full_runtime.train_suffix(full_boundary, targets, optimizer=full_suffix)
    full_runtime.backward_prefix(full_boundary, full_grads, optimizer=full_prefix)
    micro_boundary = micro_runtime.run_training_prefix(x)
    micro_loss, micro_grads = micro_runtime.train_suffix(
        micro_boundary,
        targets,
        optimizer=micro_suffix,
        microbatch_size=3,
    )
    micro_runtime.backward_prefix(micro_boundary, micro_grads, optimizer=micro_prefix)

    torch.testing.assert_close(micro_loss, full_loss, atol=1e-6, rtol=1e-5)
    assert set(micro_grads) == set(full_grads)
    for key in full_grads:
        torch.testing.assert_close(micro_grads[key], full_grads[key], atol=1e-6, rtol=1e-5)
    for left, right in zip(full_model.parameters(), micro_model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    assert (micro_suffix.zero_count, micro_suffix.step_count) == (1, 1)
    assert (micro_prefix.zero_count, micro_prefix.step_count) == (1, 1)


def test_microbatch_target_callback_and_nested_targets() -> None:
    """Nested targets use generic slicing and callbacks can override it."""

    model = TinyMicrobatchMlp()
    x = torch.randn(7, 4)
    target = {"value": torch.randn(7, 3), "constant": torch.tensor(2.0)}
    runtime = tl.split.prepare(model, x, split_request("after:relu", trainable=True))
    boundary = runtime.run_training_prefix(x)
    seen: list[tuple[int, int, int]] = []

    def loss_fn(output: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.nn.functional.mse_loss(output, values["value"] + values["constant"] * 0)

    def slicer(values: dict[str, torch.Tensor], start: int, end: int, batch: int) -> object:
        seen.append((start, end, batch))
        return {"value": values["value"][start:end], "constant": values["constant"]}

    loss, grads = runtime.train_suffix(
        boundary,
        target,
        loss_fn=loss_fn,
        microbatch_size=3,
        target_slicer=slicer,
    )
    assert torch.isfinite(loss)
    assert grads
    assert seen == [(0, 3, 7), (3, 6, 7), (6, 7, 7)]


def test_run_prefix_has_no_autograd_graph_but_training_prefix_does() -> None:
    """Frozen prefix execution owns no autograd graph; training remains connected."""

    model = TinyMicrobatchMlp()
    runtime = tl.split.prepare(
        model, torch.randn(2, 4), split_request("after:relu", trainable=True)
    )
    detached = runtime.run_prefix(torch.randn(7, 4))
    assert all(
        not value.requires_grad
        for value in detached.tensors.values()
        if isinstance(value, torch.Tensor)
    )
    assert all(
        value.grad_fn is None
        for value in detached.tensors.values()
        if isinstance(value, torch.Tensor)
    )
    connected = runtime.run_training_prefix(torch.randn(7, 4, requires_grad=True))
    assert any(
        value.requires_grad and value.grad_fn is not None
        for value in connected.tensors.values()
        if isinstance(value, torch.Tensor)
    )


def test_microbatch_refuses_invalid_size_and_non_torch_backend() -> None:
    """The explicit microbatch policy never silently falls back."""

    model = TinyMicrobatchMlp()
    x = torch.randn(7, 4)
    runtime = tl.split.prepare(model, x, split_request("after:relu", trainable=True))
    boundary = runtime.run_prefix(x)
    with pytest.raises(ValueError, match="positive"):
        runtime.train_suffix(boundary, torch.randn(7, 3), microbatch_size=0)
