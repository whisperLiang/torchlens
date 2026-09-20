"""Real-model frozen-prefix adaptation with synthetic, randomly initialized inputs."""

from __future__ import annotations

from importlib import import_module

import pytest
import torch
from real_model_helpers import _skip_if_module_missing, _skip_unless_enabled
from v2_helpers import split_request

import torchlens as tl

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


@pytest.mark.parametrize("microbatch_size", [None, 2])
def test_resnet18_frozen_prefix_suffix_training(microbatch_size: int | None) -> None:
    """A trainable ResNet prefix saves no autograd tensors during head adaptation.

    Parameters
    ----------
    microbatch_size:
        Exercise full-batch training and an uneven 2 + 1 suffix execution.
    """

    _skip_unless_enabled()
    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(19)
    model = torchvision.models.resnet18(weights=None, num_classes=4).eval()
    inputs = torch.randn(3, 3, 64, 64)
    targets = torch.tensor([0, 1, 3])
    runtime = tl.split.prepare(model, inputs[:1], split_request("before:fc", trainable=True))
    prefix_parameters = [
        param for name, param in model.named_parameters() if not name.startswith("fc.")
    ]
    initial_prefix = [param.detach().clone() for param in prefix_parameters]
    initial_head = [param.detach().clone() for param in model.fc.parameters()]
    saved_tensors = 0

    def pack(value: torch.Tensor) -> torch.Tensor:
        """Count prefix tensors that autograd would retain for backward.

        Parameters
        ----------
        value:
            A tensor requested by a backward formula.
        """

        nonlocal saved_tensors
        saved_tensors += 1
        return value

    with torch.no_grad():
        expected_output = model(inputs)
        torch.testing.assert_close(runtime.replay(inputs), expected_output, atol=1e-4, rtol=1e-3)
        expected_loss = torch.nn.functional.cross_entropy(expected_output, targets)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        boundary = runtime.run_prefix(inputs)
    assert saved_tensors == 0
    assert all(
        not value.requires_grad and value.grad_fn is None for value in boundary.tensors.values()
    )

    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
    assert not optimizer.state
    loss, gradients = runtime.train_suffix(
        boundary, targets, optimizer=optimizer, microbatch_size=microbatch_size
    )
    torch.testing.assert_close(loss.detach(), expected_loss, atol=1e-5, rtol=1e-4)
    assert gradients
    assert all(param.grad is None for param in prefix_parameters)
    for actual, original in zip(prefix_parameters, initial_prefix, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
    assert all(param.grad is not None for param in model.fc.parameters())
    assert any(
        not torch.equal(actual, original)
        for actual, original in zip(model.fc.parameters(), initial_head, strict=True)
    )
