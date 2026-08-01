"""Stateful live-model ``Trace.run`` diagnostics."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


def test_training_mode_stateless_model_does_not_warn() -> None:
    """Training mode alone should not trigger the live-state warning."""

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU()).train()
    captured = tl.trace(model, torch.randn(2, 3))

    with warnings.catch_warnings(record=True) as observed:
        captured.run(inputs=torch.randn(2, 3))

    assert not [
        warning for warning in observed if "training-mode BatchNorm" in str(warning.message)
    ]


def test_live_run_warns_once_and_batchnorm_stats_mutate() -> None:
    """Default live execution should warn before mutating BatchNorm state."""

    model = nn.BatchNorm1d(3).train()
    captured = tl.trace(model, torch.randn(4, 3))
    before = model.running_mean.detach().clone()

    with pytest.warns(
        UserWarning,
        match=r"run\(\) detected training-mode BatchNorm running-stat buffers.*pristine=True",
    ) as observed:
        with pytest.raises(ValueError, match="pristine=True"):
            captured.run(inputs=torch.randn(4, 3))

    assert "running_mean, running_var, num_batches_tracked on module '<root>'" in str(
        observed[0].message
    )
    assert not torch.equal(model.running_mean, before)
    with warnings.catch_warnings(record=True) as observed:
        with pytest.raises(ValueError, match="pristine=True"):
            captured.run(inputs=torch.randn(4, 3))
    assert not [
        warning for warning in observed if "training-mode BatchNorm" in str(warning.message)
    ]


def test_eval_mode_batchnorm_does_not_warn() -> None:
    """Eval-mode BatchNorm reads running statistics without updating them."""

    model = nn.BatchNorm1d(3).eval()
    captured = tl.trace(model, torch.randn(4, 3))

    with warnings.catch_warnings(record=True) as observed:
        with pytest.raises(ValueError, match="pristine=True"):
            captured.run(inputs=torch.randn(4, 3))

    assert not [
        warning for warning in observed if "training-mode BatchNorm" in str(warning.message)
    ]


def test_graph_change_error_names_live_state_mutation_and_isolation_hint() -> None:
    """The unchanged projector tripwire should explain likely live-state causes."""

    class _CounterBranch(nn.Module):
        """Change operation kind after the first forward via a Python counter."""

        def __init__(self) -> None:
            """Initialize the call counter."""

            super().__init__()
            self.calls = 0

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Run the branch selected by the current call count.

            Parameters
            ----------
            inputs:
                Runtime tensor.

            Returns
            -------
            torch.Tensor
                Branch result.
            """

            self.calls += 1
            if self.calls == 1:
                return inputs + 1
            return torch.relu(inputs)

    model = _CounterBranch().eval()
    captured = tl.trace(model, torch.ones(2))

    with pytest.raises(ValueError) as caught:
        captured.run(inputs=torch.ones(2))

    message = str(caught.value)
    assert "Live-model state mutation across run() calls" in message
    assert "run(..., pristine=True)" in message
