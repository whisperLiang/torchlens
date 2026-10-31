"""Execution profiling uses real backward phases without touching caller state."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch import nn
from v2_helpers import (
    prepare_fixture,
    request_fixture,
    select_node_prefix,
)

from torchlens.split import PlacementPlan, SplitFeatures


@pytest.mark.parametrize("training", [False, True])
def test_profile_isolates_parameters_gradients_rng_and_pending_autograd(training: bool) -> None:
    """An outstanding caller graph remains usable after profile measurements."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(2, 4, requires_grad=True)
    runtime = prepare_fixture(
        model,
        x,
        request_fixture(
            partial(select_node_prefix, node_id="1"),
            features=SplitFeatures(training=True, retain_trace=False),
        ),
    )
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    grads = [p.grad for p in model.parameters()]
    values = [p.detach().clone() for p in model.parameters()]
    pending = model(x).sum()
    rng = torch.get_rng_state().clone()
    profile = runtime.profile_split(
        x, warmup=1, repeats=2, targets=torch.zeros(2, 2) if training else None
    )
    assert len(profile.prefix_forward.samples_ms) == 2
    assert profile.boundary_bytes == 64
    assert (profile.suffix_backward is not None) == training
    assert (profile.prefix_backward is not None) == training
    assert profile.peak_allocated_bytes is None
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert x.grad is None
    for p, grad, value in zip(model.parameters(), grads, values, strict=True):
        assert p.grad is grad
        torch.testing.assert_close(p.grad, torch.ones_like(p))
        torch.testing.assert_close(p, value)
    pending.backward()
    assert x.grad is not None


def test_profile_failure_restores_rng_and_lazy_state() -> None:
    """A user loss exception cannot leak measurement state into the source runtime."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.ones(2, 4)
    runtime = prepare_fixture(
        model,
        x,
        request_fixture(
            partial(select_node_prefix, node_id="1"), features=SplitFeatures(training=True)
        ),
    )
    states = [
        getattr(runtime.segments, name)._state for name in ("prefix", "training_prefix", "suffix")
    ]
    before = [dict(state._entries) for state in states]
    rng = torch.get_rng_state().clone()

    def fail(output: torch.Tensor, targets: object) -> torch.Tensor:
        """Advance RNG before raising to exercise the transactional finally path."""
        torch.rand(10)
        raise RuntimeError("bad objective")

    with pytest.raises(RuntimeError, match="bad objective"):
        runtime.profile_split(x, warmup=0, repeats=1, loss_fn=fail)
    assert [state._entries for state in states] == before
    torch.testing.assert_close(torch.get_rng_state(), rng)


def test_profile_preserves_prepared_keyword_inputs() -> None:
    """Omitting keyword overrides keeps the runtime's captured input contract."""

    class KeywordModel(nn.Module):
        """Use a required tensor keyword argument."""

        def forward(self, x: torch.Tensor, *, bias: torch.Tensor) -> torch.Tensor:
            """Combine both input leaves."""
            return torch.relu(x) + bias

    x = torch.ones(2, 4)
    bias = torch.ones(2, 4)
    runtime = prepare_fixture(
        KeywordModel(),
        x,
        request_fixture(partial(select_node_prefix, node_id="relu")),
        input_kwargs={"bias": bias},
    )
    result = runtime.profile_split(x, warmup=0, repeats=1)
    assert result.suffix_forward.median_ms > 0
    torch.testing.assert_close(bias, torch.ones_like(bias))


def test_profile_custom_loss_cannot_accumulate_into_caller_parameters() -> None:
    """Regularization closing over the source model must not write its gradients."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.ones(2, 4)
    runtime = prepare_fixture(
        model,
        x,
        request_fixture(
            partial(select_node_prefix, node_id="1"), features=SplitFeatures(training=True)
        ),
    )
    runtime.profile_split(
        x,
        warmup=0,
        repeats=1,
        loss_fn=lambda output, target: output.square().mean() + model[0].weight.square().sum(),
    )
    assert all(param.grad is None for param in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_profile_cuda_phases() -> None:
    """GPU phase measurements execute a synchronized real gradient path."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).cuda()
    x = torch.ones(2, 4, device="cuda")
    runtime = prepare_fixture(
        model,
        x,
        request_fixture(
            partial(select_node_prefix, node_id="1"), features=SplitFeatures(training=True)
        ),
    )
    result = runtime.profile_split(x, targets=torch.zeros(2, 2, device="cuda"), warmup=1, repeats=2)
    assert result.suffix_backward.median_ms > 0
    assert result.prefix_backward.median_ms > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_profile_parameter_free_cross_device_synchronizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit suffix placement is synchronized even with no CUDA state tensors."""
    model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
    x = torch.ones(2, 4)
    runtime = prepare_fixture(
        model,
        x,
        request_fixture(
            partial(select_node_prefix, node_id="0"), placement=PlacementPlan.across("cpu", "cuda")
        ),
    )
    sync = torch.cuda.synchronize
    synchronized = []

    def observe(device: object = None) -> None:
        """Record the synchronized GPU while preserving actual synchronization."""
        synchronized.append(str(device))
        sync(device)

    monkeypatch.setattr(torch.cuda, "synchronize", observe)
    result = runtime.profile_split(x, warmup=0, repeats=2)
    assert any(device.startswith("cuda") for device in synchronized)
    assert result.suffix_forward.median_ms > 0
