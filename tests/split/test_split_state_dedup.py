"""Runtime-scoped replica sharing preserves state and optimizer ownership."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan
from torchlens.split.adapters.torch import TorchSplitAdapter
from torchlens.split.placement import DevicePlacement
from torchlens.split.state import SegmentState, _StateReplicaPool


class TiedStateModel(nn.Module):
    """Reuse one weight before and after a named split point."""

    def __init__(self, *, frozen: bool) -> None:
        """Keep a trainable output layer even when the shared weight is frozen."""

        super().__init__()
        self.shared = nn.Linear(4, 4, bias=False)
        self.shared.weight.requires_grad_(not frozen)
        self.cut = nn.ReLU()
        self.head = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the tied weight twice without mutating its value."""

        return self.head(self.shared(self.cut(self.shared(value))))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("frozen", [False, True])
def test_tied_replicas_share_only_frozen_state(
    device: str, frozen: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen weights share storage; separate optimizer-owned weights do not."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for real cross-device replicas.")
    if device == "cpu":
        monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = TiedStateModel(frozen=frozen)
    inputs = torch.randn(3, 4)
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:cut", trainable=True, placement=PlacementPlan.on(device)),
    )
    prefix = runtime.segments.prefix._state
    suffix = runtime.segments.suffix._state
    prefix_weight = prefix.resolve(model.shared.weight, shareable=True)
    suffix_weight = suffix.resolve(model.shared.weight, shareable=True)
    assert prefix_weight is not model.shared.weight
    assert prefix_weight.device == suffix_weight.device == torch.device(device)
    assert (prefix_weight is suffix_weight) is frozen
    assert (prefix_weight.data_ptr() == suffix_weight.data_ptr()) is frozen
    assert runtime.segments.training_prefix._state is prefix
    assert prefix.resolve(model.shared.weight) is prefix_weight
    torch.testing.assert_close(runtime.replay(inputs).cpu(), model(inputs))
    if not frozen:
        suffix_before = suffix_weight.detach().clone()
        optimizer = torch.optim.SGD(runtime.prefix_parameters(), lr=0.1)
        prefix_weight.grad = torch.ones_like(prefix_weight)
        optimizer.step()
        torch.testing.assert_close(suffix_weight, suffix_before, atol=0, rtol=0)


@pytest.mark.parametrize("shareable", [False, True])
def test_nontrainable_state_requires_explicit_sharing_proof(
    shareable: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutable buffers remain isolated, while known constants can reuse replicas."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    pool = _StateReplicaPool()
    adapter = TorchSplitAdapter()
    left = SegmentState(adapter=adapter, placement=DevicePlacement("cpu"), replica_pool=pool)
    right = SegmentState(adapter=adapter, placement=DevicePlacement("cpu:0"), replica_pool=pool)
    source = torch.ones(4)
    first = left.resolve(source, shareable=shareable)
    second = right.resolve(source, shareable=shareable)
    assert (first is second) is shareable
    assert first is not source
    if not shareable:
        first.add_(1)
        torch.testing.assert_close(second, source, atol=0, rtol=0)


def test_captured_and_live_prefix_state_stay_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training updates cannot overwrite an inference prefix using captured state."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = TiedStateModel(frozen=False)
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:cut", placement=PlacementPlan.on("cpu")),
    )
    inference = runtime.run_prefix(inputs)
    inference_values = {key: value.clone() for key, value in inference.tensors.items()}
    prefix = runtime.segments.prefix._state
    training = runtime.segments.training_prefix._state
    assert training is not prefix
    parameters = runtime.prefix_parameters()
    assert parameters
    assert {id(entry.value) for entry in prefix.entries()}.isdisjoint(
        id(value) for value in parameters
    )
    with torch.no_grad():
        for value in parameters:
            value.add_(3)
    runtime.run_training_prefix(inputs)
    after = runtime.run_prefix(inputs)
    for key, value in after.tensors.items():
        torch.testing.assert_close(value, inference_values[key], atol=0, rtol=0)


def test_inherited_replicas_pool_the_effective_updated_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migrating a runtime never replaces updated state with its original source."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    # Force migration copies on CPU, including inherited owned state.
    monkeypatch.setattr(SegmentState, "_devices_match", lambda self, left, right: False)
    adapter = TorchSplitAdapter()
    previous = SegmentState(adapter=adapter, placement=DevicePlacement("cpu"))
    source = torch.ones(4)
    effective = previous.resolve(source, shareable=True)
    effective.add_(5)
    pool = _StateReplicaPool()
    rebound = [
        SegmentState(adapter=adapter, placement=DevicePlacement("cpu"), replica_pool=pool)
        for _ in range(2)
    ]
    for binding in rebound:
        binding.inherit_from(previous)
    first, second = [binding.resolve(source, shareable=True) for binding in rebound]
    assert first is second and first is not effective
    torch.testing.assert_close(first, effective, atol=0, rtol=0)
    torch.testing.assert_close(source, torch.ones(4), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA placement requires a GPU.")
def test_same_source_on_distinct_devices_has_distinct_replicas() -> None:
    """The shared pool must not serve a CPU value to a CUDA segment."""

    model = TiedStateModel(frozen=True)
    inputs = torch.randn(3, 4)
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:cut", trainable=True, placement=PlacementPlan.across("cuda", "cpu")),
    )
    actual = runtime.replay(inputs)
    prefix_weight = runtime.segments.prefix._state.resolve(model.shared.weight, shareable=True)
    suffix_weight = runtime.segments.suffix._state.resolve(model.shared.weight, shareable=True)
    assert prefix_weight.device == torch.device("cuda", torch.cuda.current_device())
    assert suffix_weight.device == torch.device("cpu")
    assert prefix_weight is not suffix_weight
    torch.testing.assert_close(actual, model(inputs))


def test_cuda_alias_normalization_uses_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unindexed CUDA placement must not accidentally alias device zero."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    adapter = TorchSplitAdapter()
    state = SegmentState(adapter=adapter, placement=DevicePlacement("cuda"))
    assert state._devices_match("cuda", "cuda:2")
    assert not state._devices_match("cuda", "cuda:0")


def test_runtime_replica_pool_does_not_outlive_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """No global cache retains pooled state after the owning runtime is collected."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = TiedStateModel(frozen=True)
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:cut", trainable=True, placement=PlacementPlan.on("cpu")),
    )
    runtime.replay(inputs)
    runtime_ref = weakref.ref(runtime)
    pool_ref = weakref.ref(runtime.segments.prefix._state._replica_pool)
    value_ref = weakref.ref(runtime.segments.prefix._state.resolve(model.shared.weight))
    del runtime
    gc.collect()
    assert runtime_ref() is None
    assert pool_ref() is None
    assert value_ref() is None
