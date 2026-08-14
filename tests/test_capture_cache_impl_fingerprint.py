"""grind-p3 T5.1: a model IMPLEMENTATION change must invalidate the capture cache.

The capture-cache key fingerprinted only tensor content (``state_dict`` values,
training flags, non-persistent buffers) plus the call configuration. Two models
with identical parameters but DIFFERENT forward code therefore collided on the
same key: editing ``forward`` between runs silently served the STALE cached
trace of the old implementation. The key now folds in a model-implementation
signature (module tree structure, class qualnames, and per-class ``forward``
code-object digests, including instance-level ``forward`` overrides), so an
implementation change is a cache miss and a recapture.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _CacheModel(nn.Module):
    """Tiny deterministic model whose forward the tests mutate."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _cache_capture(tmp_path):
    return tl.options.CaptureOptions(cache=True, cache_dir=tmp_path / "cache")


def _op_labels(trace) -> list[str]:
    return list(trace.layer_labels)


def test_changed_class_forward_is_a_cache_miss(tmp_path) -> None:
    """Same state_dict, changed ``forward`` -> cache miss, fresh capture."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    assert any("relu" in label for label in _op_labels(first))

    original_forward = _CacheModel.forward
    try:

        def sigmoid_forward(self, inputs):  # noqa: ANN001 - test shim
            return torch.sigmoid(self.lin(inputs))

        _CacheModel.forward = sigmoid_forward
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    finally:
        _CacheModel.forward = original_forward

    assert second.capture_cache_hit is False, (
        "a changed forward implementation must not hit the stale cached trace"
    )
    labels = _op_labels(second)
    assert any("sigmoid" in label for label in labels)
    assert not any("relu" in label for label in labels)


def test_instance_forward_override_is_a_cache_miss(tmp_path) -> None:
    """An instance-level ``forward`` override also invalidates the cache."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False

    def tanh_forward(self, inputs):  # noqa: ANN001 - test shim
        return torch.tanh(self.lin(inputs))

    model.forward = types.MethodType(tanh_forward, model)
    try:
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    finally:
        del model.forward

    assert second.capture_cache_hit is False
    assert any("tanh" in label for label in _op_labels(second))


def test_unchanged_model_still_hits(tmp_path) -> None:
    """The fingerprint stays stable for an unchanged model (no false misses)."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is True


def test_submodule_structure_change_is_a_cache_miss(tmp_path) -> None:
    """Swapping a parameter-free submodule class changes the implementation."""

    class _Act(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    class _OtherAct(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

    class _Wrapper(nn.Module):
        def __init__(self, act: nn.Module) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.act = act

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.lin(x))

    torch.manual_seed(0)
    first_model = _Wrapper(_Act())
    torch.manual_seed(0)
    second_model = _Wrapper(_OtherAct())
    # Identical tensor content: only the activation submodule CLASS differs.
    second_model.load_state_dict(first_model.state_dict())

    x = torch.randn(1, 4)
    first = tl.trace(first_model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(second_model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False
    assert any("sigmoid" in label for label in _op_labels(second))
