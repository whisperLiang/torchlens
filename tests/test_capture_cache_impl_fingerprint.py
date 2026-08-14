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


class _AttrLoop(nn.Module):
    """Model whose traced program depends on a PLAIN instance attribute."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        for _ in range(self.k):
            y = torch.relu(y)
        return y


def test_changed_plain_instance_attribute_is_a_cache_miss(tmp_path) -> None:
    """Loop(5) must never be served Loop(1)'s cached trace (grind-r2 F39-1).

    The content fingerprint covers ``state_dict`` + training flags +
    non-persistent buffers and the implementation fingerprint covers module
    tree + class identity + ``forward`` code; before the fix nothing covered
    instance ``__dict__`` config, so ``Loop(5)`` hit ``Loop(1)``'s entry
    (hit=True, 3 ops instead of 7, no warning).
    """

    x = torch.randn(1, 4)
    torch.manual_seed(0)
    first = tl.trace(_AttrLoop(1), x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    torch.manual_seed(0)
    second = tl.trace(_AttrLoop(5), x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "a changed plain instance attribute must not hit the stale cached trace"
    )
    assert len(second.layer_labels) > len(first.layer_labels)
    torch.manual_seed(0)
    third = tl.trace(_AttrLoop(5), x, capture=_cache_capture(tmp_path))
    assert third.capture_cache_hit is True, (
        "an unchanged attribute inventory must still re-hit (no false misses)"
    )


def test_mutated_numeric_instance_attribute_is_a_cache_miss(tmp_path) -> None:
    """The malignant variant: same op count, numerically wrong served values."""

    class _Scaled(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = 1.0
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x) * self.scale

    model = _Scaled()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    model.scale = 2.0
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False
    assert torch.allclose(second.layer_list[-1].out, first.layer_list[-1].out * 2.0)


def test_attribute_fragments_are_bounded_and_address_free() -> None:
    """Fragment rules: primitives by value, opaque objects by type, bounded."""

    from torchlens._capture_state_helpers import _attribute_state_fragment

    assert _attribute_state_fragment(5) == 5
    assert _attribute_state_fragment("mode") == "mode"
    assert _attribute_state_fragment(torch.float32) == ("torch-value", "torch.float32")
    # Tensor attributes key by CONTENT, not identity.
    tensor_a = torch.ones(3)
    tensor_b = torch.ones(3)
    assert _attribute_state_fragment(tensor_a) == _attribute_state_fragment(tensor_b)

    # Opaque objects key by TYPE identity only -- never an address repr.
    class _Opaque:
        pass

    fragment = _attribute_state_fragment(_Opaque())
    assert fragment == _attribute_state_fragment(_Opaque())
    assert "0x" not in repr(fragment)
    # Depth ceiling terminates pathological nesting.
    nested: list = [1]
    for _ in range(10):
        nested = [nested]
    assert "<attr-depth-ceiling>" in repr(_attribute_state_fragment(nested))
