"""Regression tests for the capture-cache path (r19c hardening).

Covers a previously-uncovered gap: nothing combined ``cache=True`` with an
activation/grad transform (F2) or exercised a capability-option cache
collision (F3), and a pre-forward setup failure leaked capture-global state
(SOL-A5-002). The ``trace()`` docstring also carried three false
self-referential "deprecated alias" lines (DOC1).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


def _act_transform(value: torch.Tensor) -> torch.Tensor:
    # Module-level (picklable) so the cached trace can be serialized.
    return value.detach() + 1.0


def _grad_transform(value: torch.Tensor) -> torch.Tensor:
    return value.detach() * 2.0


# --------------------------------------------------------------------------- F2
@pytest.mark.smoke
def test_cache_with_activation_transform_roundtrips(tmp_path):
    """cache=True + activation_transform must not raise and must cache-hit.

    Fail-before: ``AttributeError: can't set attribute 'transformed_out'`` from
    a raw ``setattr`` on the read-only ``Layer.transformed_out`` proxy inside
    ``_prepare_log_for_capture_cache``.
    """
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(model, x, cache=True, cache_dir=cache_dir, activation_transform=_act_transform)
    assert first.capture_cache_hit is False

    second = tl.trace(
        model, x, cache=True, cache_dir=cache_dir, activation_transform=_act_transform
    )
    assert second.capture_cache_hit is True


@pytest.mark.smoke
def test_cache_with_grad_transform_roundtrips(tmp_path):
    """The sibling grad_transform path must also survive cache serialization."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(
        model,
        x,
        cache=True,
        cache_dir=cache_dir,
        grad_transform=_grad_transform,
        save_grads=True,
        backward_ready=True,
    )
    assert first.capture_cache_hit is False

    second = tl.trace(
        model,
        x,
        cache=True,
        cache_dir=cache_dir,
        grad_transform=_grad_transform,
        save_grads=True,
        backward_ready=True,
    )
    assert second.capture_cache_hit is True
