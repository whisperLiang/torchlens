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


# --------------------------------------------------------------------------- F3
@pytest.mark.smoke
def test_cache_key_distinguishes_intervention_ready(tmp_path):
    """A capability change must MISS the cache -- never silently return a stale trace.

    Fail-before: the cache key omitted intervention_ready, so the second call
    (intervention_ready=True) returned the earlier cached trace whose
    intervention_ready was False -- silent wrongness with no warning.
    """
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(model, x, cache=True, cache_dir=cache_dir, intervention_ready=False)
    assert first.capture_cache_hit is False
    assert first.intervention_ready is False

    second = tl.trace(model, x, cache=True, cache_dir=cache_dir, intervention_ready=True)
    # Different capability => must not reuse the intervention_ready=False trace.
    assert second.capture_cache_hit is False
    assert second.intervention_ready is True


@pytest.mark.smoke
def test_cache_key_distinguishes_save_raw_input(tmp_path):
    """Sibling payload-policy option: save_raw_input must also key the cache."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(model, x, cache=True, cache_dir=cache_dir, save_raw_input=False)
    assert first.capture_cache_hit is False

    second = tl.trace(model, x, cache=True, cache_dir=cache_dir, save_raw_input=True)
    assert second.capture_cache_hit is False


@pytest.mark.smoke
def test_cache_hit_preserved_for_identical_capability(tmp_path):
    """Fix must not break caching: identical options still cache-hit."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(model, x, cache=True, cache_dir=cache_dir, intervention_ready=True)
    assert first.capture_cache_hit is False
    assert first.intervention_ready is True

    second = tl.trace(model, x, cache=True, cache_dir=cache_dir, intervention_ready=True)
    assert second.capture_cache_hit is True
    assert second.intervention_ready is True
