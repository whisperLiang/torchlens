"""Opt-in real-world split replay tests."""

from __future__ import annotations

import os

import pytest
import torch

import torchlens as tl

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def _enabled() -> bool:
    """Return whether opt-in real model tests are enabled."""

    return os.environ.get("TORCHLENS_REAL_MODEL_TESTS") == "1"


def test_resnet18_representative_splits_skip_cleanly_when_disabled() -> None:
    """ResNet18 split replay validates a small number of representative boundaries."""

    if not _enabled():
        pytest.skip("Set TORCHLENS_REAL_MODEL_TESTS=1 to run real-model split tests.")
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)

    for spec in (tl.SplitSpec("25%"), tl.SplitSpec("50%"), tl.SplitSpec("before:fc")):
        runtime = tl.prepare_split(model, x, spec)
        boundary = runtime.run_prefix(x)
        runtime.validate_boundary(boundary)
        assert torch.allclose(runtime.run_suffix(boundary), model(x), atol=1e-4, rtol=1e-3)


def test_tiny_transformer_config_split_skip_cleanly_when_disabled() -> None:
    """A no-download transformer config exercises attention-style structure."""

    if not _enabled():
        pytest.skip("Set TORCHLENS_REAL_MODEL_TESTS=1 to run real-model split tests.")
    transformers = pytest.importorskip("transformers")
    config = transformers.DistilBertConfig(
        vocab_size=128,
        n_layers=1,
        dim=32,
        hidden_dim=64,
        n_heads=4,
    )
    model = transformers.DistilBertModel(config).eval()
    x = torch.randint(0, 128, (2, 8))

    runtime = tl.prepare_split(model, x, tl.SplitSpec("50%"))
    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    split_output = runtime.run_suffix(boundary).last_hidden_state
    full_output = model(x).last_hidden_state

    assert torch.allclose(split_output, full_output, atol=1e-4, rtol=1e-3)


def test_env_gated_local_detector_split_placeholder() -> None:
    """Local detector coverage is skipped unless an explicit project root is supplied."""

    if not _enabled() or not os.environ.get("TORCHLENS_PLANK_ROAD_ROOT"):
        pytest.skip("Set TORCHLENS_PLANK_ROAD_ROOT for local detector split tests.")
    pytest.skip("Local detector adapter fixture is intentionally project-specific.")
