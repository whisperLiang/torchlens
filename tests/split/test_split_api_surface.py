"""Public API surface tests for ``torchlens.split``."""

from __future__ import annotations

import pytest

import torchlens as tl
from torchlens.split import (
    BoundaryTensorSpec,
    ReplayBoundary,
    SplitRuntime,
    SplitSpec,
    prepare_split,
    prepare_split_replay,
)
from torchlens.split.errors import SplitSpecError


def test_split_exports_are_public() -> None:
    """Top-level and submodule split symbols should resolve."""

    assert tl.SplitSpec is SplitSpec
    assert tl.BoundaryTensorSpec is BoundaryTensorSpec
    assert tl.ReplayBoundary is ReplayBoundary
    assert tl.SplitRuntime is SplitRuntime
    assert tl.prepare_split is prepare_split
    assert tl.prepare_split_replay is prepare_split_replay
    for name in (
        "SplitSpec",
        "BoundaryTensorSpec",
        "ReplayBoundary",
        "SplitRuntime",
        "prepare_split",
        "prepare_split_replay",
    ):
        assert name in tl.__all__


def test_split_spec_defaults_and_validation() -> None:
    """SplitSpec validates local ABI decisions."""

    spec = SplitSpec("after:relu")
    assert spec.batch_symbol == "B"
    assert spec.dynamic_batch is None
    assert spec.trainable is False
    assert spec.mode == "generated_eager"
    assert spec.device_policy == "runtime"

    with pytest.raises(SplitSpecError):
        SplitSpec("")
    with pytest.raises(SplitSpecError):
        SplitSpec("50%", dynamic_batch=(4, 2))
    with pytest.raises(SplitSpecError):
        SplitSpec("0%")
    with pytest.raises(SplitSpecError):
        SplitSpec("100%")
    with pytest.raises(SplitSpecError):
        SplitSpec("50%", mode="bad")  # type: ignore[arg-type]
    with pytest.raises(SplitSpecError):
        SplitSpec("50%", device_policy="copy")  # type: ignore[arg-type]


def test_compiled_mode_is_accepted_by_spec() -> None:
    """Compiled mode is a valid spec value even though runtime v1 rejects it."""

    assert SplitSpec("50%", mode="compiled").mode == "compiled"
