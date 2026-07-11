"""Public API surface tests for ``torchlens.split`` v2."""

from __future__ import annotations

import pytest

import torchlens as tl
from torchlens.split import (
    BoundarySchema,
    SplitFeatures,
    SplitGraphIR,
    SplitPoint,
    SplitRequest,
    SplitRuntime,
    SplitVerificationStatus,
    after,
    prepare,
)


def test_split_v2_exports_are_public() -> None:
    """The typed v2 API is available without legacy split exports."""

    assert tl.BoundarySchema is BoundarySchema
    assert tl.SplitFeatures is SplitFeatures
    assert tl.SplitGraphIR is SplitGraphIR
    assert tl.SplitRequest is SplitRequest
    assert tl.SplitRuntime is SplitRuntime
    assert tl.prepare is prepare
    assert SplitPoint.__name__ == "SplitPoint"
    assert SplitVerificationStatus.EXACT.value == "exact"
    for name in (
        "BoundarySchema",
        "SplitFeatures",
        "SplitGraphIR",
        "SplitRequest",
        "SplitRuntime",
        "prepare",
    ):
        assert name in tl.__all__
        assert name in tl.split.__all__
    for name in ("CapabilityStatus", "ReplayOp", "ReplayProgram", "SplitCapabilityReport"):
        assert name in tl.split.__all__
    assert not hasattr(tl, "SplitSpec")
    assert not hasattr(tl, "prepare_split")
    assert not hasattr(tl.split, "SplitSpec")
    assert not hasattr(tl.split, "prepare_split")


def test_split_request_validation() -> None:
    """Typed points and feature ranges reject invalid v2 requests."""

    request = SplitRequest(point=after("relu"), features=SplitFeatures(dynamic_batch=(1, 4)))
    assert request.boundary == "after:relu"
    assert request.dynamic_batch == (1, 4)
    assert request.trainable is False
    with pytest.raises(ValueError):
        SplitRequest(point=SplitPoint("percent", 0))
    with pytest.raises(ValueError):
        SplitFeatures(dynamic_batch=(4, 2))
    with pytest.raises(ValueError):
        SplitRequest(point=after("relu"), validation="unknown")  # type: ignore[arg-type]
