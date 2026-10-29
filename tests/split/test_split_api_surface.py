"""Public API surface tests for ``torchlens.split`` v2."""

from __future__ import annotations

import warnings

import pytest
from v2_helpers import (
    prepare_fixture,
    request_fixture,
)

import torchlens as tl
from torchlens.split import (
    BoundarySchema,
    GraphPartition,
    SplitFeatures,
    SplitGraphIR,
    SplitRequest,
    SplitRuntime,
    SplitVerificationStatus,
    prepare,
)


def test_split_v2_exports_are_public() -> None:
    """The typed v2 API and replay capability objects are publicly exported."""

    assert tl.BoundarySchema is BoundarySchema
    assert tl.SplitFeatures is SplitFeatures
    assert tl.SplitGraphIR is SplitGraphIR
    assert tl.SplitRequest is SplitRequest
    assert tl.SplitRuntime is SplitRuntime
    assert tl.prepare is prepare
    assert GraphPartition.__name__ == "GraphPartition"
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


def test_split_request_validation() -> None:
    """Typed points and declared batch axes reject invalid v2 requests."""

    request = SplitRequest(GraphPartition(frozenset({"relu_1:1"})))
    assert request.boundary.startswith("partition:")
    assert request.features.batch_axes is None
    assert request.trainable is False
    assert request.placement.is_explicit is False
    with pytest.raises(TypeError):
        SplitRequest("invalid")
    with pytest.raises(ValueError):
        SplitFeatures(batch_axes={"bad-pointer": 0})
    with pytest.raises(TypeError):
        SplitFeatures(batch_axes={"/args/0": "0"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        SplitRequest(GraphPartition(frozenset()), validation="unknown")  # type: ignore[arg-type]


def test_split_named_state_exchange_refuses_unsupported_backend() -> None:
    """Backends without named state exchange refuse instead of exporting stale model state."""

    import torch

    from torchlens.split.errors import SplitUnsupportedError

    runtime = prepare_fixture(
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()),
        torch.ones(2, 4),
        request_fixture(selection="through:" + "1"),
    )
    for method, args in ((runtime.state_dict, ()), (runtime.load_state_dict, ({},))):
        with pytest.raises(SplitUnsupportedError) as exc:
            method(*args)
        assert exc.value.context.reason == "named_state_exchange_unsupported"


def test_prepare_uses_supported_capture_options() -> None:
    """Ordinary split preparation and replay do not emit deprecated capture API warnings."""

    import torch

    from torchlens._deprecations import TorchLensDeprecationWarning

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()).eval()
    inputs = torch.ones(2, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensDeprecationWarning)
        runtime = prepare_fixture(model, inputs, request_fixture(selection="through:" + "1"))
        torch.testing.assert_close(runtime.replay(inputs), model(inputs))
