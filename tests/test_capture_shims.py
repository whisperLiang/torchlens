"""Regression tests for the deprecated capture/IR compatibility shims.

The kernel/ledger/projector layer and the ``ModuleEvent``/``BufferEvent``
event kinds were deleted in the backend migration; these shims keep the
historical imports resolving. Contract: importable, warn ONCE per process,
inert (behavioral entry points raise; nothing in production constructs them).
"""

import importlib
import sys
import warnings

import pytest


def test_kernel_shim_import_warns_once() -> None:
    """A fresh ``torchlens.capture.kernel`` import warns exactly once.

    The deprecation warning fires on first import only; the cached module
    re-import must stay silent (warn-once-per-process contract).
    """
    sys.modules.pop("torchlens.capture.kernel", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("torchlens.capture.kernel")
        importlib.import_module("torchlens.capture.kernel")  # cached: silent
    shim_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
        and "torchlens.capture.kernel is deprecated" in str(warning.message)
    ]
    assert len(shim_warnings) == 1, "kernel shim must warn exactly once per process"


def test_kernel_shim_entry_points_raise() -> None:
    """Every behavioral CaptureKernel entry point raises the removed-layer error."""
    from torchlens.capture import kernel

    capture_kernel = kernel.CaptureKernel(session=object())
    observation = kernel.OpObservation(operation_key="linear_1_1", value=None)
    for entry_point, invoke in (
        ("process", lambda: capture_kernel.process(observation)),
        ("emit", lambda: capture_kernel.emit("linear_1_1", lambda: None)),
        ("apply_intervention", lambda: capture_kernel.apply_intervention(observation, lambda o: o)),
        ("begin_observation", lambda: capture_kernel.begin_observation("linear_1_1")),
        ("mark_metadata", lambda: capture_kernel.mark_metadata()),
        ("mark_payload", lambda: capture_kernel.mark_payload()),
    ):
        with pytest.raises(RuntimeError, match=f"CaptureKernel.{entry_point} was removed"):
            invoke()


def test_trace_projector_shim_warns_once_and_delegates() -> None:
    """TraceProjector resolves through the shim, warning once per process."""
    from torchlens.capture import projectors

    projectors._WARNED_DEPRECATED_NAMES.discard("TraceProjector")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = projectors.TraceProjector
        second = projectors.TraceProjector
    assert first is second is projectors._DeprecatedTraceProjector
    shim_warnings = [
        warning for warning in caught if issubclass(warning.category, DeprecationWarning)
    ]
    assert len(shim_warnings) == 1, "shim must warn exactly once per process"


def test_ir_event_shims_warn_once_and_stay_inert() -> None:
    """ModuleEvent/BufferEvent resolve as inert shims, warning once per name."""
    import torchlens.ir as tl_ir

    for name in ("ModuleEvent", "BufferEvent"):
        tl_ir._WARNED_DEPRECATED_NAMES.discard(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = getattr(tl_ir, name)
            second = getattr(tl_ir, name)
        assert first is second
        shim_warnings = [
            warning for warning in caught if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(shim_warnings) == 1, f"{name} must warn exactly once per process"


def test_ir_shim_unknown_attribute_still_raises() -> None:
    """The warn-once path never swallows genuine attribute errors."""
    import torchlens.ir as tl_ir
    from torchlens.capture import projectors

    with pytest.raises(AttributeError):
        tl_ir.DefinitelyNotARealExport  # noqa: B018
    with pytest.raises(AttributeError):
        projectors.DefinitelyNotARealExport  # noqa: B018
