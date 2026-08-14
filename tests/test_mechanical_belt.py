"""Pinned tests for the mechanical belt (protocol-invisible coverage).

The belt membership is DERIVED per build (never hand-listed): every wrapped
entry outside torch's override registries whose probe call fires zero
``TorchFunctionMode`` callbacks and touches tensors. On this build that is
exactly ``{torch.from_numpy, torch.from_dlpack, torch.frombuffer,
torch.Tensor.as_subclass}`` (``from_dlpack`` joined the wrap inventory with
the 9bea6649 inventory-gap closure); ``torch.from_file`` measures VISIBLE
here and must stay excluded (the build-dependent case the mechanical
derivation exists to settle).

NOTE: raw originals are held in function locals throughout — module-level or
``__main__``-level raw references get rewritten by the (pre-deletion)
crawler, which would silently vacuate these tests.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import belt

pytestmark = pytest.mark.smoke

_EXPECTED_MEMBERS = {
    ("torch", "from_numpy"),
    ("torch", "from_dlpack"),
    ("torch", "frombuffer"),
    ("torch.Tensor", "as_subclass"),
}


@pytest.fixture(autouse=True)
def _wrapped_torch() -> Iterator[None]:
    """Belt derivation needs installed wrappers."""
    tl.trace(torch.nn.Linear(2, 2), torch.randn(1, 2))
    yield


def test_belt_membership_is_derived_and_pinned() -> None:
    """The measured protocol-invisible set on this build, exactly."""

    report = belt.belt_report()
    assert report is not None
    assert set(report.members) == _EXPECTED_MEMBERS
    assert not report.probe_failures
    if hasattr(torch, "from_file"):
        # Build-dependent visibility: measured VISIBLE on this build, so the
        # net covers it and the belt must NOT claim it.
        assert ("torch", "from_file") in report.probed_visible
        assert ("torch", "from_file") not in report.members


def test_belt_members_fire_zero_mode_callbacks() -> None:
    """Independent re-measurement: no mode can see a belt member's call."""

    report = belt.belt_report()
    assert report is not None
    for entry in report.members:
        recipe = belt.PROBE_RECIPES[entry]
        namespace = belt._resolve_namespace(entry[0])
        current = getattr(namespace, entry[1])
        original = _state._decorated_to_orig.get(id(current), current)
        args, kwargs = recipe()
        mode = belt._CountingMode()
        with _state.pause_logging(), mode:
            original(*args, **kwargs)
        assert mode.calls == 0, f"{entry} fired the mode; it is not protocol-invisible"


def _original_and_wrapper(name: str) -> tuple[Any, Any]:
    wrapper = getattr(torch, name)
    original = _state._decorated_to_orig[id(wrapper)]
    return original, wrapper


def test_belt_sweep_patches_stale_module_ref_and_restores() -> None:
    """Module-attr stale refs to belt members round-trip through the ledger."""

    original, wrapper = _original_and_wrapper("from_numpy")
    mod = types.ModuleType("_tl_belt_sweep_check")
    mod.op = original
    sys.modules[mod.__name__] = mod
    try:
        patched = belt.sweep_stale_belt_references()
        assert patched >= 1
        assert mod.op is wrapper
        belt.restore_belt_references()
        assert mod.op is original
    finally:
        sys.modules.pop(mod.__name__, None)
        belt.restore_belt_references()


def test_belt_restore_preserves_user_reassignment() -> None:
    """Reversal is conditional: a slot the user rewrote is left alone."""

    original, wrapper = _original_and_wrapper("frombuffer")
    mod = types.ModuleType("_tl_belt_reassign_check")
    mod.op = original
    sys.modules[mod.__name__] = mod
    try:
        belt.sweep_stale_belt_references()
        assert mod.op is wrapper
        sentinel = object()
        mod.op = sentinel
        belt.restore_belt_references()
        assert mod.op is sentinel
    finally:
        sys.modules.pop(mod.__name__, None)
        belt.restore_belt_references()


def test_belt_sweep_is_epoch_incremental() -> None:
    """A module identity is scanned once; new imports are picked up later."""

    original, wrapper = _original_and_wrapper("from_numpy")
    belt.sweep_stale_belt_references()  # drain: everything live is now swept
    late = types.ModuleType("_tl_belt_late_import")
    late.op = original
    sys.modules[late.__name__] = late
    try:
        assert belt.sweep_stale_belt_references() >= 1
        assert late.op is wrapper
    finally:
        sys.modules.pop(late.__name__, None)
        belt.restore_belt_references()
