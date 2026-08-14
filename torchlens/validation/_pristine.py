"""Pristine-torch context for wrap-state-independent validation oracles.

R75-1 (b9-fable round-2, probe-proven): once torch is wrapped, EVERY
in-process value oracle -- the phase-0 ground-truth forward, the per-op
replay (which calls the wrapper-closure ``canonical_capture_callable``),
and backward validation's "stock autograd" forward -- observes through the
same installed wrapper shells. A TorchLens-induced numeric distortion (a
planted 0.1% ``tanh`` distortion in the wrapper layer) therefore validated
CLEAN in a wrapped process and was caught only in a fresh one: the verdict
on the identical model and bug depended on whether any capture had run
earlier in the process.

The fix: the ground-truth oracles run with the wrappers REMOVED
(``unwrap_torch``) and reinstalled afterwards. ``unwrap_torch`` restores
each callable from the append-only ``_decorated_to_orig`` ledger, whose
orig side is recorded by the wrap machinery BEFORE the decorator sees the
function -- so a corrupted decorator cannot poison the restoration, which
is exactly what makes this a genuinely independent observation root.
Re-install is cheap (cached wrapper maps; no re-decoration).

Residual, disclosed: the per-op replay still executes the RECORDED capture
callable, so replay alone remains wrap-state-correlated -- but with the
phase-0 ground truth pristine, a wrapper-layer distortion now diverges
from ground truth and fails validation regardless of process wrap state,
which closes the probe-proven class.

If the wrappers cannot be removed (a capture is active on this thread --
only reachable by calling validation from inside a traced forward), the
caller must REFUSE to bless rather than silently validating through the
wrapper root; ``pristine_torch_oracle`` propagates the typed
``CaptureContextError`` for the caller to convert into a fail-closed
verdict.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def pristine_torch_oracle() -> Iterator[bool]:
    """Run the enclosed oracle forward on pristine (unwrapped) torch.

    Yields
    ------
    bool
        ``True`` when the block runs on pristine torch: either torch was
        never wrapped in this process, or the wrappers were removed for the
        block and are reinstalled on exit (with the process-level escape
        detector and completeness-witness modes preserved -- teardown resets
        them, so they are snapshotted and re-applied).

    Raises
    ------
    torchlens.errors.CaptureContextError
        If a capture is currently active, propagated from ``unwrap_torch``.
        Callers convert this into a fail-closed (never-bless) verdict.
    """

    from .. import _state

    if not _state._is_decorated:
        yield True
        return

    detector_mode = _state._escape_detector_mode
    witness_mode = _state._completeness_witness_mode

    from ..backends.torch.wrappers import unwrap_torch, wrap_torch

    unwrap_torch()
    try:
        yield True
    finally:
        wrap_torch(
            escape_detector=detector_mode,
            completeness_witness=witness_mode,
        )
