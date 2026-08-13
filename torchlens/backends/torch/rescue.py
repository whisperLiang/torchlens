"""Rescue re-run: recover escaped ops with a TorchFunctionMode net.

Stage-2 safety net (tri-lab verdict, 2026-08-12). The primary capture is
ALWAYS mode-free: an armed ``TorchFunctionMode`` flips torch's fused fast
paths (eval MultiheadAttention 3 ops -> 27 ops, output no longer
byte-identical), so arming during a normal capture would change what
TorchLens records. Instead, when a completed capture carries an ESCAPE
SIGNAL — an escape-detector diagnostic, the unattributed-tensor-args
provenance flag, or a typed output-attribution failure — the capture is
re-run once with the net armed. The net redirects any stale pre-wrap torch
function reference to its exact wrapper (``_orig_to_decorated``), so
recovered ops have full wrapper fidelity: torch pops the mode inside the
handler, so the redirect needs no dedup token (verified experimentally).

Honesty contract:

- A rescued capture is disclosed: ``capture_verified=False``,
  ``capture_verification_reason="mode_rescue_rerun"``, and a session-time
  ``rescue_rerun`` record (never portable — the forward ran twice and mode
  presence may de-fuse fast paths).
- A rescue that recovers nothing returns the PRIMARY (mode-free) trace,
  marked ``capture_verified=False`` with reason
  ``"escape_rescue_unrecovered"`` — the residual classes (worker-thread
  stale refs, de-moded composite interiors) are declared and disclosed,
  never silent.
- A non-re-runnable capture (streaming, halt predicates) skips the rescue
  and reports the escape as before.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable

from torch.overrides import TorchFunctionMode

from ... import _state
from ..._errors import OutputAttributionError
from ...utils.rng import log_current_rng_states, set_rng_from_saved_states

if TYPE_CHECKING:
    from ...data_classes import Trace

__all__ = ["RescueTorchFunctionMode", "capture_with_rescue"]


_thread_local = threading.local()

_rescue_active = False
"""Reentrancy guard: a rescue re-run must never trigger a nested rescue."""


class RescueTorchFunctionMode(TorchFunctionMode):
    """Redirect stale original-torch-function calls to their wrappers.

    Armed ONLY during a rescue re-run. When a call reaches the mode with a
    function TorchLens wrapped (a stale pre-wrap reference — the wrapped
    namespaces call wrappers directly), it is redirected to the wrapper so
    the op is logged exactly like a normal capture. Everything else passes
    through untouched. The handler gates on ``_state._logging_enabled`` so
    TorchLens-internal tensor work (postprocess, ``pause_logging`` regions)
    never redirects.
    """

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if getattr(_thread_local, "busy", False) or not _state._logging_enabled:
            return func(*args, **kwargs)
        decorated = _state._orig_to_decorated.get(id(func))
        if decorated is None:
            return func(*args, **kwargs)
        _thread_local.busy = True
        try:
            return decorated(*args, **kwargs)
        finally:
            _thread_local.busy = False


def _escape_signal(trace: "Trace") -> str | None:
    """Return the escape-signal kind carried by a finished trace, if any."""

    if getattr(trace, "escape_diagnostics", None):
        return "escape_detector_diagnostic"
    if getattr(trace, "_had_unattributed_tensor_args", False):
        return "unattributed_tensor_args"
    return None


def _op_name_counts(trace: "Trace") -> Counter[str]:
    """Multiset of canonicalized op func names for recovery comparison.

    Mode presence respells tensor dunders through the override protocol
    (``__add__`` -> ``add``, the pinned stage-0 delta), so spellings are
    canonicalized by stripping underscores before diffing — otherwise every
    operator-using model would read as a false "recovery".
    """

    return Counter(
        name.strip("_")
        for op in getattr(trace, "ops", ())
        if (name := getattr(op, "func_name", None))
    )


def _disclosure(
    *,
    trigger: str,
    recovered: bool,
    recovered_ops: tuple[str, ...] = (),
    primary_escape_diagnostics: tuple[Any, ...] = (),
    primary_error: str | None = None,
    rescue_error: str | None = None,
    residual_signal: str | None = None,
) -> dict[str, Any]:
    """Build the session-time ``rescue_rerun`` disclosure record."""

    return {
        "trigger": trigger,
        "recovered": recovered,
        "recovered_ops": recovered_ops,
        "primary_escape_diagnostics": primary_escape_diagnostics,
        "primary_error": primary_error,
        "rescue_error": rescue_error,
        "residual_signal": residual_signal,
        "forward_runs": 2,
    }


def _mark(trace: "Trace", reason: str, info: dict[str, Any]) -> None:
    """Stamp the rescue disclosure onto a trace (session-time facts)."""

    trace.capture_verified = False
    trace.capture_verification_reason = reason
    trace.rescue_rerun = info


def capture_with_rescue(
    run_capture: Callable[[], "Trace"],
    *,
    eligible: bool = True,
) -> "Trace":
    """Run one capture; on an escape signal, re-run once with the net armed.

    Parameters
    ----------
    run_capture:
        Zero-argument callable performing exactly one full capture with
        identical configuration each call.
    eligible:
        Whether a rescue re-run is permitted. Streaming saves, sinks, and
        halt-predicate partials are not re-runnable; they report the escape
        and skip the rescue.

    Returns
    -------
    Trace
        The rescued trace when the re-run recovered ops (disclosed with
        reason ``"mode_rescue_rerun"``); otherwise the primary trace, marked
        ``"escape_rescue_unrecovered"`` when a signal fired, unchanged when
        no signal fired.
    """

    global _rescue_active
    if _rescue_active or not eligible:
        return run_capture()

    rng_snapshot = log_current_rng_states()
    primary: "Trace | None" = None
    primary_error: OutputAttributionError | None = None
    try:
        primary = run_capture()
    except OutputAttributionError as exc:
        primary_error = exc

    if primary_error is not None:
        trigger = "output_attribution_failed"
    else:
        assert primary is not None
        signal = _escape_signal(primary)
        if signal is None:
            return primary
        trigger = signal

    _rescue_active = True
    try:
        set_rng_from_saved_states(rng_snapshot)
        with RescueTorchFunctionMode():
            rescued = run_capture()
    except Exception as exc:
        if primary_error is not None:
            raise primary_error from None
        assert primary is not None
        _mark(
            primary,
            "escape_rescue_unrecovered",
            _disclosure(
                trigger=trigger,
                recovered=False,
                primary_escape_diagnostics=tuple(getattr(primary, "escape_diagnostics", ()) or ()),
                rescue_error=f"{type(exc).__name__}: {exc}",
            ),
        )
        return primary
    finally:
        _rescue_active = False

    if primary_error is not None:
        # The primary could not even attribute its output; a completed rescue
        # capture is the recovery by definition.
        _mark(
            rescued,
            "mode_rescue_rerun",
            _disclosure(
                trigger=trigger,
                recovered=True,
                primary_error=str(primary_error),
                residual_signal=_escape_signal(rescued),
            ),
        )
        return rescued

    assert primary is not None
    recovered_counts = _op_name_counts(rescued) - _op_name_counts(primary)
    if recovered_counts:
        _mark(
            rescued,
            "mode_rescue_rerun",
            _disclosure(
                trigger=trigger,
                recovered=True,
                recovered_ops=tuple(sorted(recovered_counts.elements())),
                primary_escape_diagnostics=tuple(getattr(primary, "escape_diagnostics", ()) or ()),
                residual_signal=_escape_signal(rescued),
            ),
        )
        return rescued

    # The net saw nothing new: a residual class beyond any mode (worker
    # thread, de-moded composite interior). Keep the mode-free primary and
    # disclose that the escape stands unrecovered.
    _mark(
        primary,
        "escape_rescue_unrecovered",
        _disclosure(
            trigger=trigger,
            recovered=False,
            primary_escape_diagnostics=tuple(getattr(primary, "escape_diagnostics", ()) or ()),
        ),
    )
    return primary
