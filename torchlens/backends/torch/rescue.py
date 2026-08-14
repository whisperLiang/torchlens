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
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from torch.overrides import TorchFunctionMode

from ... import _state
from ..._errors import OutputAttributionError
from ...utils.rng import log_current_rng_states, set_rng_from_saved_states

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = [
    "CaptureAttemptFailedWarning",
    "RescueTorchFunctionMode",
    "capture_with_rescue",
]


class CaptureAttemptFailedWarning(RuntimeWarning):
    """Warning category for the one capture-attempt-failed advisory.

    A dedicated category (still a ``RuntimeWarning``, so user filters keep
    matching) lets the rescue driver DEFER the advisory while a rescue re-run
    is still possible: the warning tells the user diagnostics ride the
    exception (``exc.partial_log``), which is only truthful when that
    exception actually propagates. A successful rescue swallows the failure,
    so the deferred advisory is dropped; every path that re-raises flushes it
    first.
    """


_thread_local = threading.local()
"""Per-thread rescue state: the mode's ``busy`` token and the re-run guard.

Both are thread-local for the same reason. ``busy`` guards reentrancy of a
handler that fires on EVERY thread's torch calls. ``rescue_active`` guards
against a rescue re-run triggering a nested rescue, which is a property of one
call stack -- and as a process global it also let an unrelated thread's capture
inherit the suppression and silently lose its own safety net, while a capture
that is mid-rescue is doing model prep outside ``active_logging`` where the
admission refusal does not apply.
"""


def _rescue_is_active() -> bool:
    """Return whether this thread is already inside a rescue re-run.

    Returns
    -------
    bool
        ``True`` while this thread runs a rescue re-run.
    """

    return bool(getattr(_thread_local, "rescue_active", False))


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


@contextmanager
def _record_emitted_warnings(seen: set[tuple[type, str]]) -> Iterator[None]:
    """Record every warning shown during the block, still forwarding it.

    One user ``tl.trace()`` call may run the forward twice (primary + rescue
    re-run); per-session advisory warnings (functorch boundary, provenance)
    must reach the user ONCE per trace call, not once per forward.
    """

    forward = warnings.showwarning

    def recorder(message: Any, category: Any, *args: Any, **kwargs: Any) -> None:
        """Record ``(category, message)`` in ``seen``, then forward to the real handler."""

        seen.add((category, str(message)))
        forward(message, category, *args, **kwargs)

    warnings.showwarning = recorder
    try:
        yield
    finally:
        warnings.showwarning = forward


@contextmanager
def _suppress_repeated_warnings(seen: set[tuple[type, str]]) -> Iterator[None]:
    """Drop warnings already emitted by the primary run; forward novel ones."""

    forward = warnings.showwarning

    def dedup(message: Any, category: Any, *args: Any, **kwargs: Any) -> None:
        """Forward only warnings whose ``(category, message)`` is not already in ``seen``."""

        if (category, str(message)) in seen:
            return
        forward(message, category, *args, **kwargs)

    warnings.showwarning = dedup
    try:
        yield
    finally:
        warnings.showwarning = forward


@contextmanager
def _defer_capture_failed_warnings(
    deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]],
) -> Iterator[None]:
    """Hold back capture-attempt-failed advisories; forward everything else.

    The advisory points the user at ``exc.partial_log`` — truthful only when
    the exception propagates. While a rescue re-run may still swallow the
    failure, the advisory is parked in ``deferred``; the driver flushes it on
    every re-raising path and drops it when the rescue succeeds.
    """

    forward = warnings.showwarning

    def hold(message: Any, category: Any, *args: Any, **kwargs: Any) -> None:
        """Park capture-failed advisories in ``deferred``; forward the rest."""

        if isinstance(category, type) and issubclass(category, CaptureAttemptFailedWarning):
            deferred.append((message, category, args, kwargs))
            return
        forward(message, category, *args, **kwargs)

    warnings.showwarning = hold
    try:
        yield
    finally:
        warnings.showwarning = forward


def _flush_deferred_warnings(
    deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]],
) -> None:
    """Re-emit parked advisories through the current warning handler."""

    for message, category, args, kwargs in deferred:
        warnings.showwarning(message, category, *args, **kwargs)
    deferred.clear()


def _escape_signal(trace: Trace) -> str | None:
    """Return the escape-signal kind carried by a finished trace, if any.

    An authoritative POSITIVE verdict outranks the heuristic provenance
    flag: when the armed dispatch witness accounted for every dispatch and
    verified the capture (e.g. an ``autograd.grad`` boundary is a known
    no-provenance source), the flag is a false alarm and no rescue runs.
    """

    if getattr(trace, "capture_verified", None) is True:
        return None
    if getattr(trace, "escape_diagnostics", None):
        return "escape_detector_diagnostic"
    if getattr(trace, "_had_unattributed_tensor_args", False):
        return "unattributed_tensor_args"
    return None


def _op_name_counts(trace: Trace) -> Counter[str]:
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
    lost_ops: tuple[str, ...] = (),
    primary_escape_diagnostics: tuple[Any, ...] = (),
    primary_error: str | None = None,
    rescue_error: str | None = None,
    residual_signal: str | None = None,
    skipped_reason: str | None = None,
    forward_runs: int = 2,
) -> dict[str, Any]:
    """Build the session-time ``rescue_rerun`` disclosure record."""

    return {
        "trigger": trigger,
        "recovered": recovered,
        "recovered_ops": recovered_ops,
        "lost_ops": lost_ops,
        "primary_escape_diagnostics": primary_escape_diagnostics,
        "primary_error": primary_error,
        "rescue_error": rescue_error,
        "residual_signal": residual_signal,
        "skipped_reason": skipped_reason,
        "forward_runs": forward_runs,
    }


def _buffer_write_labels(trace: Trace) -> tuple[str, ...]:
    """Labels of primary-forward ops that ACTUALLY wrote module buffer state.

    A rescue re-run executes the user's forward a SECOND time. When the
    primary forward wrote buffers (train-mode BatchNorm running stats and
    ``num_batches_tracked``, any in-forward buffer counter), the re-run
    double-applies those writes: RNG is restored between runs, module state is
    not restorable. Captures whose primary shows buffer writes therefore
    refuse the re-run. A custom in-forward PYTHON-attribute counter (not a
    registered buffer) still mutates twice on rescued captures -- the
    documented residual (see docs/migration/scoped_detached_patching.md).

    The refusal keys on an ACTUAL write, not on journal presence: fused norm
    mutators (``batch_norm``, ``instance_norm``, ``native_group_norm``) are
    journaled unconditionally, so every EVAL-mode BN/IN/GN capture carries
    ``buffer_write_kind`` records whose ``buffer_value_changed`` is ``False``
    (bytes provably unchanged; re-running is state-neutral). Only a record
    whose value changed -- or whose change status is unknown (fail closed) --
    refuses the re-run.
    """

    labels: list[str] = []
    for op in getattr(trace, "ops", ()) or ():
        if getattr(op, "buffer_write_kind", None) is None:
            continue
        if getattr(op, "buffer_value_changed", None) is False:
            continue
        label = getattr(op, "label_raw", None) or getattr(op, "layer_label", None)
        labels.append(str(label or getattr(op, "func_name", "?")))
    return tuple(labels)


def _mark(trace: Trace, reason: str, info: dict[str, Any]) -> None:
    """Stamp the rescue disclosure onto a trace (session-time facts).

    An unrecovered escape must never SILENCE a more specific verdict: when
    the primary already carries a verification reason (dispatch witness,
    shadow detector, dynamo boundary), that reason stays authoritative and
    the rescue attempt is disclosed only through ``rescue_rerun``. The
    ``escape_rescue_unrecovered`` reason is reserved for the formerly-silent
    class where the primary made no claim at all.
    """

    trace.capture_verified = False
    existing_reason = getattr(trace, "capture_verification_reason", None)
    if getattr(trace, "_raw_dynamo_region_detected", False) or existing_reason == (
        "dynamo_region_not_logged"
    ):
        # R16-3: the dynamo-region verdict has TOP precedence at both finalize
        # sites (compile threads and unaccounted dispatches are symptoms of
        # that same region); a recovered rescue must not clobber it. The
        # rescue attempt stays disclosed through ``rescue_rerun`` below.
        pass
    elif reason == "mode_rescue_rerun" or not existing_reason:
        trace.capture_verification_reason = reason
    trace.rescue_rerun = info


def capture_with_rescue(
    run_capture: Callable[[], Trace],
    *,
    eligible: bool = True,
) -> Trace:
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

    if _rescue_is_active() or not eligible:
        return run_capture()

    rng_snapshot = log_current_rng_states()
    primary: Trace | None = None
    primary_error: OutputAttributionError | None = None
    emitted_warnings: set[tuple[type, str]] = set()
    primary_deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]] = []
    try:
        with (
            _record_emitted_warnings(emitted_warnings),
            _defer_capture_failed_warnings(primary_deferred),
        ):
            primary = run_capture()
    except OutputAttributionError as exc:
        primary_error = exc
    except BaseException:
        # No rescue for this failure class: the parked advisory is truthful
        # (the exception propagates with its diagnostics), so re-emit it.
        _flush_deferred_warnings(primary_deferred)
        raise

    if primary_error is not None:
        trigger = "output_attribution_failed"
    else:
        assert primary is not None
        signal = _escape_signal(primary)
        if signal is None:
            return primary
        trigger = signal
        # R16-2: a rescue re-run executes the user's forward a SECOND time.
        # When the primary forward WROTE buffer state (train-mode BatchNorm
        # counters and running stats, in-forward buffer counters), the re-run
        # double-applies those writes (RNG is restored, module state is not),
        # so the re-run is refused and the escape stands disclosed.
        buffer_writes = _buffer_write_labels(primary)
        if buffer_writes:
            shown = ", ".join(buffer_writes[:3])
            warnings.warn(
                "TorchLens detected an escape signal but skipped the rescue "
                f"re-run: the forward wrote module buffer state ({shown}), and "
                "re-running it would double-apply those writes. The escape "
                "stands unrecovered; call model.eval() (or fix the stale torch "
                "reference) and re-capture.",
                UserWarning,
                stacklevel=3,
            )
            _mark(
                primary,
                "escape_rescue_unrecovered",
                _disclosure(
                    trigger=trigger,
                    recovered=False,
                    primary_escape_diagnostics=tuple(
                        getattr(primary, "escape_diagnostics", ()) or ()
                    ),
                    skipped_reason="buffer_writes_double_forward",
                    forward_runs=1,
                ),
            )
            return primary

    _thread_local.rescue_active = True
    rescue_deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]] = []
    try:
        set_rng_from_saved_states(rng_snapshot)
        # The rescue run's own capture-failed advisory is deferred too: its
        # exception never propagates (the primary's error or trace does), so
        # an advisory pointing at ITS exc.partial_log would always be untrue.
        with (
            _suppress_repeated_warnings(emitted_warnings),
            _defer_capture_failed_warnings(rescue_deferred),
            RescueTorchFunctionMode(),
        ):
            rescued = run_capture()
    except Exception as exc:
        if primary_error is not None:
            # The primary's failure propagates with its diagnostics attached,
            # so its parked advisory is truthful again — re-emit it.
            _flush_deferred_warnings(primary_deferred)
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
        _thread_local.rescue_active = False

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
    # R16-1: the recovery oracle is TWO-SIDED. ``Counter.__sub__`` alone drops
    # losses, so mode-induced de-fusion (eval MHA: 3 fused ops -> 27 small
    # ops) read as pure gains and a benign false alarm silently swapped the
    # user's canonical fused trace for a structurally different
    # mode-perturbed one marked recovered. A rescue counts as recovery ONLY
    # when the rescued op multiset is a strict SUPERSET of the primary's;
    # any loss means mode perturbation, and the mode-free primary stays
    # authoritative with both deltas disclosed.
    primary_counts = _op_name_counts(primary)
    rescued_counts = _op_name_counts(rescued)
    recovered_counts = rescued_counts - primary_counts
    lost_counts = primary_counts - rescued_counts
    if recovered_counts and not lost_counts:
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

    # Nothing new, or a structurally different (mode-perturbed) graph: keep
    # the mode-free primary and disclose that the escape stands unrecovered,
    # including both deltas and whether the rescue still carried the signal.
    _mark(
        primary,
        "escape_rescue_unrecovered",
        _disclosure(
            trigger=trigger,
            recovered=False,
            recovered_ops=tuple(sorted(recovered_counts.elements())),
            lost_ops=tuple(sorted(lost_counts.elements())),
            primary_escape_diagnostics=tuple(getattr(primary, "escape_diagnostics", ()) or ()),
            residual_signal=_escape_signal(rescued),
        ),
    )
    return primary
