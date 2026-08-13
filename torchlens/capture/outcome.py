"""The capture-outcome authority: one typed lattice for every capture product.

This module is the SINGLE authority for terminal capture truth (early-stopping
unification design of record, v4). It owns:

- the frozen status vocabulary (:class:`CaptureStatus`, :class:`CapturePhase`,
  :class:`FailureOrigin`) and the durable :class:`CaptureOutcome` record;
- the persisted attestation codec (string-only payload, closed-vocabulary
  parse) and the status-specific load coherence matrix;
- the evidence-based derivation lattice for artifacts without attestation;
- the per-outcome capability table and its ONE enforcement chokepoint;
- the diagnostic failure-origin classifier.

Settlement adapters (``settle_capture``, ``demote_outcome``, the cook /
recorder / backend-finalize stamps) also live here so no second writer can
exist. Consumers branch on the typed record or the stable refusal codes,
never on exception text.
"""

from __future__ import annotations

import traceback
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..errors import CaptureError, TorchLensError

if TYPE_CHECKING:
    pass


class CaptureStatus(str, Enum):
    """Terminal status vocabulary for one capture product."""

    COMPLETE = "complete"
    HALTED = "halted"
    ABORTED_NONFINITE = "aborted_nonfinite"
    FAILED = "failed"
    UNATTESTED = "unattested"
    UNKNOWN = "unknown"


class CapturePhase(str, Enum):
    """Failure phase attribution (FAILED outcomes only)."""

    FORWARD = "forward"
    FINALIZE = "finalize"
    POSTPROCESS = "postprocess"
    TEARDOWN = "teardown"


class FailureOrigin(str, Enum):
    """Diagnostic failure origin (FAILED outcomes only; never capability-steering)."""

    USER_OP = "user_op"
    TORCHLENS = "torchlens"
    INTERRUPT = "interrupt"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    """Durable, string-only terminal outcome record for one capture product.

    Parameters
    ----------
    status:
        Settled terminal status.
    phase:
        Failure phase (FAILED only).
    origin:
        Diagnostic failure origin (FAILED only); never steers capability.
    reason:
        Halt reason / abort reason / failure summary, string-only.
    error_type:
        Qualified exception type name for FAILED outcomes.
    boundary_kind:
        Stop-boundary event kind (halt/nonfinite boundaries).
    boundary_label:
        Stop-boundary op label (raw where postprocess never ran).
    frontier_labels:
        Halt-frontier labels (final labels when postprocess ran, else raw).
    n_ops_committed:
        Count of committed op-kind lane entries at settlement.
    inference_only:
        Whether the capture ran under the inference-only public mode.
    recovered:
        True for disk-recovered fastlog products.
    derived:
        True when reconstructed from structural evidence, not a settle stamp.
    settlement_note:
        Secondary-failure / demotion / provenance disclosure, string-only.
    """

    status: CaptureStatus
    phase: CapturePhase | None = None
    origin: FailureOrigin | None = None
    reason: str | None = None
    error_type: str | None = None
    boundary_kind: str | None = None
    boundary_label: str | None = None
    frontier_labels: tuple[str, ...] | None = None
    n_ops_committed: int | None = None
    inference_only: bool = False
    recovered: bool = False
    derived: bool = False
    settlement_note: str | None = None

    @property
    def partial(self) -> bool | None:
        """Return the honest partiality claim for this outcome.

        ``True`` for a known prefix or failure, ``False`` for settled
        complete, ``None`` where partiality is unknowable (UNATTESTED /
        UNKNOWN claims nothing).
        """

        if self.status is CaptureStatus.COMPLETE:
            return False
        if self.status in (
            CaptureStatus.HALTED,
            CaptureStatus.ABORTED_NONFINITE,
            CaptureStatus.FAILED,
        ):
            return True
        return None

    def to_payload(self) -> dict[str, Any]:
        """Return the persisted string-only payload for this record."""

        return {
            "status": self.status.value,
            "phase": None if self.phase is None else self.phase.value,
            "origin": None if self.origin is None else self.origin.value,
            "reason": self.reason,
            "error_type": self.error_type,
            "boundary_kind": self.boundary_kind,
            "boundary_label": self.boundary_label,
            "frontier_labels": (
                None if self.frontier_labels is None else list(self.frontier_labels)
            ),
            "n_ops_committed": self.n_ops_committed,
            "inference_only": self.inference_only,
            "recovered": self.recovered,
            "derived": self.derived,
            "settlement_note": self.settlement_note,
        }


class CaptureOutcomeError(TorchLensError):
    """Typed capability refusal from the capture-outcome chokepoint.

    ``fields["code"]`` carries the stable gate code (``"N1"``..``"N5"``);
    branch on it, never on message text.
    """


class StopSignalSwallowedError(CaptureError):
    """A halt/nonfinite stop request was swallowed by user code (F6).

    Raised at the capture boundary when the stop-request latch is set but the
    forward returned normally: a user ``except``/``except BaseException`` ate
    the control signal, so the capture can never be blessed COMPLETE.
    """


@dataclass(frozen=True, slots=True)
class StopRequest:
    """One latched stop request (F6), set on the active trace at raise time.

    Parameters
    ----------
    kind:
        ``"halt"`` or ``"nonfinite"``.
    reason:
        The stop reason recorded at the raise site.
    boundary_kind:
        Event kind at the stop boundary, when known.
    boundary_label:
        Op label at the stop boundary, when known.
    """

    kind: str
    reason: str | None = None
    boundary_kind: str | None = None
    boundary_label: str | None = None


# ---------------------------------------------------------------------------
# Capability table + chokepoint
# ---------------------------------------------------------------------------

_ALLOW = "allow"

CAPTURE_OUTCOME_CAPABILITIES: dict[str, dict[CaptureStatus, str]] = {
    # N2 -- validation ENTRY refusal only; tripwire bodies stay byte-untouched.
    # UNATTESTED deliberately keeps entry OPEN: validation is the tripwire that
    # catches a freeze-corrupted legacy representation, so refusing it would
    # disarm the check for exactly the most-suspect artifacts.
    "validation_entry": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: "allow_scoped:halted ground-truth output block skipped",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N2",
        CaptureStatus.FAILED: "refuse:N2",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N2",
    },
    # N1 -- analysis-save refusal. Replaces today's ungated pass-through of
    # provably-unfinished artifacts; HALTED and UNATTESTED re-export stay open.
    "save_analysis": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N1",
        CaptureStatus.FAILED: "refuse:N1",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N1",
    },
    # N4 -- runnable-save refusal for HALTED (fail-closed default, JMT veto).
    # ABORTED/FAILED/UNKNOWN are subsumed by N1 at the same entry.
    "save_runnable": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: "refuse:N4",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N1",
        CaptureStatus.FAILED: "refuse:N1",
        CaptureStatus.UNATTESTED: "allow_scoped:existing runnable preflights are the gate",
        CaptureStatus.UNKNOWN: "refuse:N1",
    },
    # N5 (HALTED) / N3 (ABORTED/FAILED/UNKNOWN) -- LIVE providers only:
    # refresh, save_new_outs, push, replay, rerun, run() on a live trace, and
    # fast=True. The recorded prefix cannot re-drive a full native forward.
    "live_replay": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: "refuse:N5",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
    # Loaded-sparse run() executes exactly the recorded taken-path prefix DAG
    # under pause_logging(): the HALTED failure mode cannot occur there.
    "loaded_sparse_run": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
    # log_backward / backward / recording_backward: the autograd graph of a
    # halted capture IS the captured prefix, so HALTED stays allowed.
    "backward": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
}
"""Frozen per-outcome capability cells; consumers branch ONLY through
:func:`require_capture_capability`. Cells are ``"allow"``,
``"allow_scoped:<note>"``, or ``"refuse:<code>"``."""


_REFUSAL_HINTS: dict[str, str] = {
    "N1": (
        "This capture did not settle as a saveable product. Inspect it live "
        "(draw, report.explain, tl.partial.from_failed_capture) instead of "
        "exporting it."
    ),
    "N2": (
        "Validation entry requires a settled COMPLETE, HALTED, or legacy "
        "finished capture; a failed or unproven capture has no replayable "
        "ground truth to validate against."
    ),
    "N3": (
        "Replay and backward surfaces require a structurally complete "
        "captured graph; this capture's graph is failed or unproven."
    ),
    "N4": (
        "A halted capture records a PREFIX of the forward; the sparse "
        "runnable contract requires the complete taken path. Re-capture "
        "without halt= to save a runnable artifact."
    ),
    "N5": (
        "Live-provider replay re-drives the FULL native forward, but a "
        "halted capture recorded only a prefix; results would silently "
        "disagree with the recorded graph. Re-capture without halt= "
        "(re-arming raise_on_nan/halt options as needed), or load the "
        "artifact and use loaded-sparse run()."
    ),
}


def outcome_for(trace: object) -> "CaptureOutcome | None":
    """Return the settled outcome sidecar attached to ``trace``, if any."""

    outcome = getattr(trace, "__dict__", {}).get("_capture_outcome")
    return outcome if isinstance(outcome, CaptureOutcome) else None


def require_capture_capability(
    trace: object,
    capability: str,
    *,
    detail: str | None = None,
) -> CaptureOutcome:
    """Enforce one capability cell for ``trace``'s settled outcome.

    Parameters
    ----------
    trace:
        Product whose ``_capture_outcome`` sidecar is consulted. A live trace
        with no sidecar is treated as UNKNOWN with a warning (fail-closed
        defense-in-depth; unreachable for shipped constructors per the P0
        settlement inventory).
    capability:
        Row key in :data:`CAPTURE_OUTCOME_CAPABILITIES`.
    detail:
        Optional gate-site context appended to the refusal message.

    Returns
    -------
    CaptureOutcome
        The consulted outcome, for callers that scope behavior on it.

    Raises
    ------
    CaptureOutcomeError
        When the cell refuses; ``fields["code"]`` carries the stable code.
    """

    outcome = outcome_for(trace)
    if outcome is None:
        warnings.warn(
            "TorchLens capability check found no settled capture outcome on "
            f"this {type(trace).__name__}; treating it as UNKNOWN (fail-closed). "
            "Every shipped product constructor settles an outcome, so this "
            "usually means a hand-built or pre-settlement object.",
            RuntimeWarning,
            stacklevel=3,
        )
        outcome = CaptureOutcome(status=CaptureStatus.UNKNOWN, derived=True)
    cell = CAPTURE_OUTCOME_CAPABILITIES[capability][outcome.status]
    if cell.startswith("refuse:"):
        code = cell.split(":", 1)[1]
        message = (
            f"TorchLens refuses {capability!r} for a capture with outcome "
            f"status {outcome.status.value!r} (gate {code}). "
            f"{_REFUSAL_HINTS.get(code, '')}"
        )
        if outcome.reason:
            message += f" Capture outcome reason: {outcome.reason}"
        if detail:
            message += f" {detail}"
        raise CaptureOutcomeError(
            message,
            code=code,
            capability=capability,
            status=outcome.status.value,
        )
    return outcome


# ---------------------------------------------------------------------------
# Persistence codec + load derivation
# ---------------------------------------------------------------------------


def parse_outcome_payload(payload: object) -> CaptureOutcome:
    """Parse one persisted attestation payload against the closed vocabularies.

    Parameters
    ----------
    payload:
        The persisted ``_capture_outcome`` value (a string-only dict).

    Returns
    -------
    CaptureOutcome
        Parsed record.

    Raises
    ------
    ValueError
        On any shape, type, or vocabulary violation (the caller degrades to
        UNKNOWN with a warning; a load never crashes on a bad attestation).
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"capture outcome payload must be a mapping, got {type(payload).__name__}")
    data = dict(payload)

    def _enum_or_none(key: str, enum_cls: type[Enum]) -> Any:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"capture outcome field {key!r} must be a string, got {value!r}")
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ValueError(
                f"capture outcome field {key!r} has unknown vocabulary value {value!r}"
            ) from exc

    status = _enum_or_none("status", CaptureStatus)
    if status is None:
        raise ValueError("capture outcome payload is missing 'status'")
    phase = _enum_or_none("phase", CapturePhase)
    origin = _enum_or_none("origin", FailureOrigin)

    def _str_or_none(key: str) -> str | None:
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"capture outcome field {key!r} must be a string or None")
        return value

    frontier = data.get("frontier_labels")
    if frontier is not None:
        if not isinstance(frontier, (list, tuple)) or not all(
            isinstance(item, str) for item in frontier
        ):
            raise ValueError("capture outcome field 'frontier_labels' must be a list of strings")
        frontier = tuple(frontier)
    n_ops = data.get("n_ops_committed")
    if n_ops is not None and (isinstance(n_ops, bool) or not isinstance(n_ops, int)):
        raise ValueError("capture outcome field 'n_ops_committed' must be an int or None")

    def _bool(key: str, default: bool = False) -> bool:
        value = data.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"capture outcome field {key!r} must be a bool")
        return value

    return CaptureOutcome(
        status=status,
        phase=phase,
        origin=origin,
        reason=_str_or_none("reason"),
        error_type=_str_or_none("error_type"),
        boundary_kind=_str_or_none("boundary_kind"),
        boundary_label=_str_or_none("boundary_label"),
        frontier_labels=frontier,
        n_ops_committed=n_ops,
        inference_only=_bool("inference_only"),
        recovered=_bool("recovered"),
        derived=_bool("derived"),
        settlement_note=_str_or_none("settlement_note"),
    )


def attestation_coherent(
    outcome: CaptureOutcome,
    *,
    halted: bool,
    finished: bool,
) -> bool:
    """Validate one parsed attestation against the structural evidence.

    The status-specific coherence matrix (design 2.4): a parsed attestation
    is adopted only when its status is possible given the artifact's
    structural fields, all read by TRUTHINESS.
    """

    if outcome.status is CaptureStatus.COMPLETE:
        return (not halted) and finished
    if outcome.status is CaptureStatus.HALTED:
        # The re-raise path settles attested HALTED without postprocess, so
        # either ``_tracing_finished`` value is coherent.
        return halted
    if outcome.status is CaptureStatus.ABORTED_NONFINITE:
        return not halted
    if outcome.status is CaptureStatus.FAILED:
        # A halted-postprocess secondary failure settles FAILED with
        # ``halted=True`` already stamped pre-postprocess.
        return True
    if outcome.status is CaptureStatus.UNKNOWN:
        return True
    if outcome.status is CaptureStatus.UNATTESTED:
        # Only re-saves of DERIVED outcomes may carry UNATTESTED; a settle
        # stamp can never produce it.
        return (not halted) and finished and outcome.derived
    return False


def derive_outcome_from_structural_state(state: Mapping[str, Any]) -> CaptureOutcome:
    """Derive a conservative outcome from structural evidence alone.

    The lattice (design 2.4), truthiness throughout so ``False``/``None``/
    absent are one class::

        halted and finished      -> HALTED      (derived)
        halted and not finished  -> UNKNOWN     (halted postprocess never completed)
        finished and not halted  -> UNATTESTED  (complete and tail-failure indistinguishable)
        otherwise                -> UNKNOWN
    """

    finished = bool(state.get("_tracing_finished"))
    halted = bool(state.get("halted"))
    inference_only = bool(state.get("inference_only"))
    if halted and finished:
        reason = state.get("halt_reason")
        frontier = state.get("halt_frontier")
        return CaptureOutcome(
            status=CaptureStatus.HALTED,
            reason=reason if isinstance(reason, str) else None,
            boundary_label=frontier if isinstance(frontier, str) else None,
            inference_only=inference_only,
            derived=True,
        )
    if finished and not halted:
        return CaptureOutcome(
            status=CaptureStatus.UNATTESTED,
            inference_only=inference_only,
            derived=True,
        )
    return CaptureOutcome(
        status=CaptureStatus.UNKNOWN,
        inference_only=inference_only,
        derived=True,
    )


def resolve_loaded_outcome(state: Mapping[str, Any]) -> CaptureOutcome:
    """Resolve the outcome for one loaded Trace state dict (pickle or .tlspec).

    Adopts a persisted attestation when it parses against the closed
    vocabularies AND is coherent per :func:`attestation_coherent`; otherwise
    derives from the structural lattice. Any contradiction, parse failure, or
    unknown vocabulary value degrades to UNKNOWN with ONE warning naming the
    incoherence -- fail-closed, never a load crash.
    """

    payload = state.get("_capture_outcome")
    if payload is None:
        return derive_outcome_from_structural_state(state)
    try:
        outcome = parse_outcome_payload(payload)
    except ValueError as exc:
        warnings.warn(
            f"TorchLens could not parse this artifact's capture-outcome attestation ({exc}); "
            "treating the capture outcome as UNKNOWN (fail-closed).",
            RuntimeWarning,
            stacklevel=3,
        )
        return CaptureOutcome(
            status=CaptureStatus.UNKNOWN,
            derived=True,
            settlement_note=f"attestation_parse_failed: {exc}",
        )
    halted = bool(state.get("halted"))
    finished = bool(state.get("_tracing_finished"))
    if not attestation_coherent(outcome, halted=halted, finished=finished):
        warnings.warn(
            "TorchLens found an attested capture outcome "
            f"({outcome.status.value!r}) that contradicts this artifact's "
            f"structural evidence (halted={halted}, tracing_finished={finished}); "
            "treating the capture outcome as UNKNOWN (fail-closed).",
            RuntimeWarning,
            stacklevel=3,
        )
        return CaptureOutcome(
            status=CaptureStatus.UNKNOWN,
            derived=True,
            settlement_note=(
                f"attestation_incoherent: status={outcome.status.value} "
                f"halted={halted} finished={finished}"
            ),
        )
    return outcome


# ---------------------------------------------------------------------------
# Failure-origin classification (diagnostic only)
# ---------------------------------------------------------------------------

_TORCHLENS_PKG_DIR = Path(__file__).resolve().parent.parent

# C-backed user-op failures surface with the wrapper trampoline as their
# innermost Python frame; those lines execute the USER's op, so they classify
# as USER_OP, not TORCHLENS. Matched on source text to survive line drift.
_TRAMPOLINE_SOURCE_MARKER = "out_orig = func("


def _frame_zone(filename: str) -> str:
    """Classify one frame filename as 'torchlens', 'library', or 'user'."""

    if filename.startswith("<frozen"):
        return "library"
    try:
        path = Path(filename).resolve()
    except (OSError, ValueError):
        return "library"
    try:
        path.relative_to(_TORCHLENS_PKG_DIR)
        return "torchlens"
    except ValueError:
        pass
    if "site-packages" in path.parts or "dist-packages" in path.parts:
        return "library"
    return "user"


def classify_failure_origin(exc: BaseException) -> FailureOrigin:
    """Classify one terminal exception's origin (diagnostic only).

    Walks the traceback from the innermost frame outward. The innermost
    attributable frame wins: user code -> USER_OP; a torchlens frame ->
    TORCHLENS, except the wrapper trampoline (executing the user's op) ->
    USER_OP. Interrupts classify INTERRUPT; no traceback or nothing
    attributable -> UNKNOWN. Misclassification is capability-safe by
    construction -- origin never steers a gate.
    """

    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return FailureOrigin.INTERRUPT
    if isinstance(exc, StopSignalSwallowedError):
        return FailureOrigin.TORCHLENS
    tb = exc.__traceback__
    if tb is None:
        return FailureOrigin.UNKNOWN
    try:
        frames = traceback.extract_tb(tb)
    except Exception:
        return FailureOrigin.UNKNOWN
    for frame in reversed(frames):
        zone = _frame_zone(frame.filename)
        if zone == "user":
            return FailureOrigin.USER_OP
        if zone == "torchlens":
            line = frame.line or ""
            if _TRAMPOLINE_SOURCE_MARKER in line:
                return FailureOrigin.USER_OP
            return FailureOrigin.TORCHLENS
    return FailureOrigin.UNKNOWN


__all__ = [
    "CAPTURE_OUTCOME_CAPABILITIES",
    "CaptureOutcome",
    "CaptureOutcomeError",
    "CapturePhase",
    "CaptureStatus",
    "FailureOrigin",
    "StopRequest",
    "StopSignalSwallowedError",
    "attestation_coherent",
    "classify_failure_origin",
    "derive_outcome_from_structural_state",
    "outcome_for",
    "parse_outcome_payload",
    "require_capture_capability",
    "resolve_loaded_outcome",
]
