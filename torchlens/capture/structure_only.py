"""Structure-only capture: typed refusals, capability authority, discharge.

L7a wave-0 module (D8-default branch). One in-code capability authority
(:data:`STRUCTURE_ONLY_CAPABILITIES`) with ONE chokepoint
(:func:`require_structure_only_capability`), mirrored by the human table in
``docs/reference/structure_only_capabilities.md``; the teaching-refusal error
types raised by the torch mode belt and forward-boundary backstop; the
hypothesis claim vocabulary (:class:`StructureClaimStatus`); and the real-run
discharge machinery (:func:`discharge_against`, surfaced as
``Trace.discharge_against``).

NAMING: every public spelling in this module is DOCUMENTED-UNSTABLE pending
the rolling naming session, and every refusal code is additionally S2-gated
(the slate ratifies SPELLING, S2 ratifies EXISTENCE and regime). No
deprecation shim is owed on rename. S2 SEAM (labeled): registration of these
error classes in ``torchlens.errors`` and the settlement-side marker stamp in
``torchlens/capture/outcome.py`` ride the S2 author's ratification PR — this
module never touches the S2 fence files.
"""

from __future__ import annotations

import enum
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from ..errors._base import CaptureError

if TYPE_CHECKING:
    pass

__all__ = [
    "STRUCTURE_ONLY_CAPABILITIES",
    "CapabilityRow",
    "MetaKernelUnavailableError",
    "StructureClaimStatus",
    "StructureDischarge",
    "StructureOnlyCapabilityError",
    "ValueDependentBranchError",
    "claim_status_for",
    "discharge_against",
    "registered_discharge",
    "require_structure_only_capability",
]


# ---------------------------------------------------------------------------
# Teaching-refusal error types (memo sec 2.3/2.4)
# ---------------------------------------------------------------------------


class ValueDependentBranchError(CaptureError, RuntimeError):
    """A value escape reached the Python host from USER code under
    ``structure_only=True`` (memo sec 2.2 Layer 1 / Layer 2).

    Device-neutral by contract: under structure-only the recorded graph must
    never be VALUE-SELECTED, so enumerated escapes from user frames refuse on
    meta AND real tensors alike. ``fields["code"]`` is
    ``value_dependent_branch_unsupported``; ``fields["consumer_kind"]`` uses
    the closed ``ast_branches`` vocabulary plus ``scalar_escape`` and
    ``unclassified_escape``; ``fields["offenses"]`` carries
    ``{file, line, consumer_kind, tensor_label, escape_method, substrate}``
    records.
    """


class MetaKernelUnavailableError(CaptureError, RuntimeError):
    """An op with no meta kernel died inside torch dispatch under
    ``structure_only=True`` (memo sec 2.4).

    Classification is by raising-frame provenance and exception family, never
    message text; the original ``NotImplementedError`` is chained via
    ``raise ... from``. ``fields["code"]`` is ``meta_kernel_unavailable``.
    """


class StructureOnlyCapabilityError(CaptureError, RuntimeError):
    """A value-requiring consumer refused on a structure-only trace.

    Raised only by :func:`require_structure_only_capability`;
    ``fields["code"]`` carries the row's stable refusal code and
    ``fields["capability"]`` the row key.
    """


# ---------------------------------------------------------------------------
# Claim vocabulary (memo sec 3.2; S2-gated, documented-unstable)
# ---------------------------------------------------------------------------


class StructureClaimStatus(str, enum.Enum):
    """Tri-state evidence class of a structure-only trace's value-bearing
    claims. Never silently promoted (G4): ``CORROBORATED`` is written only by
    the discharge authority, and a registered refuted discharge flips the
    in-session state to ``REFUTED`` (G5)."""

    HYPOTHESIS = "hypothesis"
    CORROBORATED = "corroborated"
    REFUTED = "refuted"


# ---------------------------------------------------------------------------
# Chokepoint refusal codes raised only through the capability rows below
# (constant-spelled so the error-contract lockstep sees them; enrolled in
# tests/test_error_contract_lockstep.py::_CONSTANT_SPELLED_CODES). All
# S2-gated, DOCUMENTED-UNSTABLE.
# ---------------------------------------------------------------------------

STRUCTURE_ONLY_SAVE_UNSUPPORTED = "structure_only_save_unsupported"
STRUCTURE_ONLY_RUNNABLE_UNSUPPORTED = "structure_only_runnable_unsupported"
STRUCTURE_ONLY_REPLAY_UNSUPPORTED = "structure_only_replay_unsupported"
STRUCTURE_ONLY_VALIDATION_UNSUPPORTED = "structure_only_validation_unsupported"
STRUCTURE_ONLY_BACKWARD_UNSUPPORTED = "structure_only_backward_unsupported"
STRUCTURE_ONLY_EPISODE_UNSUPPORTED = "structure_only_episode_unsupported"


# ---------------------------------------------------------------------------
# Capability table (memo sec 6) — the IN-CODE authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityRow:
    """One frozen row of the structure-only capability contract.

    ``status_v1`` uses the CLOSED five-member grammar
    ``supported_structural | supported_hypothesis | refuse:<code> |
    verify:<Vn> | out_of_scope:<contract-ref>``; ``refusal_code`` is populated
    exactly when the status is ``refuse:<code>``. The ``claim`` column's
    conditional wording is the contract: an amender may strike a condition it
    has FULFILLED, never widen the claim.
    """

    key: str
    claim: str
    status_v1: str
    flip_event: str
    evidence: str
    amend_owner: str
    refusal_code: str | None = None


_ROWS: Final[tuple[CapabilityRow, ...]] = (
    CapabilityRow(
        key="graph_structure",
        claim=(
            "The op graph, edges, order, and module nesting of THIS meta "
            "execution are recorded exactly."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py (E-1 structural pins)",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="param_geometry",
        claim=(
            "Parameter/buffer names, shapes, dtypes are recorded as declared; "
            "the persistence partition is recorded IF V8 verifies "
            "meta-compatibility."
        ),
        status_v1="supported_structural",
        flip_event="V8 verdict",
        evidence="pending: V8 verification test",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="shapes_dtypes",
        claim=(
            "Per-op shapes/dtypes are HYPOTHESES: valid under meta "
            "propagation, unproven until discharged by a real capture of the "
            "same graph."
        ),
        status_v1="supported_hypothesis",
        flip_event="discharge",
        evidence="tests/test_structure_only_discharge.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="flops_estimates",
        claim=(
            "FLOPs/MACs are derived from hypothesis shapes; they inherit "
            "hypothesis status and are labelled estimated, never measured."
        ),
        status_v1="supported_hypothesis",
        flip_event="discharge",
        evidence="tests/test_structure_only_honesty.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="memory_estimates",
        claim=(
            "Memory figures are geometry estimates; measured-memory columns "
            "render unknown, never zero."
        ),
        status_v1="supported_hypothesis",
        flip_event="discharge",
        evidence="tests/test_structure_only_honesty.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="taken_path_conditionals",
        claim=(
            "Conditional structure of the taken path is recorded; any "
            "VALUE-dependent branch through the enumerated escape surface "
            "refuses typed at the user's source line REGARDLESS of the "
            "tensor's device; unenumerated meta deaths refuse typed via the "
            "backstop; unenumerated REAL-value escapes in form (b) are "
            "undetectable and are priced by hypothesis status (coverage claim "
            "exactly per memo sec 2.1 C-ENUM/C-BACKSTOP/C-RESIDUAL)."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_teaching.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="meta_admission",
        claim=(
            "Meta-materialized models (form (a)) are admitted ONLY under D8; "
            "until D8 is granted the entry gate refuses unchanged."
        ),
        status_v1="refuse:unsupported_tensor_variant",
        flip_event="D8 granted",
        evidence="tests/test_structure_only_honesty.py (baseline gate pins)",
        amend_owner="S2-amendment",
        refusal_code="unsupported_tensor_variant",
    ),
    CapabilityRow(
        key="value_payloads",
        claim=(
            "Activations, argument values, output values are never recorded; requests refuse typed."
        ),
        status_v1="refuse:structure_only_values_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_values_unsupported",
    ),
    CapabilityRow(
        key="previews",
        claim="Value previews/thumbnails require values; refused.",
        status_v1="refuse:structure_only_values_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_values_unsupported",
    ),
    CapabilityRow(
        key="nonfinite_predicates",
        claim=(
            "raise_on_nan and nonfinite halt predicates have no values to "
            "test; the combination refuses typed at entry."
        ),
        status_v1="refuse:structure_only_option_conflict",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_option_conflict",
    ),
    CapabilityRow(
        key="runnable_ready_composition",
        claim=(
            "structure_only + runnable_ready is refused at entry: runnable "
            "eligibility and the structure substrate are incompatible in v1."
        ),
        status_v1="refuse:structure_only_option_conflict",
        flip_event="L7b amendment lands",
        evidence=(
            "tests/test_structure_only_entry.py; conflict lift rides the S2 "
            "StateSource amendment (request R-L7B-1) with the belt-coverage "
            "pin re-authored in the same change"
        ),
        amend_owner="L7b",
        refusal_code="structure_only_option_conflict",
    ),
    CapabilityRow(
        key="substrate_uniformity",
        claim=(
            "Form (a) requires meta inputs with meta state; mixed real/meta "
            "at entry refuses typed (memo sec 1.5 E-3/E-4); partially-meta "
            "state is not pre-validated and dies typed mid-forward via the "
            "backstop (E-5). Only reachable under D8; until then the entry "
            "gate refuses every meta cell unchanged."
        ),
        status_v1="refuse:unsupported_tensor_variant",
        flip_event="D8 granted",
        evidence="blocked on D8",
        amend_owner="S2-amendment",
        refusal_code="unsupported_tensor_variant",
    ),
    CapabilityRow(
        key="viz_graph_render",
        claim=(
            "Graph rendering (incl. size_by consuming hypothesis shapes) "
            "works, carrying the structure-only banner."
        ),
        status_v1="supported_hypothesis",
        flip_event="never",
        evidence="tests/test_structure_only_honesty.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="viz_payload_visualizers",
        claim=(
            "Payload-consuming visualizers (activation heatmaps, custom value "
            "visualizers) require values; refused typed."
        ),
        status_v1="refuse:structure_only_values_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_values_unsupported",
    ),
    CapabilityRow(
        key="structure_digests",
        claim=(
            "Graph-shape and meta-domain content digests are always computed; "
            "they can never collide with value-bearing digests."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_honesty.py (G2 domain pin)",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="discharge",
        claim=(
            "A real capture of the same graph upgrades hypothesis rows to "
            "corroborated or refutes them; upgrades happen ONLY via the "
            "discharge authority."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_discharge.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="refuted_rows",
        claim=("Consumers that tolerate hypothesis rows refuse REFUTED rows typed (G5)."),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_discharge.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="teaching_refusals",
        claim=(
            "Enumerated value escapes (device-neutral, both forms) and "
            "missing meta kernels refuse typed with the user source line; "
            "other meta-mechanism deaths are typed via the backstop without "
            "branch classification; unenumerated real-value escapes are "
            "outside the detectable surface (C-RESIDUAL); user exceptions "
            "propagate unchanged."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_teaching.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="save_analysis_artifact",
        claim=(
            "Persisting a structure-only trace is refused UNTIL the "
            "coordinated tlspec bump lands the marker + load rows; artifacts "
            "are then marked and load-validated. Registrar exit gates "
            "(test-only activation switch, fail-closed load marker) are the "
            "one sanctioned round-trip path before the bump."
        ),
        status_v1="refuse:structure_only_save_unsupported",
        flip_event="wave-3 bump",
        evidence="tests/test_structure_only_capabilities.py",
        amend_owner="P1",
        refusal_code=STRUCTURE_ONLY_SAVE_UNSUPPORTED,
    ),
    CapabilityRow(
        key="save_runnable",
        claim=(
            "Runnable save is refused in v1; IF the L7b late-bind posture "
            "lands (wave 1, post-L4, S1-serialized), declared late-bind slots "
            "replace this refusal for eligible models."
        ),
        status_v1="refuse:structure_only_runnable_unsupported",
        flip_event="L7b amendment lands",
        evidence=(
            "tests/test_structure_only_capabilities.py; entry-dark bridge + "
            "mandatory bind-digest authority shipped: "
            "tests/test_structure_only_bridge.py (flip blocked on the S2 "
            "StateSource amendment, request R-L7B-1)"
        ),
        amend_owner="L7b",
        refusal_code=STRUCTURE_ONLY_RUNNABLE_UNSUPPORTED,
    ),
    CapabilityRow(
        key="live_replay",
        claim=(
            "Replay requires values; refused unless and until late-bind (see "
            "save_runnable) provides them at bind time."
        ),
        status_v1="refuse:structure_only_replay_unsupported",
        flip_event="L7b amendment lands",
        evidence=(
            "tests/test_structure_only_capabilities.py; entry-dark bridge + "
            "S1 validator-reuse binding path shipped: "
            "tests/test_structure_only_bridge.py (flip blocked on the S2 "
            "StateSource amendment, request R-L7B-1)"
        ),
        amend_owner="L7b",
        refusal_code=STRUCTURE_ONLY_REPLAY_UNSUPPORTED,
    ),
    CapabilityRow(
        key="validation_entry",
        claim=(
            "There is nothing to validate against; refused permanently by "
            "design (discharge is the verification story)."
        ),
        status_v1="refuse:structure_only_validation_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_capabilities.py",
        amend_owner="L7a",
        refusal_code=STRUCTURE_ONLY_VALIDATION_UNSUPPORTED,
    ),
    CapabilityRow(
        key="backward_grads",
        claim=(
            "Backward/gradient capture is refused in v1; any future support "
            "is an L9-adjacent S2 amendment, not implied here."
        ),
        status_v1="refuse:structure_only_backward_unsupported",
        flip_event="S2 amendment",
        evidence="tests/test_structure_only_capabilities.py",
        amend_owner="S2-amendment",
        refusal_code=STRUCTURE_ONLY_BACKWARD_UNSUPPORTED,
    ),
    CapabilityRow(
        key="episode_composition",
        claim=(
            "Episode capture composes with structure-only ONLY if a later S2 "
            "amendment rules it; refused in v1."
        ),
        status_v1="refuse:structure_only_episode_unsupported",
        flip_event="S2 amendment",
        evidence="reserved: no episode surface exists on this branch yet",
        amend_owner="S2-amendment",
        refusal_code=STRUCTURE_ONLY_EPISODE_UNSUPPORTED,
    ),
    CapabilityRow(
        key="distributed",
        claim=(
            "Distributed structure-only capture is out of scope this sprint; "
            "the existing distributed refusal contract governs."
        ),
        status_v1="out_of_scope:distributed-contract",
        flip_event="S2 amendment",
        evidence="torchlens/_distributed.py refusal surface",
        amend_owner="S2-amendment",
    ),
    CapabilityRow(
        key="fake_tensor_substrate",
        claim=(
            "Capturing a REAL model structure-only via FakeTensorMode is a "
            "VERIFY item, not a capability."
        ),
        status_v1="verify:V1",
        flip_event="V1 verdict",
        evidence="pending: V1 spike",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="symbolic_shapes",
        claim=(
            "Symbolic/dynamic shapes remain refused at the variant gate; a "
            "ShapeEnv-backed range hypothesis is a VERIFY item."
        ),
        status_v1="verify:V2",
        flip_event="V2 verdict",
        evidence="pending: V2 verification",
        amend_owner="L7a",
    ),
)

STRUCTURE_ONLY_CAPABILITIES: Final[Mapping[str, CapabilityRow]] = {row.key: row for row in _ROWS}
"""Frozen structure-only capability rows; consumers branch ONLY through
:func:`require_structure_only_capability` (lockstep-tested both directions)."""


_VALID_AMEND_OWNERS: Final[frozenset[str]] = frozenset({"L7a", "L7b", "S2-amendment", "P1"})

_L7B_ROW_TEACHING: Final[Mapping[str, str]] = {
    "save_runnable": (
        "This is the L7b v1 floor: runnable save flips to the declared "
        "late-bind posture (state slots declared at capture time, values "
        "bound at run time with mandatory byte digests) when its S2 "
        "StateSource amendment lands — this row's named flip event. Until "
        "then, capture the real model with intervention_ready=True to "
        "produce a runnable artifact."
    ),
    "live_replay": (
        "A structure-only capture records no values to replay. Until the "
        "L7b declared late-bind posture lands (this row's named flip "
        "event), run the real model directly, or corroborate this trace "
        "against a real capture via trace.discharge_against(real_trace)."
    ),
}
"""Row-scoped teaching sentences for the L7b-owned rows (house rule: every
refusal names the boundary and what to do instead). Amends refusal TEACHING
only — claims, statuses, and codes are untouched; the entries are keyed to
rows whose ``amend_owner`` is L7b."""


def _validate_row_grammar(row: CapabilityRow) -> None:
    """Enforce the closed 6.2 grammar at import; a bad row is a bug."""

    status = row.status_v1
    valid = (
        status in ("supported_structural", "supported_hypothesis")
        or (status.startswith("refuse:") and len(status) > len("refuse:"))
        or (status.startswith("verify:V") and status[len("verify:V") :].isdigit())
        or (status.startswith("out_of_scope:") and len(status) > len("out_of_scope:"))
    )
    if not valid:
        raise AssertionError(f"capability row {row.key!r} violates the closed grammar: {status!r}")
    if status.startswith("refuse:"):
        if row.refusal_code != status.split(":", 1)[1]:
            raise AssertionError(
                f"capability row {row.key!r}: refusal_code must equal the refuse:<code> cell"
            )
    elif row.refusal_code is not None:
        raise AssertionError(f"capability row {row.key!r}: refusal_code without refuse status")
    if row.amend_owner not in _VALID_AMEND_OWNERS:
        raise AssertionError(f"capability row {row.key!r}: unknown amend_owner {row.amend_owner!r}")
    if not row.claim.strip() or not row.flip_event.strip() or not row.evidence.strip():
        raise AssertionError(f"capability row {row.key!r} has an empty contract column")


for _row in _ROWS:
    _validate_row_grammar(_row)
del _row


# ---------------------------------------------------------------------------
# Discharge registry (weak-keyed side table; the trace object never mutates)
# ---------------------------------------------------------------------------

_DISCHARGE_REGISTRY: weakref.WeakKeyDictionary[Any, StructureDischarge] = (
    weakref.WeakKeyDictionary()
)
"""Per-structure-trace registered discharge (the ledger pattern of
``completeness_witness._HOST_ESCAPE`` tables). G4: only
:func:`discharge_against` writes here; G5: a refuted entry flips the
in-session claim state consulted by the chokepoint."""


def registered_discharge(trace: Any) -> StructureDischarge | None:
    """Return the registered discharge for ``trace``, if any."""

    return _DISCHARGE_REGISTRY.get(trace)


def claim_status_for(trace: Any) -> StructureClaimStatus:
    """Return the in-session claim status of a structure-only trace.

    Born ``HYPOTHESIS``; flipped only by a registered discharge (G4/G5).
    """

    discharge = _DISCHARGE_REGISTRY.get(trace)
    if discharge is None:
        return StructureClaimStatus.HYPOTHESIS
    if discharge.verdict is StructureClaimStatus.REFUTED:
        return StructureClaimStatus.REFUTED
    return StructureClaimStatus.CORROBORATED


# ---------------------------------------------------------------------------
# THE chokepoint (memo sec 6.1; G1)
# ---------------------------------------------------------------------------


def require_structure_only_capability(
    trace: Any,
    capability: str,
    *,
    detail: str | None = None,
) -> CapabilityRow | None:
    """Enforce one structure-only capability row for ``trace``.

    No-op (returns ``None``) when ``trace`` is not a structure-only capture:
    the default path pays one attribute read. For structure-only traces a
    ``refuse:<code>`` row raises :class:`StructureOnlyCapabilityError` with
    the row's stable code, a ``supported_hypothesis`` row additionally
    refuses when a registered REFUTED discharge has flipped the trace's claim
    state (G5), and supported/verify rows return the row.

    The ONE sanctioned bypass: while the S3 pre-release registrar's test-only
    activation switch is on, the ``save_analysis_artifact`` row's refusal is
    lifted so portability exit gates can round-trip the DROP-declared marker
    — every switch-on write stamps the fail-closed pre-release marker, so a
    switched artifact can never circulate as a real one (registrar contract,
    torchlens/_io/prerelease.py).
    """

    if not bool(getattr(trace, "structure_only", False)):
        return None
    row = STRUCTURE_ONLY_CAPABILITIES[capability]
    if row.status_v1.startswith("refuse:"):
        if capability == "save_analysis_artifact":
            from .._io.prerelease import prerelease_fields_active

            if prerelease_fields_active():
                return row
        code = row.refusal_code or row.status_v1.split(":", 1)[1]
        message = (
            f"TorchLens refuses {capability!r} for a structure-only capture "
            f"(code {code}). {row.claim}"
        )
        if detail:
            message += f" {detail}"
        teaching = _L7B_ROW_TEACHING.get(capability)
        if teaching is not None:
            message += f" {teaching}"
        message += (
            " Remedy: run a real capture (tl.trace without structure_only) "
            "for value-bearing surfaces, or see "
            "docs/reference/structure_only_capabilities.md."
        )
        raise StructureOnlyCapabilityError(
            message,
            code=code,
            capability=capability,
            status=row.status_v1,
            flip_event=row.flip_event,
        )
    if (
        row.status_v1 == "supported_hypothesis"
        and claim_status_for(trace) is StructureClaimStatus.REFUTED
    ):
        discharge = _DISCHARGE_REGISTRY.get(trace)
        first = discharge.first_contradiction if discharge is not None else None
        raise StructureOnlyCapabilityError(
            f"TorchLens refuses {capability!r}: this structure-only "
            "capture's hypotheses were REFUTED by a registered real-run "
            f"discharge (first contradiction: {first}). A refuted "
            "hypothesis is strictly worse than no capture. Remedy: "
            "re-capture after fixing the model/meta divergence, or "
            "consume the discharge record's contradiction table directly.",
            code="structure_only_refuted_hypothesis",
            capability=capability,
            status=row.status_v1,
        )
    return row


# ---------------------------------------------------------------------------
# Discharge machinery (memo sec 3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimComparison:
    """One per-claim discharge row."""

    claim_kind: str
    site: str
    hypothesis_value: Any
    observed_value: Any
    verdict: StructureClaimStatus


@dataclass(frozen=True)
class StructureDischarge:
    """Frozen result of discharging a structure-only trace against a real
    capture. ``verdict`` is CORROBORATED iff EVERY compared claim matched
    (first contradiction wins the overall floor); ``claims`` is the per-claim
    table; the two digests witness which graphs were joined."""

    verdict: StructureClaimStatus
    claims: tuple[ClaimComparison, ...]
    structure_digest: str
    real_digest: str
    graph_matched: bool
    first_contradiction: str | None


def _require_discharge_preconditions(structure_trace: Any, real_trace: Any) -> None:
    """Typed precondition refusals (never silent Nones)."""

    if not bool(getattr(structure_trace, "structure_only", False)):
        raise StructureOnlyCapabilityError(
            "discharge_against is only defined on a structure-only capture; "
            "this trace is an ordinary capture. Remedy: call it on the "
            "structure-only trace, passing the real capture as the argument.",
            code="structure_only_discharge_precondition",
            capability="discharge",
        )
    if bool(getattr(real_trace, "structure_only", False)):
        raise StructureOnlyCapabilityError(
            "discharge_against requires an ORDINARY (non-structure-only) real "
            "capture as the oracle; received another structure-only trace. "
            "Remedy: capture the model without structure_only and discharge "
            "against that trace.",
            code="structure_only_discharge_precondition",
            capability="discharge",
        )
    real_outcome = getattr(real_trace, "outcome", None)
    status_value = getattr(getattr(real_outcome, "status", None), "value", None)
    if status_value != "complete":
        raise StructureOnlyCapabilityError(
            "discharge_against requires a settled COMPLETE real capture as "
            f"the oracle; received outcome status {status_value!r}. Remedy: "
            "re-run the real capture to completion.",
            code="structure_only_discharge_precondition",
            capability="discharge",
        )


def _layer_claims(layer: Any) -> tuple[tuple[str, Any], ...]:
    """Extract the compared claim kinds from one layer record."""

    shape = getattr(layer, "shape", None)
    param_shapes = getattr(layer, "param_shapes", None) or ()
    return (
        ("shape", tuple(shape) if shape is not None else None),
        ("dtype", str(getattr(layer, "dtype", None))),
        # Layer.param_shapes is an ordered sequence of parameter shapes
        # (weight, bias, ...); order is part of the geometry claim.
        ("param_geometry", tuple(tuple(entry) for entry in param_shapes)),
    )


def discharge_against(structure_trace: Any, real_trace: Any) -> StructureDischarge:
    """Discharge a structure-only trace's hypotheses against a real capture.

    The join is POSITIONAL over ``layer_list``, LICENSED BY DIGEST EQUALITY
    of the address-free graph-shape hash (``tl.hash.trace``): the digest
    covers each record's index and parent_indices, so equal digests guarantee
    identically-ordered record sequences and record *i* corresponds to record
    *i* — repeated ops, recurrent passes, multi-output layers, and symmetric
    subgraphs are covered by construction. Never joins by label string.

    Structurally different graphs settle ``REFUTED`` at the graph level
    without any per-claim comparison. The result registers in the weak-keyed
    side table (G5) and NEVER mutates either trace.
    """

    from .. import hash as tl_hash

    _require_discharge_preconditions(structure_trace, real_trace)
    structure_digest = tl_hash.trace(structure_trace)
    real_digest = tl_hash.trace(real_trace)
    if structure_digest != real_digest:
        discharge = StructureDischarge(
            verdict=StructureClaimStatus.REFUTED,
            claims=(),
            structure_digest=structure_digest,
            real_digest=real_digest,
            graph_matched=False,
            first_contradiction=(
                "graph structure: address-free graph-shape digests differ; "
                "no per-claim rows were compared"
            ),
        )
        _DISCHARGE_REGISTRY[structure_trace] = discharge
        return discharge

    claims: list[ClaimComparison] = []
    first_contradiction: str | None = None
    for index, (hyp_layer, real_layer) in enumerate(
        zip(structure_trace.layer_list, real_trace.layer_list, strict=False)
    ):
        site = (
            getattr(hyp_layer, "label", None)
            or getattr(hyp_layer, "layer_label", None)
            or f"layer[{index}]"
        )
        for claim_kind, hyp_value in _layer_claims(hyp_layer):
            observed = dict(_layer_claims(real_layer))[claim_kind]
            matched = hyp_value == observed
            claims.append(
                ClaimComparison(
                    claim_kind=claim_kind,
                    site=str(site),
                    hypothesis_value=hyp_value,
                    observed_value=observed,
                    verdict=(
                        StructureClaimStatus.CORROBORATED
                        if matched
                        else StructureClaimStatus.REFUTED
                    ),
                )
            )
            if not matched and first_contradiction is None:
                first_contradiction = (
                    f"{site}: {claim_kind} hypothesis {hyp_value!r} vs observed {observed!r}"
                )
    discharge = StructureDischarge(
        verdict=(
            StructureClaimStatus.REFUTED
            if first_contradiction is not None
            else StructureClaimStatus.CORROBORATED
        ),
        claims=tuple(claims),
        structure_digest=structure_digest,
        real_digest=real_digest,
        graph_matched=True,
        first_contradiction=first_contradiction,
    )
    _DISCHARGE_REGISTRY[structure_trace] = discharge
    return discharge
