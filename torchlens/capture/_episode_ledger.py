"""Episode ledger: per-step status disclosure for ``capture_kind=episode``.

An EPISODE is one wrapped session capture product (the L2 SESSION ruling,
ratified at S2 2026-08-16): a wrapper ``nn.Module`` whose ``forward`` steps a
declared STEPPED MODULE N times is captured as ONE product carrying
``capture_kind=episode`` and this ledger ON the product, attached at
``trace.annotations["episode"]``. The ledger is a DISCLOSURE, never a
settlement authority: the product's one settled ``CaptureOutcome`` is written
by the existing authority (``torchlens.capture.outcome``) with its vocabulary
unchanged, and no derivation here may bless COMPLETE or revise a settled
record (the R06 discipline).

Pattern-of-record: :mod:`torchlens.distributed._ledger` (S7 instruction) —
closed ``Literal`` vocabularies, frozen payload key sets, fail-closed
``from_payload``, write-once container, derived views that refuse
out-of-vocabulary tokens. Parse boundaries raise ``ValueError``; callers
convert into the typed episode refusal family (stable codes, see
:mod:`torchlens.errors`).

Schema of record: the S6/S7 drafts in the L2 spike memo, BINDING as of the
S2 ratification. Every spelling here is DOCUMENTED-UNSTABLE pending the
rolling naming session (spike section 6.5); semantics are pinned.

PERSISTENCE IS GATED (S3 version discipline): ``annotations["episode"]`` is a
NEW persistence key and never rides a real tlspec-v7 artifact. The writer
ships inert behind the pre-release registrar switch
(:mod:`torchlens._io.prerelease`); the coordinated wave-3 bump activates it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ..options import EpisodeSpec

__all__ = [
    "CAPTURE_KIND_EPISODE",
    "EPISODE_ANNOTATIONS_KEY",
    "EpisodeFoldResult",
    "EpisodeLedger",
    "EpisodeLedgerHeader",
    "EpisodeLedgerRow",
    "ResolvedEpisode",
    "attach_episode_header",
    "attach_failed_episode_ledger",
    "derive_episode_status",
    "episode_ledger_for",
    "producer_digest",
    "resolve_episode_declaration",
    "row_status_for_member_outcome",
    "row_status_for_recording_status",
    "validate_loaded_episode_annotations",
    "write_episode_ledger",
]

#: Annotations key the ledger (header + rows) lives under on the Trace.
EPISODE_ANNOTATIONS_KEY = "episode"

#: The capture-kind marker value stamped in the ledger header.
CAPTURE_KIND_EPISODE = "episode"

Role = Literal["prefill", "decode"]
"""Row role: row 0 is the prefill (the stepped module's call 1); later rows decode."""

RowStatus = Literal["complete", "interrupted", "absent"]
"""Closed row-status vocabulary (S7).

``complete`` is row-scoped FORWARD-STEP truth (did step k's forward return),
never a claim about the PRODUCT — product truth is the settled
``CaptureOutcome``. ``interrupted`` means the settled outcome's frontier lies
INSIDE step k. ``absent`` means the step never started.
"""

EscalationReason = Literal["step_failed", "divergence", "requested"]
"""Closed vocabulary for the escalation disclosure (E-A2)."""

FidelityBasis = Literal["tokens", "diverged", "none", "forced"]
"""Token-fidelity disclosure basis (E-A3; ``forced`` = teacher-forced feed,
an explicitly NON-VERIFYING disclosed mode — never a settlement input)."""

TokenFeed = Literal["free", "forced"]
"""Whether the episode's step inputs were model-emitted (``free``) or
teacher-forced from a declared token sequence (``forced``)."""

ProvenanceTier = Literal["exact", "ledger_only"]
"""Per-step provenance tier (S2 combination table). A session product is
uniformly ``exact``; mixed tiers arise only in the floor, where the episode
tier is the MINIMUM of its members' tiers, never the maximum."""

#: Derived episode-status terms for the floor fold (section 2.3). PROVISIONAL
#: spellings; these are derivations, never settled CaptureOutcome values.
EpisodeFoldStatus = Literal[
    "episode_complete",
    "episode_halted_at_step",
    "episode_aborted_at_step",
    "episode_failed_at_step",
    "episode_unknown",
]

_ROLES = frozenset({"prefill", "decode"})
_ROW_STATUSES = frozenset({"complete", "interrupted", "absent"})
_ESCALATION_REASONS = frozenset({"step_failed", "divergence", "requested"})
_FIDELITY_BASES = frozenset({"tokens", "diverged", "none", "forced"})
_TOKEN_FEEDS = frozenset({"free", "forced"})
_PROVENANCE_TIERS = frozenset({"exact", "ledger_only"})

# ``to_payload`` emits exactly these keys; an unknown key in a loaded payload
# is a forged or drifted artifact, never something to silently ignore.
_HEADER_PAYLOAD_KEYS = frozenset(
    {
        "episode_id",
        "capture_kind",
        "stepped_module",
        "n_steps_declared",
        "entry_seed",
        "token_feed",
        "provenance_tier",
        "structure_only",
        "escalated_from",
        "reason",
        "fidelity_basis",
    }
)
_ROW_PAYLOAD_KEYS = frozenset(
    {
        "episode_step",
        "role",
        "coord",
        "tokens",
        "status",
        "frontier",
        "cache_len",
        "rng_digest",
        "escalation",
    }
)
_FRONTIER_KEYS = frozenset({"boundary_kind", "boundary_label"})
_COORD_KEYS = frozenset({"member_call_index", "pass_range", "member"})


def _require_vocabulary(payload: Mapping[str, Any], key: str, vocabulary: frozenset[str]) -> str:
    """Return a required closed-vocabulary field, refusing anything else."""

    value = payload.get(key)
    if value not in vocabulary:
        raise ValueError(
            f"episode-ledger field {key!r} is {value!r}, outside the closed "
            f"vocabulary {sorted(vocabulary)}"
        )
    return str(value)


def _require_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    """Return a required non-negative integer field, refusing anything else."""

    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"episode-ledger field {key!r} must be a non-negative int")
    return value


def _optional_non_negative_int(payload: Mapping[str, Any], key: str) -> int | None:
    """Return an optional non-negative integer field, refusing other types."""

    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"episode-ledger field {key!r} must be a non-negative int or null")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    """Return an optional string field, refusing a non-string value."""

    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"episode-ledger field {key!r} must be a string or null")
    return value


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    """Return a required non-empty string field, refusing anything else."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"episode-ledger field {key!r} must be a non-empty string")
    return value


@dataclass(frozen=True)
class EpisodeLedgerHeader:
    """Episode-level ledger header (S7).

    Parameters
    ----------
    episode_id:
        Opaque identifier naming this episode entity; the S6 ``episode_member``
        relation rows reference it.
    stepped_module:
        Module identity (address within the episode root) whose successive
        top-level calls define step boundaries.
    entry_seed:
        The capture's effective ``random_seed`` (the managed RNG recipe: the
        drawn-or-passed seed is recorded, never inferred).
    n_steps_declared:
        Declared step count, when the declaration carries one.
    token_feed:
        ``free`` (model-emitted step inputs) or ``forced`` (teacher-forced
        feed; explicitly non-verifying, disclosed).
    provenance_tier:
        Uniform per-product tier for the session home (``exact`` /
        ``ledger_only``).
    structure_only:
        Mirror of the capture's structure-only marker. Episodes with
        ``structure_only=True`` TYPED-REFUSE at entry this sprint (S2 table);
        the field exists so a loaded ledger can validate the token presence
        rule in both directions.
    escalated_from:
        Producer digest of the cheap-tier product this capture escalates
        (:func:`producer_digest`); present iff this product is an escalation.
    reason:
        Escalation reason (closed vocabulary); present iff escalation.
    fidelity_basis:
        Token-fidelity disclosure (E-A3); present iff escalation or forced.
    """

    episode_id: str
    stepped_module: str
    entry_seed: int
    n_steps_declared: int | None = None
    token_feed: TokenFeed = "free"
    provenance_tier: ProvenanceTier = "exact"
    structure_only: bool = False
    escalated_from: str | None = None
    reason: EscalationReason | None = None
    fidelity_basis: FidelityBasis | None = None

    def __post_init__(self) -> None:
        if self.token_feed not in _TOKEN_FEEDS:
            raise ValueError(f"episode-ledger token_feed {self.token_feed!r} is out of vocabulary")
        if self.provenance_tier not in _PROVENANCE_TIERS:
            raise ValueError(
                f"episode-ledger provenance_tier {self.provenance_tier!r} is out of vocabulary"
            )
        if self.reason is not None and self.reason not in _ESCALATION_REASONS:
            raise ValueError(f"episode-ledger reason {self.reason!r} is out of vocabulary")
        if self.fidelity_basis is not None and self.fidelity_basis not in _FIDELITY_BASES:
            raise ValueError(
                f"episode-ledger fidelity_basis {self.fidelity_basis!r} is out of vocabulary"
            )
        # E-A2: the escalation disclosure is present iff escalation. reason and
        # escalated_from travel together, always.
        if (self.escalated_from is None) != (self.reason is None):
            raise ValueError(
                "episode-ledger escalation disclosure is all-or-nothing: "
                "escalated_from and reason must be present together"
            )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for portable artifacts."""

        return {
            "episode_id": self.episode_id,
            "capture_kind": CAPTURE_KIND_EPISODE,
            "stepped_module": self.stepped_module,
            "n_steps_declared": self.n_steps_declared,
            "entry_seed": self.entry_seed,
            "token_feed": self.token_feed,
            "provenance_tier": self.provenance_tier,
            "structure_only": self.structure_only,
            "escalated_from": self.escalated_from,
            "reason": self.reason,
            "fidelity_basis": self.fidelity_basis,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeLedgerHeader:
        """Rebuild a header from :meth:`to_payload` output, FAIL-CLOSED.

        Raises
        ------
        ValueError
            On an unknown key, a missing/ill-typed field, a value outside a
            closed vocabulary, or a ``capture_kind`` that is not ``episode``
            (callers convert into ``episode_ledger_without_declaration``).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("episode-ledger header payload must be a mapping")
        unknown = set(payload) - _HEADER_PAYLOAD_KEYS
        if unknown:
            raise ValueError(f"episode-ledger header has unknown keys {sorted(unknown)}")
        missing = _HEADER_PAYLOAD_KEYS - set(payload)
        if missing:
            raise ValueError(f"episode-ledger header is missing keys {sorted(missing)}")
        if payload["capture_kind"] != CAPTURE_KIND_EPISODE:
            raise ValueError(
                f"episode-ledger capture_kind is {payload['capture_kind']!r}, "
                f"not {CAPTURE_KIND_EPISODE!r}"
            )
        structure_only = payload["structure_only"]
        if not isinstance(structure_only, bool):
            raise ValueError("episode-ledger field 'structure_only' must be a bool")
        entry_seed = payload["entry_seed"]
        if not isinstance(entry_seed, int) or isinstance(entry_seed, bool):
            raise ValueError("episode-ledger field 'entry_seed' must be an int")
        reason = _optional_str(payload, "reason")
        if reason is not None and reason not in _ESCALATION_REASONS:
            raise ValueError(f"episode-ledger reason {reason!r} is out of vocabulary")
        fidelity_basis = _optional_str(payload, "fidelity_basis")
        if fidelity_basis is not None and fidelity_basis not in _FIDELITY_BASES:
            raise ValueError(
                f"episode-ledger fidelity_basis {fidelity_basis!r} is out of vocabulary"
            )
        return cls(
            episode_id=_require_str(payload, "episode_id"),
            stepped_module=_require_str(payload, "stepped_module"),
            entry_seed=entry_seed,
            n_steps_declared=_optional_non_negative_int(payload, "n_steps_declared"),
            token_feed=cast("TokenFeed", _require_vocabulary(payload, "token_feed", _TOKEN_FEEDS)),
            provenance_tier=cast(
                "ProvenanceTier",
                _require_vocabulary(payload, "provenance_tier", _PROVENANCE_TIERS),
            ),
            structure_only=structure_only,
            escalated_from=_optional_str(payload, "escalated_from"),
            reason=cast("EscalationReason | None", reason),
            fidelity_basis=cast("FidelityBasis | None", fidelity_basis),
        )


@dataclass(frozen=True)
class EpisodeLedgerRow:
    """One per-step ledger row (S7).

    Parameters
    ----------
    episode_step:
        0-based step index; row 0 is the prefill.
    role:
        ``prefill`` for row 0, ``decode`` afterwards.
    status:
        Row status (closed vocabulary; see :data:`RowStatus`).
    coord:
        Addressing coordinates. Session home: ``{"member_call_index": int,
        "pass_range": [lo, hi] | None}``. Floor home: ``{"member": <bundle
        member name>}`` — the ONLY ``member`` spelling in the schema, always
        the Bundle-key sense.
    cache_len:
        Token-prefix length step k's forward reads (prompt length + k). A
        DERIVED arithmetic disclosure over the declared token axis, not a
        cache introspection.
    tokens:
        Emitted token ids for this step. REQUIRED on value-mode episodes;
        ABSENT (whole field, typed absence) on structure-only episodes.
        Load-validated in both directions.
    frontier:
        ``{"boundary_kind": str, "boundary_label": str}`` — interrupted rows
        only, from the settled record's disclosure. Never promotes a ragged
        frontier to a step boundary.
    rng_digest:
        Optional declared-scope RNG snapshot digest (floor escalation E-B3).
    escalation:
        Optional reference to an S6 ``escalates`` relation row (floor home).
    """

    episode_step: int
    role: Role
    status: RowStatus
    coord: Mapping[str, Any]
    cache_len: int
    tokens: tuple[int, ...] | None = None
    frontier: Mapping[str, str] | None = None
    rng_digest: str | None = None
    escalation: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"episode-ledger row role {self.role!r} is out of vocabulary")
        if self.status not in _ROW_STATUSES:
            raise ValueError(f"episode-ledger row status {self.status!r} is out of vocabulary")
        if self.episode_step < 0:
            raise ValueError("episode-ledger row episode_step must be non-negative")
        if (self.role == "prefill") != (self.episode_step == 0):
            raise ValueError(
                "episode-ledger row 0 is the prefill and only row 0 may carry role='prefill'"
            )
        if self.frontier is not None and self.status != "interrupted":
            raise ValueError("episode-ledger frontier is disclosed on interrupted rows only")
        unknown_coord = set(self.coord) - _COORD_KEYS
        if unknown_coord:
            raise ValueError(f"episode-ledger row coord has unknown keys {sorted(unknown_coord)}")

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for portable artifacts."""

        return {
            "episode_step": self.episode_step,
            "role": self.role,
            "status": self.status,
            "coord": dict(self.coord),
            "cache_len": self.cache_len,
            "tokens": list(self.tokens) if self.tokens is not None else None,
            "frontier": dict(self.frontier) if self.frontier is not None else None,
            "rng_digest": self.rng_digest,
            "escalation": self.escalation,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeLedgerRow:
        """Rebuild a row from :meth:`to_payload` output, FAIL-CLOSED.

        Raises
        ------
        ValueError
            On an unknown key, a missing/ill-typed field, or a value outside
            a closed vocabulary (callers convert into
            ``episode_ledger_incoherent``).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("episode-ledger row payload must be a mapping")
        unknown = set(payload) - _ROW_PAYLOAD_KEYS
        if unknown:
            raise ValueError(f"episode-ledger row has unknown keys {sorted(unknown)}")
        missing = _ROW_PAYLOAD_KEYS - set(payload)
        if missing:
            raise ValueError(f"episode-ledger row is missing keys {sorted(missing)}")
        coord = payload["coord"]
        if not isinstance(coord, Mapping):
            raise ValueError("episode-ledger row field 'coord' must be a mapping")
        tokens_value = payload["tokens"]
        tokens: tuple[int, ...] | None
        if tokens_value is None:
            tokens = None
        elif isinstance(tokens_value, Sequence) and not isinstance(tokens_value, (str, bytes)):
            if not all(
                isinstance(token, int) and not isinstance(token, bool) for token in tokens_value
            ):
                raise ValueError("episode-ledger row field 'tokens' must hold ints")
            tokens = tuple(int(token) for token in tokens_value)
        else:
            raise ValueError("episode-ledger row field 'tokens' must be a list of ints or null")
        frontier_value = payload["frontier"]
        frontier: dict[str, str] | None
        if frontier_value is None:
            frontier = None
        elif isinstance(frontier_value, Mapping):
            if set(frontier_value) != _FRONTIER_KEYS:
                raise ValueError(
                    f"episode-ledger row frontier must carry exactly {sorted(_FRONTIER_KEYS)}"
                )
            if not all(isinstance(value, str) for value in frontier_value.values()):
                raise ValueError("episode-ledger row frontier values must be strings")
            frontier = {key: str(value) for key, value in frontier_value.items()}
        else:
            raise ValueError("episode-ledger row field 'frontier' must be a mapping or null")
        return cls(
            episode_step=_require_non_negative_int(payload, "episode_step"),
            role=cast("Role", _require_vocabulary(payload, "role", _ROLES)),
            status=cast("RowStatus", _require_vocabulary(payload, "status", _ROW_STATUSES)),
            coord=dict(coord),
            cache_len=_require_non_negative_int(payload, "cache_len"),
            tokens=tokens,
            frontier=frontier,
            rng_digest=_optional_str(payload, "rng_digest"),
            escalation=_optional_str(payload, "escalation"),
        )


class EpisodeLedger:
    """Write-once episode ledger: one header plus ordered per-step rows.

    Rows are written ONCE at settlement/finalize (S7 L3); the finalized view
    is identity-stable and immutable. No post-hoc row edits: escalation
    annotates via S6 relation rows, never rewrites history, and demotion
    replaces the outcome record, never these rows.
    """

    def __init__(self, header: EpisodeLedgerHeader, rows: Sequence[EpisodeLedgerRow]) -> None:
        self._header = header
        self._rows: tuple[EpisodeLedgerRow, ...] = tuple(rows)
        _check_monotone_prefix_law(self._rows)
        _check_token_presence_rule(header, self._rows)

    @property
    def header(self) -> EpisodeLedgerHeader:
        """The episode-level header."""

        return self._header

    @property
    def rows(self) -> tuple[EpisodeLedgerRow, ...]:
        """Immutable, identity-stable view of the per-step rows in order."""

        return self._rows

    @property
    def steps_completed(self) -> int:
        """Derived disclosure: count of ``complete`` rows."""

        return sum(1 for row in self._rows if row.status == "complete")

    @property
    def truncated_at_step(self) -> int | None:
        """Derived disclosure: index of the interrupted or first absent row.

        ``None`` when every row is complete (nothing was truncated).
        """

        for row in self._rows:
            if row.status != "complete":
                return row.episode_step
        return None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for portable artifacts."""

        return {
            "header": self._header.to_payload(),
            "rows": [row.to_payload() for row in self._rows],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeLedger:
        """Rebuild a ledger from :meth:`to_payload` output, FAIL-CLOSED.

        The rebuild routes rows through the same monotone-prefix and
        token-presence checks the live writer enforces, so a forged or
        drifted payload can never load as a coherent ledger.

        Raises
        ------
        ValueError
            On any invalid header/row payload or a prefix-law violation
            (callers convert into ``episode_ledger_incoherent``).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("episode-ledger payload must be a mapping")
        if set(payload) != {"header", "rows"}:
            raise ValueError(
                "episode-ledger payload must carry exactly the keys {'header', 'rows'}"
            )
        rows_payload = payload["rows"]
        if not isinstance(rows_payload, list):
            raise ValueError("episode-ledger payload field 'rows' must be a list")
        header = EpisodeLedgerHeader.from_payload(payload["header"])
        rows = [EpisodeLedgerRow.from_payload(entry) for entry in rows_payload]
        return cls(header, rows)


def _check_monotone_prefix_law(rows: Sequence[EpisodeLedgerRow]) -> None:
    """Enforce the MONOTONE PREFIX LAW (S7 L1), refusing violations.

    Rows 0..k-1 complete, row k in {complete, interrupted}, rows k+1.. absent.
    At most one interrupted row; it is the last non-absent row. Steps must be
    the contiguous run 0..N-1.

    Raises
    ------
    ValueError
        On any geometry outside the law (callers convert into
        ``episode_ledger_incoherent`` and the outcome derivation degrades
        fail-closed).
    """

    seen_interrupted = False
    seen_absent = False
    for position, row in enumerate(rows):
        if row.episode_step != position:
            raise ValueError(
                "episode-ledger rows must be the contiguous 0-based step run; "
                f"row at position {position} carries episode_step {row.episode_step}"
            )
        if row.status == "complete":
            if seen_interrupted or seen_absent:
                raise ValueError(
                    "episode-ledger monotone prefix law violated: a complete row "
                    f"follows a non-complete row (step {row.episode_step})"
                )
        elif row.status == "interrupted":
            if seen_interrupted or seen_absent:
                raise ValueError(
                    "episode-ledger monotone prefix law violated: interrupted row "
                    f"at step {row.episode_step} is not the first non-complete row"
                )
            seen_interrupted = True
        else:  # absent
            seen_absent = True


def _check_token_presence_rule(
    header: EpisodeLedgerHeader, rows: Sequence[EpisodeLedgerRow]
) -> None:
    """Enforce the S7 token PRESENCE RULE in both directions.

    Structure-only episodes must not carry token payloads
    (``episode_ledger_payload_in_structure_only``). Value-mode episodes must
    carry tokens on every complete row WHEN the episode ran to completion
    (all rows complete): emitted tokens derive from the product's own root
    output, which a truncated (halted/aborted/failed) episode never produced,
    so token absence on a truncated ledger is the disclosed consequence of
    the truncation, never incoherence. Interrupted/absent rows never carry
    tokens (the step emitted none).

    Raises
    ------
    ValueError
        With a message carrying the owning refusal code's semantics; callers
        map the structure-only direction onto
        ``episode_ledger_payload_in_structure_only`` and the value-mode
        direction onto ``episode_ledger_incoherent``.
    """

    all_complete = bool(rows) and all(row.status == "complete" for row in rows)
    for row in rows:
        if header.structure_only and row.tokens is not None:
            raise ValueError(
                "episode_ledger_payload_in_structure_only: a structure-only "
                f"episode ledger carries tokens at step {row.episode_step}"
            )
        if row.status in ("interrupted", "absent") and row.tokens is not None:
            raise ValueError(
                "episode-ledger rows that did not complete cannot carry emitted "
                f"tokens (step {row.episode_step}, status {row.status!r})"
            )
        if not header.structure_only and all_complete and row.tokens is None:
            raise ValueError(
                "episode_ledger_incoherent: a value-mode episode ledger is "
                f"missing tokens at complete step {row.episode_step}"
            )


def row_status_for_member_outcome(status: str, phase: str | None) -> RowStatus:
    """Map a member's settled ``(CaptureStatus, CapturePhase)`` to a row status.

    The 2.1 mapping table (a), total over the shipped vocabulary INCLUDING the
    phase-less FAILED record (phase is optional on the shipped record; the row
    is NON-UPGRADING — with the failure phase unattributed, forward-step
    completion cannot be inferred, so the row never claims complete).

    Parameters
    ----------
    status:
        ``CaptureStatus`` value name (e.g. ``"COMPLETE"``) or value string.
    phase:
        ``CapturePhase`` value string, or ``None`` when the settled record
        carries no phase (legal only on FAILED).

    Returns
    -------
    RowStatus
        The derived row status. UNATTESTED/UNKNOWN members return
        ``"interrupted"`` here only as the non-upgrading floor for a
        driver-less derivation; the fold routes such members to
        ``episode_unknown`` regardless (mapping-table note: their rows load
        as UNVERIFIED disclosures and nothing gates on them).
    """

    normalized = status.upper()
    if normalized == "COMPLETE":
        return "complete"
    if normalized in ("HALTED", "ABORTED_NONFINITE"):
        return "interrupted"
    if normalized == "FAILED":
        if phase in ("finalize", "postprocess", "teardown"):
            # The step's FORWARD returned; the failure is post-forward product
            # processing. Nothing is blessed: product truth stays FAILED and
            # the fold consumes the member outcome, not the row.
            return "complete"
        # phase == "forward" or phase absent (unattributed): non-upgrading.
        return "interrupted"
    if normalized in ("UNATTESTED", "UNKNOWN"):
        return "interrupted"
    raise ValueError(f"unknown member CaptureStatus {status!r} for episode-ledger row mapping")


def row_status_for_recording_status(status: str) -> RowStatus:
    """Map a ``Recording.status`` to a row status (2.1 mapping table (b)).

    ``recovered`` maps to ``interrupted`` FAIL-CLOSED: a recovered pass is
    never blessed by derivation (the R06 rule).
    """

    mapping: dict[str, RowStatus] = {
        "complete": "complete",
        "halted": "interrupted",
        "partial_error": "interrupted",
        "recovered": "interrupted",
    }
    try:
        return mapping[status]
    except KeyError:
        raise ValueError(
            f"unknown Recording.status {status!r} for episode-ledger row mapping"
        ) from None


@dataclass(frozen=True)
class EpisodeFoldResult:
    """Result of the floor fold (section 2.3): a DERIVATION, never settlement.

    Parameters
    ----------
    status:
        Derived episode-status term (closed vocabulary).
    at_step:
        Step index qualifying the ``*_at_step`` terms; ``None`` otherwise.
    member_status:
        The deciding member's settled ``CaptureStatus`` string, disclosed
        verbatim (``ABORTED_NONFINITE`` keeps its exact shipped spelling).
    member_phase:
        The deciding FAILED member's phase, disclosed verbatim, or
        ``"unattributed"`` when the shipped record carries none.
    provenance_tier:
        Episode tier: the MINIMUM of the members' tiers (S2 mixed row),
        ``exact`` only when every member is exact.
    """

    status: EpisodeFoldStatus
    at_step: int | None = None
    member_status: str | None = None
    member_phase: str | None = None
    provenance_tier: ProvenanceTier = "exact"


def derive_episode_status(
    member_outcomes: Sequence[tuple[str, str | None]],
    *,
    n_declared: int | None,
    ledger: EpisodeLedger | None,
    escalation_members: frozenset[int] | None = None,
    member_tiers: Sequence[ProvenanceTier] | None = None,
) -> EpisodeFoldResult:
    """Fold member outcomes + ledger geometry into a derived episode status.

    THE FOLD IS A TOTAL FUNCTION over (member ``CaptureStatus`` sequence x
    ledger geometry); arms are evaluated in order, FIRST MATCH WINS, with an
    explicit fail-closed default arm. It never writes a settled
    ``CaptureOutcome`` anywhere — there is no Bundle-level settlement.

    Parameters
    ----------
    member_outcomes:
        Ordered ``(CaptureStatus-string, CapturePhase-string-or-None)`` pairs,
        one per declared step member, prefix order.
    n_declared:
        The episode's declared step count — a LEDGER-ONLY declaration fact.
        ``None`` after a pre-bump round-trip without a re-supplied ledger, in
        which case arms 2 and 4 cannot fire and an all-COMPLETE prefix
        degrades FAIL-CLOSED to ``episode_unknown`` (disclosed truth loss,
        spike section 4).
    ledger:
        The episode ledger when available (in-process floor use), else
        ``None``.
    escalation_members:
        Indices of members that are S6 ``escalates`` targets — excluded from
        the fold domain before evaluation (E-B5: they annotate the episode,
        they are not part of the prefix).
    member_tiers:
        Optional per-member provenance tiers; the episode tier is their
        MINIMUM (``ledger_only`` beats ``exact`` downward).

    Returns
    -------
    EpisodeFoldResult
        The derived status with its qualifying disclosures.
    """

    excluded = escalation_members or frozenset()
    members = [pair for index, pair in enumerate(member_outcomes) if index not in excluded]
    tiers: list[ProvenanceTier] = []
    if member_tiers is not None:
        tiers = [tier for index, tier in enumerate(member_tiers) if index not in excluded]
        for tier in tiers:
            if tier not in _PROVENANCE_TIERS:
                raise ValueError(f"episode provenance tier {tier!r} is out of vocabulary")
    episode_tier: ProvenanceTier = (
        "ledger_only" if any(tier == "ledger_only" for tier in tiers) else "exact"
    )

    # Arm 1: any member UNATTESTED or UNKNOWN, or the ledger violates the law.
    if ledger is not None:
        try:
            _check_monotone_prefix_law(ledger.rows)
        except ValueError:
            return EpisodeFoldResult(status="episode_unknown", provenance_tier=episode_tier)
    for status, _phase in members:
        if status.upper() in ("UNATTESTED", "UNKNOWN"):
            return EpisodeFoldResult(status="episode_unknown", provenance_tier=episode_tier)

    statuses = [status.upper() for status, _phase in members]
    complete_prefix = 0
    for status in statuses:
        if status == "COMPLETE":
            complete_prefix += 1
        else:
            break
    non_complete_tail = statuses[complete_prefix:]

    # Arm 2: all N declared members COMPLETE (needs the ledger-only N).
    if n_declared is not None and not non_complete_tail and len(statuses) == n_declared:
        return EpisodeFoldResult(status="episode_complete", provenance_tier=episode_tier)

    # Arms 3/5/6: exactly one non-complete member, and it is last.
    if len(non_complete_tail) == 1:
        k = complete_prefix
        status = non_complete_tail[0]
        phase = members[k][1]
        if status == "HALTED":
            return EpisodeFoldResult(
                status="episode_halted_at_step",
                at_step=k,
                member_status="HALTED",
                provenance_tier=episode_tier,
            )
        if status == "ABORTED_NONFINITE":
            return EpisodeFoldResult(
                status="episode_aborted_at_step",
                at_step=k,
                member_status="ABORTED_NONFINITE",
                provenance_tier=episode_tier,
            )
        if status == "FAILED":
            return EpisodeFoldResult(
                status="episode_failed_at_step",
                at_step=k,
                member_status="FAILED",
                member_phase=phase if phase is not None else "unattributed",
                provenance_tier=episode_tier,
            )

    # Arm 4: clean COMPLETE prefix + a ledger-declared driver halt at k+1
    # (the until=-at-boundary case: no member for the halted step ever
    # started; by construction witnessed by NO member).
    if (
        not non_complete_tail
        and ledger is not None
        and ledger.truncated_at_step is not None
        and ledger.truncated_at_step == len(statuses)
    ):
        return EpisodeFoldResult(
            status="episode_halted_at_step",
            at_step=len(statuses),
            provenance_tier=episode_tier,
        )

    # Arm 7: anything else — explicit fail-closed default arm. DELIBERATE
    # CASE, not an oversight: a member FAILED post-forward whose driver
    # CONTINUED the episode lands here; episode_unknown is the honest fold.
    return EpisodeFoldResult(status="episode_unknown", provenance_tier=episode_tier)


def producer_digest(outcome_payload: Mapping[str, Any], ledger: EpisodeLedger) -> str:
    """Digest identifying a cheap-tier producer for the E-A2 disclosure.

    Defined (per the spike's E-A2 declaration) over the producer's persisted
    ``_capture_outcome`` payload plus its ledger rows: hex SHA-256 over the
    canonical-JSON encoding of both. Spelling and construction are
    DOCUMENTED-UNSTABLE (naming session 2).
    """

    canonical = json.dumps(
        {
            "outcome": dict(outcome_payload),
            "ledger": ledger.to_payload(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Entry declaration, settlement-time writer, and load validation (session home)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedEpisode:
    """An entry-validated episode declaration bound to a concrete model.

    Produced by :func:`resolve_episode_declaration` at ``tl.trace`` entry,
    after the E-A4 preflight; consumed by the header attach and the
    settlement-time ledger writer.
    """

    episode_id: str
    address: str
    n_steps: int | None
    token_axis: int
    forced_tokens: tuple[int, ...] | None
    escalated_from: str | None
    reason: EscalationReason | None
    expected_tokens: tuple[tuple[int, ...], ...] | None = None


def _declaration_error(message: str, *, code: str | None = None) -> Exception:
    """Build the typed entry refusal (import-local to keep the leaf light)."""

    from ..errors.episode import EpisodeDeclarationError, EpisodeErrorCode

    return EpisodeDeclarationError(
        message,
        code=code if code is not None else EpisodeErrorCode.EPISODE_DECLARATION_INVALID.value,
    )


def _ledger_error(message: str, *, code: str) -> Exception:
    """Build the typed ledger refusal."""

    from ..errors.episode import EpisodeLedgerError

    return EpisodeLedgerError(message, code=code)


def resolve_episode_declaration(spec: EpisodeSpec, model: Any) -> ResolvedEpisode:
    """Validate an episode declaration at entry, BEFORE execution.

    Performs the structural entry checks and the E-A4 preflight: every
    declared episode-carried state item must be snapshot/restorable within
    the declared scope, unconditionally — whether or not any mid-episode
    checkpoint is ever used. Any failure refuses typed here, before the
    forward runs.

    Parameters
    ----------
    spec:
        The user's :class:`torchlens.options.EpisodeSpec`.
    model:
        The episode root about to be traced.

    Returns
    -------
    ResolvedEpisode
        The bound declaration (stepped-module address resolved).

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` for structural problems;
        ``episode_state_unsnapshotable`` for the E-A4 refusal.
    """

    import torch.nn as nn

    from ..errors.episode import EpisodeErrorCode
    from ..options import EpisodeSpec as _EpisodeSpec

    if not isinstance(spec, _EpisodeSpec):
        raise _declaration_error(
            f"episode= expects an EpisodeSpec, got {type(spec).__name__}. "
            "Remedy: pass tl.options.EpisodeSpec(stepped_module=...)"
        )
    stepped = spec.stepped_module
    if not isinstance(stepped, nn.Module):
        raise _declaration_error(
            "EpisodeSpec.stepped_module must be an nn.Module (the stepped model "
            f"whose calls define step boundaries), got {type(stepped).__name__}."
        )
    if stepped is model:
        raise _declaration_error(
            "EpisodeSpec.stepped_module is the episode root itself. The wrapper-"
            "module entry requires the stepped model to be a PROPER submodule of "
            "the traced root (the callable-root entry is a separate, later "
            "contract). Remedy: wrap the generation loop in an nn.Module whose "
            "forward steps this model, and trace the wrapper."
        )
    address: str | None = None
    for name, candidate in model.named_modules():
        if candidate is stepped and name:
            address = name
            break
    if address is None:
        raise _declaration_error(
            "EpisodeSpec.stepped_module is not a submodule of the traced episode "
            "root; the ledger's step boundaries are the stepped module's calls "
            "inside the root's forward."
        )
    if spec.rng != "managed":
        raise _declaration_error(
            f"EpisodeSpec.rng={spec.rng!r} is unsupported; only the 'managed' "
            "seeding discipline ships (the capture's effective random_seed is "
            "recorded as the ledger entry_seed)."
        )
    if spec.n_steps is not None and (not isinstance(spec.n_steps, int) or spec.n_steps <= 0):
        raise _declaration_error(
            f"EpisodeSpec.n_steps={spec.n_steps!r} must be a positive int or None."
        )
    if not isinstance(spec.token_axis, int):
        raise _declaration_error(
            f"EpisodeSpec.token_axis={spec.token_axis!r} must be an int axis index."
        )
    forced: tuple[int, ...] | None = None
    if spec.forced_tokens is not None:
        try:
            forced = tuple(int(token) for token in spec.forced_tokens)
        except (TypeError, ValueError):
            raise _declaration_error(
                "EpisodeSpec.forced_tokens must be an iterable of ints (the "
                "teacher-forced feed), got "
                f"{type(spec.forced_tokens).__name__}."
            ) from None
        if not forced:
            raise _declaration_error("EpisodeSpec.forced_tokens must be non-empty when given.")
        if spec.n_steps is not None and len(forced) != spec.n_steps:
            raise _declaration_error(
                f"EpisodeSpec.forced_tokens has {len(forced)} tokens but n_steps="
                f"{spec.n_steps}; the forced feed must cover exactly the declared steps."
            )
    reason = spec.reason
    if (spec.escalated_from is None) != (reason is None):
        raise _declaration_error(
            "escalated_from and reason must be declared together (the escalation "
            "disclosure is all-or-nothing, E-A2)."
        )
    if reason is not None and reason not in _ESCALATION_REASONS:
        raise _declaration_error(
            f"EpisodeSpec.reason={reason!r} is outside the closed vocabulary "
            f"{sorted(_ESCALATION_REASONS)}."
        )
    # E-A4: declared-state preflight, unconditional, at declaration time.
    import torch

    for index, item in enumerate(spec.state):
        if isinstance(item, torch.Tensor):
            continue  # tensors snapshot/restore by clone within declared scope
        try:
            copy.deepcopy(item)
        except Exception as exc:
            raise _declaration_error(
                f"EpisodeSpec.state[{index}] ({type(item).__name__}) is not "
                "snapshot/restorable within the declared checkpoint scope: "
                f"deepcopy failed with {type(exc).__name__}: {exc}. Remedy: "
                "declare only snapshotable state, or make the item deep-copyable.",
                code=EpisodeErrorCode.EPISODE_STATE_UNSNAPSHOTABLE.value,
            ) from exc
    return ResolvedEpisode(
        episode_id=f"ep-{uuid.uuid4().hex[:16]}",
        address=address,
        n_steps=spec.n_steps,
        token_axis=spec.token_axis,
        forced_tokens=forced,
        escalated_from=spec.escalated_from,
        reason=cast("EscalationReason | None", reason),
        expected_tokens=(
            tuple(tuple(int(t) for t in step) for step in spec.expected_tokens)
            if spec.expected_tokens is not None
            else None
        ),
    )


def attach_episode_header(trace: Any, resolved: ResolvedEpisode) -> None:
    """Stamp the episode declaration marker on the trace BEFORE the forward.

    The pre-capture marker makes the episode declaration visible to
    postprocess consumers (episode folding policy) and rides the partial
    product on a failed forward. The settlement-time writer replaces it with
    the finalized ``{"header", "rows"}`` payload.
    """

    trace.annotations[EPISODE_ANNOTATIONS_KEY] = {
        "declared": {
            "episode_id": resolved.episode_id,
            "capture_kind": CAPTURE_KIND_EPISODE,
            "stepped_module": resolved.address,
            "n_steps_declared": resolved.n_steps,
        }
    }


def _fidelity_basis(
    resolved: ResolvedEpisode, tokens_by_row: list[tuple[int, ...]] | None
) -> FidelityBasis | None:
    """Derive the E-A3 fidelity disclosure for the ledger header.

    A teacher-forced feed is ``forced`` (non-verifying by declaration). An
    escalation with BOTH token columns available compares the escalated
    column against the producer's up to the shorter prefix: equal records
    ``tokens``, a mismatch records ``diverged`` (the escalation FAILED its
    purpose — the product is still a valid capture of what it ran, it just
    is not an escalation of the original episode, and says so). Escalations
    without comparable columns record ``none``. Never a settlement input.
    """

    if resolved.forced_tokens is not None:
        return "forced"
    if resolved.escalated_from is None:
        return None
    if tokens_by_row is None or resolved.expected_tokens is None:
        return "none"
    prefix = min(len(tokens_by_row), len(resolved.expected_tokens))
    if prefix == 0:
        return "none"
    for step in range(prefix):
        if tokens_by_row[step] != resolved.expected_tokens[step]:
            return "diverged"
    return "tokens"


def _build_header(
    trace: Any,
    resolved: ResolvedEpisode,
    *,
    fidelity: FidelityBasis | None = None,
) -> EpisodeLedgerHeader:
    """Build the finalized ledger header from the settled trace."""

    entry_seed = getattr(trace, "random_seed", None)
    if not isinstance(entry_seed, int) or isinstance(entry_seed, bool):
        raise _ledger_error(
            "episode ledger requires the capture's effective random_seed (the "
            "managed RNG recipe); the trace carries none.",
            code="episode_ledger_incoherent",
        )
    forced = resolved.forced_tokens is not None
    return EpisodeLedgerHeader(
        episode_id=resolved.episode_id,
        stepped_module=resolved.address,
        entry_seed=entry_seed,
        n_steps_declared=resolved.n_steps,
        token_feed="forced" if forced else "free",
        provenance_tier="exact",
        structure_only=False,
        escalated_from=resolved.escalated_from,
        reason=resolved.reason,
        fidelity_basis=fidelity,
    )


def _call_returned(call: Any) -> bool:
    """Whether a ModuleCall shows exit evidence (the forward returned).

    ``ModuleCall`` carries no explicit exit flag; output ops, a positive
    forward duration, and a captured output structure are each written only
    by the module-exit event, so any of them witnesses a returned call.
    """

    if getattr(call, "output_ops", None):
        return True
    if getattr(call, "forward_duration", 0.0) and call.forward_duration > 0:
        return True
    return getattr(call, "output_structure", None) is not None


def _pass_range_for_call(call: Any) -> tuple[int, int] | None:
    """Derive the [min, max] op pass-coordinate range inside one call."""

    passes: list[int] = []
    for op_label in getattr(call, "ops", ()) or ():
        text = str(op_label)
        if ":" in text:
            suffix = text.rsplit(":", 1)[1]
            if suffix.isdigit():
                passes.append(int(suffix))
    if not passes:
        return None
    return (min(passes), max(passes))


def _prompt_length(trace: Any, token_axis: int) -> int:
    """Prompt length along the declared token axis (0 for promptless roots).

    A DERIVED arithmetic disclosure: the root input tensor's size along the
    declared token axis when one exists, else 0.
    """

    try:
        input_ops = trace.input_ops
        if not input_ops:
            return 0
        shape = getattr(input_ops[0], "shape", None)
        if not shape:
            return 0
        return int(shape[token_axis])
    except (IndexError, TypeError, ValueError):
        return 0


def _emitted_tokens(trace: Any, resolved: ResolvedEpisode, n_rows: int) -> list[tuple[int, ...]]:
    """Read per-step emitted tokens from the product's own root output.

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` when the root output payload was not
        retained, is not a single integer tensor, or its size along the
        declared token axis disagrees with the step count.
    """

    import torch

    output_ops = list(trace.output_ops)
    if len(output_ops) != 1:
        raise _declaration_error(
            f"episode token derivation requires exactly ONE root output tensor; "
            f"this capture has {len(output_ops)} output layers. Remedy: return "
            "one tensor of emitted tokens from the episode root's forward."
        )
    out = output_ops[0].out
    if out is None:
        raise _declaration_error(
            "episode token derivation requires the root output payload, which "
            "this capture did not retain. Remedy: include the output in the "
            "save= selection (or use the default save policy)."
        )
    if not isinstance(out, torch.Tensor) or out.is_floating_point() or out.is_complex():
        raise _declaration_error(
            "episode token derivation requires an integer token tensor as the "
            "root output; got "
            f"{type(out).__name__ if not isinstance(out, torch.Tensor) else str(out.dtype)}."
        )
    axis = resolved.token_axis
    try:
        axis_size = out.shape[axis]
    except IndexError:
        raise _declaration_error(
            f"token_axis={axis} is out of range for the root output shape {tuple(out.shape)}."
        ) from None
    if axis_size != n_rows:
        raise _declaration_error(
            f"the root output carries {axis_size} steps along token_axis={axis} "
            f"but the episode ledger has {n_rows} rows; the declaration does not "
            "match the executed episode."
        )
    normalized_axis = axis if axis >= 0 else out.dim() + axis
    tokens: list[tuple[int, ...]] = []
    for step in range(n_rows):
        step_slice = out.select(normalized_axis, step)
        tokens.append(tuple(int(v) for v in step_slice.reshape(-1).tolist()))
    return tokens


def write_episode_ledger(trace: Any, resolved: ResolvedEpisode) -> EpisodeLedger:
    """Write the finalized episode ledger onto the settled product (S7 L3).

    Runs ONCE at settlement/finalize, after postprocess, reading only the
    settled ``CaptureOutcome`` (via the public accessor) and the finished
    module-call records. Rows are write-once: a second write refuses typed.

    Returns
    -------
    EpisodeLedger
        The finalized, validated ledger (also attached at
        ``trace.annotations["episode"]`` as its payload).
    """

    existing = trace.annotations.get(EPISODE_ANNOTATIONS_KEY)
    if isinstance(existing, Mapping) and "rows" in existing:
        raise _ledger_error(
            "episode ledger rows are write-once at settlement; this product "
            "already carries a finalized ledger.",
            code="episode_ledger_incoherent",
        )
    outcome = trace.outcome
    status = outcome.status.name if outcome is not None else "UNKNOWN"
    phase_value = getattr(outcome, "phase", None) if outcome is not None else None
    phase = getattr(phase_value, "value", phase_value)

    try:
        module_record = trace.modules[resolved.address]
        calls = list(module_record.calls)
    except Exception:
        calls = []
    started = len(calls)

    if status == "COMPLETE" and resolved.n_steps is not None and started != resolved.n_steps:
        raise _ledger_error(
            f"episode declared n_steps={resolved.n_steps} but the COMPLETE "
            f"capture ran {started} stepped-module calls; the declaration does "
            "not match the executed episode.",
            code="episode_ledger_incoherent",
        )

    n_total = resolved.n_steps if resolved.n_steps is not None else started
    n_total = max(n_total, started)

    statuses: list[RowStatus]
    if status == "COMPLETE" or (
        status == "FAILED" and phase in ("finalize", "postprocess", "teardown")
    ):
        statuses = ["complete"] * started
    elif status in ("HALTED", "ABORTED_NONFINITE") or (
        status == "FAILED" and phase in ("forward", None)
    ):
        if started == 0:
            statuses = []
        else:
            tail: RowStatus = "complete" if _call_returned(calls[-1]) else "interrupted"
            prefix: list[RowStatus] = ["complete"] * (started - 1)
            statuses = [*prefix, tail]
    else:  # UNATTESTED / UNKNOWN: structural, unverified disclosures.
        statuses = [
            cast("RowStatus", "complete" if _call_returned(call) else "interrupted")
            for call in calls
        ]

    prompt_len = _prompt_length(trace, resolved.token_axis)

    tokens_by_row: list[tuple[int, ...]] | None = None
    if (
        statuses
        and all(row_status == "complete" for row_status in statuses)
        and (started == n_total)
    ):
        tokens_by_row = _emitted_tokens(trace, resolved, started)

    header = _build_header(trace, resolved, fidelity=_fidelity_basis(resolved, tokens_by_row))

    frontier: dict[str, str] | None = None
    halt_frontier = getattr(trace, "halt_frontier", None)
    if halt_frontier:
        first = halt_frontier[0] if isinstance(halt_frontier, (list, tuple)) else halt_frontier
        frontier = {"boundary_kind": "op", "boundary_label": str(first)}

    rows: list[EpisodeLedgerRow] = []
    for step in range(n_total):
        if step < started:
            row_status = statuses[step]
            call = calls[step]
            coord: dict[str, Any] = {
                "member_call_index": getattr(call, "call_index", step + 1),
                "pass_range": list(_pass_range_for_call(call) or ()) or None,
            }
        else:
            row_status = "absent"
            coord = {"member_call_index": step + 1, "pass_range": None}
        rows.append(
            EpisodeLedgerRow(
                episode_step=step,
                role="prefill" if step == 0 else "decode",
                status=row_status,
                coord=coord,
                cache_len=prompt_len + step,
                tokens=(tokens_by_row[step] if tokens_by_row is not None else None),
                frontier=(frontier if row_status == "interrupted" else None),
            )
        )

    try:
        ledger = EpisodeLedger(header, rows)
    except ValueError as exc:
        raise _ledger_error(str(exc), code="episode_ledger_incoherent") from exc
    trace.annotations[EPISODE_ANNOTATIONS_KEY] = ledger.to_payload()
    return ledger


def attach_failed_episode_ledger(exc: BaseException, resolved: ResolvedEpisode) -> None:
    """Best-effort ledger attach on the FAILED path (exc.partial_log).

    The partial product has a settled FAILED outcome but no finished
    module-call records, so step truth is recovered from the surviving
    module enter/exit event lanes (exact: an entered call that ran zero
    traced ops is still a started step; only an exit witnesses a return),
    falling back to the raw op module stacks (non-upgrading). Failures here
    only warn — the user's exception is never masked.
    """

    partial = getattr(exc, "partial_log", None)
    trace = getattr(partial, "trace", None)
    if trace is None:
        return
    try:
        started = 0
        returned: int | None = None
        events = getattr(trace, "_capture_events", None) or getattr(trace, "capture_events", None)
        enter_events = getattr(events, "module_enter_events", None) if events else None
        if enter_events:
            started = sum(
                1 for event in enter_events if getattr(event, "address", None) == resolved.address
            )
            exit_events = getattr(events, "module_exit_events", None) or ()
            returned = sum(
                1 for event in exit_events if getattr(event, "address", None) == resolved.address
            )
        else:
            raw_ws = getattr(trace, "_raw_graph_ws", None)
            raw_layers = getattr(raw_ws, "raw_layer_dict", None) or {}
            for raw_op in raw_layers.values():
                for entry in getattr(raw_op, "modules", ()) or ():
                    if (
                        isinstance(entry, tuple)
                        and len(entry) == 2
                        and entry[0] == resolved.address
                    ):
                        started = max(started, int(entry[1]))
        header = _build_header(trace, resolved, fidelity=_fidelity_basis(resolved, None))
        n_total = max(resolved.n_steps or started, started)
        complete_steps = returned if returned is not None else max(started - 1, 0)
        complete_steps = min(complete_steps, started)
        rows: list[EpisodeLedgerRow] = []
        for step in range(n_total):
            if step < complete_steps:
                row_status: RowStatus = "complete"
            elif step < started:
                row_status = "interrupted"
            else:
                row_status = "absent"
            rows.append(
                EpisodeLedgerRow(
                    episode_step=step,
                    role="prefill" if step == 0 else "decode",
                    status=row_status,
                    coord={"member_call_index": step + 1, "pass_range": None},
                    cache_len=step,
                )
            )
        ledger = EpisodeLedger(header, rows)
        trace.annotations[EPISODE_ANNOTATIONS_KEY] = ledger.to_payload()
    except Exception as attach_exc:  # noqa: BLE001 - never mask the user's failure
        from ..errors import TorchLensWarning

        warnings.warn(
            "episode ledger could not be attached to the failed partial "
            f"product: {type(attach_exc).__name__}: {attach_exc}",
            TorchLensWarning,
            stacklevel=2,
        )


def escalation_spec(
    producer: Any,
    *,
    stepped_module: Any,
    reason: EscalationReason,
    n_steps: int | None = None,
    token_axis: int | None = None,
) -> EpisodeSpec:
    """Build the E-A escalation declaration FROM a cheap-tier episode product.

    E-A1: the escalation product of an episode is a NEW whole-episode wrapped
    session capture — same declaration, same inputs, same recorded entry
    seed, re-run from t=0; there is no partial escalation product under the
    session ruling. This helper derives the disclosure fields from the
    producer: ``escalated_from`` (the producer digest over its persisted
    outcome payload + ledger rows), the declared step count, and the
    producer's token column so the write-time E-A3 fidelity comparison can
    discharge (``tokens`` / ``diverged`` / ``none``).

    Re-run the escalation with the producer's recorded entry seed
    (``producer.random_seed``) to satisfy E-A1's same-seed requirement.

    Raises
    ------
    EpisodeLedgerError
        ``episode_ledger_without_declaration`` when the producer carries no
        finalized episode ledger.
    """

    from ..options import EpisodeSpec as _EpisodeSpec

    ledger = episode_ledger_for(producer)
    if ledger is None:
        raise _ledger_error(
            "escalation requires a producer carrying a finalized episode "
            "ledger; this product has none.",
            code="episode_ledger_without_declaration",
        )
    outcome = producer.outcome
    outcome_payload = outcome.to_payload() if outcome is not None else {}
    digest = producer_digest(outcome_payload, ledger)
    expected = tuple(row.tokens for row in ledger.rows if row.tokens is not None)
    return _EpisodeSpec(
        stepped_module=stepped_module,
        n_steps=n_steps if n_steps is not None else ledger.header.n_steps_declared,
        token_axis=token_axis if token_axis is not None else -1,
        escalated_from=digest,
        reason=reason,
        expected_tokens=expected if expected else None,
    )


def episode_ledger_for(trace: Any) -> EpisodeLedger | None:
    """Parse and return the trace's episode ledger, or ``None``.

    Returns ``None`` for non-episode products, pre-finalize declarations,
    and quarantined (load-degraded) payloads.
    """

    payload = getattr(trace, "annotations", {}).get(EPISODE_ANNOTATIONS_KEY)
    if not isinstance(payload, Mapping) or set(payload) != {"header", "rows"}:
        return None
    try:
        return EpisodeLedger.from_payload(payload)
    except (TypeError, ValueError):
        return None


def validate_loaded_episode_annotations(trace: Trace) -> None:
    """Validate a loaded ``annotations["episode"]`` payload, FAIL-CLOSED.

    Load semantics (S7 L1/L2/L4 + the S2 combination table):

    - A payload whose header does not declare ``capture_kind=episode``
      refuses typed (``episode_ledger_without_declaration``): an episode
      ledger on a non-episode capture is ILLEGAL.
    - A structure-only ledger carrying tokens refuses typed
      (``episode_ledger_payload_in_structure_only``).
    - Any other geometry/parse violation QUARANTINES the payload (replaced
      by a diagnostic record; rows stop being claims) with ONE warning, and
      records the fail-closed degrade signal the outcome derivation consumes
      at the coordinated schema bump (the ledger never upgrades an outcome;
      the degrade wiring in the settlement authority is the S2 author's bump
      change, tracked on the record written here).
    - On UNATTESTED/UNKNOWN products a valid ledger loads as an UNVERIFIED
      disclosure — kept as-is; nothing gates on it.
    """

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict):
        return
    payload = annotations.get(EPISODE_ANNOTATIONS_KEY)
    if payload is None:
        return
    if isinstance(payload, Mapping) and set(payload) == {"quarantined", "code", "detail"}:
        return  # already-quarantined record round-tripping
    if isinstance(payload, Mapping) and set(payload) == {"declared"}:
        return  # pre-finalize declaration marker (session-time shape)
    parse_error: str | None = None
    if isinstance(payload, Mapping) and set(payload) == {"header", "rows"}:
        header_payload = payload.get("header")
        if (
            isinstance(header_payload, Mapping)
            and header_payload.get("capture_kind") != CAPTURE_KIND_EPISODE
        ):
            raise _ledger_error(
                "an episode ledger is attached to a capture whose header does "
                "not declare capture_kind=episode; an episode ledger without "
                "the declaration is illegal.",
                code="episode_ledger_without_declaration",
            )
        try:
            EpisodeLedger.from_payload(payload)
            return  # valid — keep as claims (or unverified disclosure)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            if "episode_ledger_payload_in_structure_only" in message:
                raise _ledger_error(
                    message, code="episode_ledger_payload_in_structure_only"
                ) from exc
            parse_error = message
    else:
        parse_error = "episode annotations payload does not carry the {'header', 'rows'} shape"
    from ..errors import TorchLensWarning

    warnings.warn(
        "episode ledger failed load validation and was quarantined "
        f"(episode_ledger_incoherent): {parse_error}. The product's episode "
        "rows are no longer claims; the outcome derivation treats the ledger "
        "fail-closed.",
        TorchLensWarning,
        stacklevel=2,
    )
    annotations[EPISODE_ANNOTATIONS_KEY] = {
        "quarantined": True,
        "code": "episode_ledger_incoherent",
        "detail": parse_error,
    }
