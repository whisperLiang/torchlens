"""Bundle member-relation table (S6): typed, closed, OPTIONAL relations.

One optional relation table per :class:`~torchlens.bundle.Bundle`. Absence
means a plain Bundle with unchanged semantics; membership NEVER requires
alignment or relations (the S6 flexibility invariant). The table is DATA,
not derivation: loads never recompute it, but its invariants (R1/R2/R3) are
re-checked on every load and every mutation.

Pattern-of-record: :mod:`torchlens.distributed._ledger` — closed
vocabularies, frozen payload key sets, fail-closed ``from_payload``,
immutable identity-stable finalized views. Parse boundaries raise
``ValueError``; the public Bundle wiring converts into the typed
:class:`~torchlens.errors.BundleRelationError` family (stable codes on
``exc.fields["code"]``, never message text).

Schema of record: DRAFT S6 in the L2 spike memo, BINDING as of the S2
ratification. Every spelling is DOCUMENTED-UNSTABLE pending the rolling
naming session (spike section 6.5); semantics are pinned.

Payload spelling: PAIR rows persist their endpoints under the S6 keys
``from``/``to`` (``from`` is a Python keyword, so the dataclass attributes
are ``from_member``/``to_member``); MEMBER rows persist under ``member``.

PERSISTENCE IS GATED (S3 version discipline): ``member_relations`` is a NEW
``bundle.json`` key and never rides a real tlspec-v7 artifact. The writer
ships inert behind the pre-release registrar switch
(:mod:`torchlens._io.prerelease`); the coordinated wave-3 bump activates it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors.episode import BundleRelationError

__all__ = [
    "MEMBER_ROW_KINDS",
    "PAIR_ROW_KINDS",
    "MemberRelationRow",
    "MemberRelationTable",
]

#: PAIR-row kinds -> the EXACT param-key set each kind declares (closed, v1).
PAIR_ROW_KINDS: dict[str, frozenset[str]] = {
    "alternative_of": frozenset(),
    "forked_from": frozenset({"at_step"}),
    "successor_of": frozenset(),
    "escalates": frozenset({"at_step"}),
}

#: MEMBER-row kinds -> the EXACT param-key set each kind declares (closed, v1).
MEMBER_ROW_KINDS: dict[str, frozenset[str]] = {
    "episode_member": frozenset({"episode_id", "at_step", "role"}),
}

_EPISODE_ROLES = frozenset({"prefill", "decode"})

# ``to_payload`` emits exactly these keys per shape; an unknown key in a
# loaded payload is a forged or drifted artifact, never silently ignored.
_PAIR_PAYLOAD_KEYS = frozenset({"kind", "from", "to", "params"})
_MEMBER_PAYLOAD_KEYS = frozenset({"kind", "member", "params"})


def _require_member_name(value: Any, *, key: str, kind: str) -> str:
    """Return a required non-empty member-name string, refusing anything else."""

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"bundle-relation row of kind {kind!r} requires a non-empty member "
            f"name for {key!r}, got {value!r}"
        )
    return value


def _validated_params(kind: str, params: Any) -> dict[str, Any]:
    """Return the closed, per-kind validated param mapping for one row.

    Raises
    ------
    ValueError
        On a non-mapping, an unknown or missing param key, or an ill-typed
        param value (the S6 R2 refusal family; callers convert into
        ``bundle_relation_schema_invalid``).
    """

    if kind in PAIR_ROW_KINDS:
        declared = PAIR_ROW_KINDS[kind]
    else:
        declared = MEMBER_ROW_KINDS[kind]
    if not isinstance(params, Mapping):
        raise ValueError(
            f"bundle-relation row params must be a mapping, got {type(params).__name__}"
        )
    unknown = set(params) - declared
    if unknown:
        raise ValueError(
            f"bundle-relation row of kind {kind!r} carries undeclared param "
            f"keys {sorted(unknown)}; declared keys are {sorted(declared)}"
        )
    missing = declared - set(params)
    if missing:
        raise ValueError(
            f"bundle-relation row of kind {kind!r} is missing required param keys {sorted(missing)}"
        )
    validated: dict[str, Any] = {}
    for key in sorted(declared):
        value = params[key]
        _validate_param_value(key, value)
        validated[key] = value
    return validated


def _validate_param_value(key: str, value: Any) -> None:
    """Type/vocabulary check for one declared relation param value."""

    if key == "at_step":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"bundle-relation param 'at_step' must be a non-negative int, got {value!r}"
            )
    elif key == "episode_id":
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"bundle-relation param 'episode_id' must be a non-empty string, got {value!r}"
            )
    elif key == "role" and value not in _EPISODE_ROLES:
        raise ValueError(
            f"bundle-relation param 'role' is {value!r}, outside the closed "
            f"vocabulary {sorted(_EPISODE_ROLES)}"
        )


@dataclass(frozen=True)
class MemberRelationRow:
    """One frozen S6 relation row; ``kind`` selects one of TWO closed shapes.

    PAIR rows (kinds ``alternative_of`` / ``forked_from`` / ``successor_of``
    / ``escalates``) carry ``from_member``/``to_member`` (persisted as the
    S6 ``from``/``to`` keys) and leave ``member`` unset. MEMBER rows (kind
    ``episode_member``) carry ``member`` and leave the endpoints unset.
    ``params`` holds exactly the keys the kind declares — closed per-kind
    vocabularies, validated at construction and again at every load.
    """

    kind: str
    from_member: str | None = None
    to_member: str | None = None
    member: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind in PAIR_ROW_KINDS:
            _require_member_name(self.from_member, key="from", kind=self.kind)
            _require_member_name(self.to_member, key="to", kind=self.kind)
            if self.member is not None:
                raise ValueError(
                    f"bundle-relation PAIR row of kind {self.kind!r} must not "
                    "carry the MEMBER-shape 'member' endpoint"
                )
        elif self.kind in MEMBER_ROW_KINDS:
            _require_member_name(self.member, key="member", kind=self.kind)
            if self.from_member is not None or self.to_member is not None:
                raise ValueError(
                    f"bundle-relation MEMBER row of kind {self.kind!r} must not "
                    "carry the PAIR-shape 'from'/'to' endpoints"
                )
        else:
            known = sorted(set(PAIR_ROW_KINDS) | set(MEMBER_ROW_KINDS))
            raise ValueError(
                f"bundle-relation row kind {self.kind!r} is outside the closed vocabulary {known}"
            )
        object.__setattr__(self, "params", _validated_params(self.kind, self.params))

    def named_members(self) -> tuple[str, ...]:
        """Return the member names this row references, in schema order."""

        if self.kind in PAIR_ROW_KINDS:
            # __post_init__ proved both endpoints are non-empty strings.
            return (str(self.from_member), str(self.to_member))
        return (str(self.member),)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload in the S6 key spelling."""

        if self.kind in PAIR_ROW_KINDS:
            return {
                "kind": self.kind,
                "from": self.from_member,
                "to": self.to_member,
                "params": dict(self.params),
            }
        return {
            "kind": self.kind,
            "member": self.member,
            "params": dict(self.params),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MemberRelationRow:
        """Rebuild one row from :meth:`to_payload` output, FAIL-CLOSED.

        Raises
        ------
        ValueError
            On an unknown kind, a wrong row shape for the kind, an unknown
            or missing payload/param key, or an ill-typed value (callers
            convert into ``bundle_relation_schema_invalid``).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("bundle-relation row payload must be a mapping")
        kind = payload.get("kind")
        if kind in PAIR_ROW_KINDS:
            expected_keys = _PAIR_PAYLOAD_KEYS
        elif kind in MEMBER_ROW_KINDS:
            expected_keys = _MEMBER_PAYLOAD_KEYS
        else:
            known = sorted(set(PAIR_ROW_KINDS) | set(MEMBER_ROW_KINDS))
            raise ValueError(
                f"bundle-relation row kind {kind!r} is outside the closed vocabulary {known}"
            )
        if set(payload) != expected_keys:
            raise ValueError(
                f"bundle-relation row of kind {kind!r} must carry exactly the "
                f"keys {sorted(expected_keys)}, got {sorted(payload)}"
            )
        if kind in PAIR_ROW_KINDS:
            return cls(
                kind=str(kind),
                from_member=_require_member_name(payload["from"], key="from", kind=str(kind)),
                to_member=_require_member_name(payload["to"], key="to", kind=str(kind)),
                params=_validated_params(str(kind), payload["params"]),
            )
        return cls(
            kind=str(kind),
            member=_require_member_name(payload["member"], key="member", kind=str(kind)),
            params=_validated_params(str(kind), payload["params"]),
        )


class MemberRelationTable:
    """Immutable S6 relation table: an identity-stable tuple of frozen rows.

    Mutation never happens in place (R4): ``Bundle.relate`` and the R5
    cascade paths build a NEW validated table, so views handed out never
    change underfoot.
    """

    def __init__(self, rows: Sequence[MemberRelationRow] = ()) -> None:
        for row in rows:
            if not isinstance(row, MemberRelationRow):
                raise TypeError(
                    "MemberRelationTable rows must be MemberRelationRow instances, "
                    f"got {type(row).__name__}"
                )
        self._rows: tuple[MemberRelationRow, ...] = tuple(rows)

    @property
    def rows(self) -> tuple[MemberRelationRow, ...]:
        """Immutable, identity-stable view of the relation rows in order."""

        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Any:
        return iter(self._rows)

    def rows_naming(self, member_name: str) -> tuple[MemberRelationRow, ...]:
        """Return the rows that reference ``member_name`` (R5 mutator guards)."""

        return tuple(row for row in self._rows if member_name in row.named_members())

    def to_payload(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable payload for portable artifacts."""

        return [row.to_payload() for row in self._rows]

    @classmethod
    def from_payload(cls, payload: Any) -> MemberRelationTable:
        """Rebuild a table from :meth:`to_payload` output, FAIL-CLOSED.

        Raises
        ------
        ValueError
            On a non-list payload or any invalid row payload (callers
            convert into ``bundle_relation_schema_invalid``).
        """

        if not isinstance(payload, list):
            raise ValueError(
                f"bundle-relation table payload must be a list, got {type(payload).__name__}"
            )
        return cls([MemberRelationRow.from_payload(entry) for entry in payload])

    def validate_against_members(self, member_names: Iterable[str]) -> None:
        """Re-check the load-time invariants against a concrete member set.

        R1: every named member must be a current Bundle member — no dangling
        edges, ever. R3 kind-scoped invariants: ``episode_member`` rows for
        one ``episode_id`` carry distinct ``at_step`` values and exactly one
        ``role="prefill"`` row, sitting at ``at_step`` 0.

        Raises
        ------
        BundleRelationError
            ``bundle_relation_member_missing`` for a dangling row (R1) or
            ``bundle_relation_schema_invalid`` for a kind-scoped invariant
            violation (R3).
        """

        names = set(member_names)
        for index, row in enumerate(self._rows):
            for named in row.named_members():
                if named not in names:
                    raise BundleRelationError(
                        f"bundle relation row {index} (kind {row.kind!r}) names "
                        f"member {named!r}, which is not a Bundle member. "
                        "Relation rows may only reference current members (S6 R1).",
                        code="bundle_relation_member_missing",
                        row_index=index,
                        row_kind=row.kind,
                        missing_member=named,
                    )
        episodes: dict[str, list[MemberRelationRow]] = {}
        for row in self._rows:
            if row.kind == "episode_member":
                episodes.setdefault(str(row.params["episode_id"]), []).append(row)
        for episode_id, rows in episodes.items():
            steps = [int(row.params["at_step"]) for row in rows]
            if len(steps) != len(set(steps)):
                raise BundleRelationError(
                    f"episode_member rows for episode {episode_id!r} carry "
                    "duplicate at_step values; steps must be distinct (S6 R3).",
                    code="bundle_relation_schema_invalid",
                    episode_id=episode_id,
                )
            prefill_rows = [row for row in rows if row.params["role"] == "prefill"]
            if len(prefill_rows) != 1 or int(prefill_rows[0].params["at_step"]) != 0:
                raise BundleRelationError(
                    f"episode_member rows for episode {episode_id!r} must carry "
                    "exactly one role='prefill' row at at_step 0 (S6 R3).",
                    code="bundle_relation_schema_invalid",
                    episode_id=episode_id,
                )
