"""The C1 merge derivation: audit-first joining, delta alignment, witnesses.

ONE pure function (:func:`derive_merge`) produces the complete merge
derivation from per-rank evidence. It runs at merge time and again, verbatim,
at load rederivation (design-merge-ranks-c v5, 4.3) -- the saved descriptor is
only a cache of its output.

Order of operations is normative (v5 1.3): the PRE-JOIN membership-lineage
audit runs BEFORE any correlation-key joining or presence-gap derivation; a
conflicted membership refuses structurally -- no uid of it joins and no
unmatched key of it becomes a presence gap.

Correlation alignment is by counting (P3): per ``(group_uid, channel)`` each
rank's recorded boundaries align as seq DELTAS from that rank's first recorded
key -- absolute seq bases are rank-local facts (arm-time histories differ) and
are never compared. Witness digests are redundant byte-exact evidence that can
only DEMOTE a verdict, never rescue or repair one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from ..distributed._audit import MembershipLineageVerdict, audit_membership_lineages
from ..distributed._ledger import InstallEpoch
from ._enums import (
    WITNESS_IDENTITY_KINDS,
    WITNESS_NOT_APPLICABLE_KINDS,
    WITNESS_VERDICT_BACKENDS,
    BoundaryConsistency,
    MergeAlignment,
    MergedErrorCode,
    MergeValueStatus,
)
from ._errors import MergedFinding, MergeInputError
from ._evidence import P2P_KINDS, RankEvidence

__all__ = [
    "JoinKey",
    "JoinRecord",
    "MergeDerivation",
    "PerRankRef",
    "derive_merge",
]

JoinKey = tuple[str, int, str, int]
"""``(membership_digest, lifetime_ordinal, channel, seq_delta)``."""

_STRUCTURAL_KINDS = frozenset(
    {
        "group_lifetime_evidence_conflict",
        "relation_violation",
        "order_contradiction",
        "correlation_delta_mismatch",
    }
)


@dataclass(frozen=True)
class PerRankRef:
    """One rank's contribution to a join: references, never copies (P1)."""

    rank: int
    boundary_index: int
    seq_abs: int
    op_labels_raw: tuple[str, ...]
    group_rank: int | None
    witness_policy: str
    n_contribution_roles: int
    n_destination_roles: int
    contribution_digests: tuple[str, ...] | None
    destination_digests: tuple[str, ...] | None
    c10d_group_seq: int | None

    def to_payload(self) -> dict[str, Any]:
        """Canonical JSON projection (descriptor cache row)."""

        return {
            "rank": self.rank,
            "boundary_index": self.boundary_index,
            "seq_abs": self.seq_abs,
            "op_labels_raw": list(self.op_labels_raw),
            "group_rank": self.group_rank,
            "witness_policy": self.witness_policy,
            "n_contribution_roles": self.n_contribution_roles,
            "n_destination_roles": self.n_destination_roles,
            "contribution_digests": (
                None if self.contribution_digests is None else list(self.contribution_digests)
            ),
            "destination_digests": (
                None if self.destination_digests is None else list(self.destination_digests)
            ),
            "c10d_group_seq": self.c10d_group_seq,
        }


@dataclass(frozen=True)
class JoinRecord:
    """One cross-rank collective join at component level."""

    key: JoinKey
    kind: str
    reduce_op: str | None
    membership: tuple[int, ...]
    backend: str | None
    presence: tuple[int, ...]
    missing: tuple[int, ...]
    per_rank: Mapping[int, PerRankRef]
    consistency: BoundaryConsistency

    def to_payload(self) -> dict[str, Any]:
        """Canonical JSON projection (descriptor cache row)."""

        return {
            "key": list(self.key),
            "kind": self.kind,
            "reduce_op": self.reduce_op,
            "membership": list(self.membership),
            "backend": self.backend,
            "presence": list(self.presence),
            "missing": list(self.missing),
            "per_rank": {
                str(rank): ref.to_payload() for rank, ref in sorted(self.per_rank.items())
            },
            "consistency": self.consistency.value,
        }


@dataclass(frozen=True)
class MergeDerivation:
    """Complete, deterministic output of the one merge derivation."""

    ranks: tuple[int, ...]
    expected_ranks: tuple[int, ...] | None
    joins: tuple[JoinRecord, ...]
    findings: tuple[MergedFinding, ...]
    audit: Mapping[str, MembershipLineageVerdict]
    stored_alignment: MergeAlignment
    stored_value_status: MergeValueStatus
    conflicted_memberships: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structural_findings(self) -> tuple[MergedFinding, ...]:
        """Findings that make the merge ``conflicted``."""

        return tuple(f for f in self.findings if f.kind in _STRUCTURAL_KINDS)

    @property
    def gap_findings(self) -> tuple[MergedFinding, ...]:
        """The typed presence-gap ledger."""

        return tuple(f for f in self.findings if f.kind == "presence_gap")

    @property
    def divergence_findings(self) -> tuple[MergedFinding, ...]:
        """Value-divergence findings (never structural)."""

        return tuple(f for f in self.findings if f.kind == "value_divergence")

    def to_payload(self) -> dict[str, Any]:
        """Canonical JSON projection: the descriptor cache body.

        Pure function of the input rank cores -- contains no timestamps, no
        environment strings, and no absolute paths, so byte-equality of this
        payload is the load-time tamper oracle.
        """

        return {
            "ranks": list(self.ranks),
            "expected_ranks": (None if self.expected_ranks is None else list(self.expected_ranks)),
            "joins": [join.to_payload() for join in self.joins],
            "findings": [finding.to_payload() for finding in self.findings],
            "audit": {
                digest: {
                    "status": verdict.status,
                    "kind": verdict.kind,
                    "presenting_ranks": list(verdict.presenting_ranks),
                    "complete_witness_ranks": list(verdict.complete_witness_ranks),
                }
                for digest, verdict in sorted(self.audit.items())
            },
            "conflicted_memberships": list(self.conflicted_memberships),
            "stored_alignment": self.stored_alignment.value,
            "stored_value_status": self.stored_value_status.value,
        }


def _guard_scope(evidence: Mapping[int, RankEvidence]) -> None:
    """Refuse out-of-C1-scope boundaries typed (p2p/pipeline: C3; DTensor: C2)."""

    for rank in sorted(evidence):
        for index, entry in enumerate(evidence[rank].boundaries):
            kind = entry["kind"]
            channel = entry["correlation"]["channel"]
            if kind in P2P_KINDS or channel != "coll":
                raise MergeInputError(
                    f"Rank {rank} boundary {index} is a point-to-point boundary "
                    f"(kind={kind!r}, channel={channel!r}). C1 merges symmetric "
                    "collectives only; p2p/pipeline pairing is rung C3.",
                    code=MergedErrorCode.MERGE_SCOPE_UNSUPPORTED,
                    rank=rank,
                    kind=kind,
                    channel=channel,
                )
            for role in entry.get("roles", ()):
                if role.get("logical_shape") is not None or role.get("placements") is not None:
                    raise MergeInputError(
                        f"Rank {rank} boundary {index} carries DTensor dual "
                        "geometry; sharded-topology merging is rung C2 and stays "
                        "refused until its capture-fidelity census is green.",
                        code=MergedErrorCode.MERGE_SCOPE_UNSUPPORTED,
                        rank=rank,
                        kind=kind,
                    )


def _role_digests(entry: dict[str, Any]) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Return ``(contribution_digests, destination_digests)`` or ``None`` each."""

    witness = entry["witness"]
    contribution = witness.get("contribution_digests")
    destination = witness.get("destination_digests")
    return (
        None if contribution is None else tuple(contribution),
        None if destination is None else tuple(destination),
    )


def _roles_of(entry: dict[str, Any], role_names: Iterable[str]) -> list[dict[str, Any]]:
    """Return the ordered role entries matching ``role_names``."""

    wanted = set(role_names)
    return [role for role in entry.get("roles", ()) if role.get("role") in wanted]


def _relation_findings(
    key: JoinKey,
    kind: str,
    membership: tuple[int, ...],
    entries: dict[int, dict[str, Any]],
) -> list[MergedFinding]:
    """Evaluate the per-kind relation table over the presenting entries (1.5)."""

    findings: list[MergedFinding] = []
    ranks = tuple(sorted(entries))

    def violation(detail: str) -> None:
        """Append one ``relation_violation`` finding for this key."""

        findings.append(
            MergedFinding(
                kind="relation_violation",
                detail=f"{kind} at key {key}: {detail}",
                membership_digest=key[0],
                key=key,
                ranks=ranks,
            )
        )

    kinds_seen = {entry["kind"] for entry in entries.values()}
    if len(kinds_seen) > 1:
        violation(f"ranks disagree on the collective kind: {sorted(kinds_seen)}")
        return findings

    reduce_ops = {entry.get("reduce_op") for entry in entries.values()}
    if len(reduce_ops) > 1:
        violation(f"ranks disagree on the reduce op: {sorted(map(str, reduce_ops))}")

    def shapes(entry: dict[str, Any], names: Iterable[str]) -> tuple[tuple[int, ...], ...]:
        """Collect the shapes of ``entry``'s roles matching any of ``names``, in order."""

        return tuple(tuple(role["shape"]) for role in _roles_of(entry, names))

    group_size = len(membership)
    contribution_roles = ("contribution", "contribution_destination")
    destination_roles = ("destination", "contribution_destination")
    if kind in ("all_reduce", "all_to_all_single", "all_gather_into_tensor", "broadcast"):
        shape_sets = {
            shapes(entry, contribution_roles) or shapes(entry, destination_roles)
            for entry in entries.values()
        }
        if len(shape_sets) > 1:
            violation(f"tensor shapes disagree across ranks: {sorted(shape_sets)}")
    if kind == "all_gather":
        for rank, entry in sorted(entries.items()):
            n_destinations = len(_roles_of(entry, destination_roles))
            if n_destinations != group_size:
                violation(
                    f"rank {rank} records {n_destinations} destination(s) for a "
                    f"group of {group_size}"
                )
    if kind == "all_to_all":
        for rank, entry in sorted(entries.items()):
            n_in = len(_roles_of(entry, contribution_roles))
            n_out = len(_roles_of(entry, destination_roles))
            if n_in != group_size or n_out != group_size:
                violation(
                    f"rank {rank} records {n_in} contribution(s) and {n_out} "
                    f"destination(s) for a group of {group_size}"
                )
    if kind in ("broadcast", "reduce", "gather", "scatter"):
        # Root-only aggregate presence (1.5): at most one rank may present
        # the root's role signature.
        if kind == "broadcast":
            roots = [
                rank
                for rank, entry in sorted(entries.items())
                if not _roles_of(entry, ("destination",))
            ]
        elif kind == "reduce":
            roots = [
                rank
                for rank, entry in sorted(entries.items())
                if _roles_of(entry, destination_roles)
            ]
        else:
            # gather roots hold a destination LIST; scatter roots a
            # contribution LIST (group_size entries; leaves hold one tensor).
            list_roles = destination_roles if kind == "gather" else contribution_roles
            roots = [
                rank
                for rank, entry in sorted(entries.items())
                if len(_roles_of(entry, list_roles)) == group_size and group_size > 1
            ]
        if len(roots) > 1:
            violation(f"multiple ranks {roots} present root-only aggregate roles")
    return findings


def _witness_consistency(
    kind: str,
    backend: str | None,
    membership: tuple[int, ...],
    presence: tuple[int, ...],
    per_rank: Mapping[int, PerRankRef],
) -> BoundaryConsistency:
    """Totalized per-join witness derivation (v5 3.3; demote-only)."""

    if kind in WITNESS_NOT_APPLICABLE_KINDS:
        return BoundaryConsistency.NOT_APPLICABLE
    if backend is None or backend not in WITNESS_VERDICT_BACKENDS:
        return BoundaryConsistency.NOT_APPLICABLE
    group_size = len(membership)

    if kind in WITNESS_IDENTITY_KINDS:
        # Every member rank ends holding identical bytes. The comparable
        # digest is the rank's DESTINATION digest; a broadcast root has no
        # destination role and its contribution bytes ARE the final bytes.
        # A rank with destination roles but absent destination digests
        # (witness level "none", async completion unobserved) contributes
        # nothing -- its pre-collective contribution digest is never a
        # substitute (that comparison could only fabricate a mismatch).
        values: list[tuple[str, ...]] = []
        for rank in presence:
            ref = per_rank[rank]
            if ref.n_destination_roles:
                value = ref.destination_digests
            else:
                value = ref.contribution_digests
            if value:
                values.append(tuple(value))
        if len(set(values)) > 1:
            return BoundaryConsistency.MISMATCHED
        if len(values) < group_size:
            return BoundaryConsistency.NOT_PRESENT
        return BoundaryConsistency.ATTESTED

    # Slice kinds: gather / scatter / all_to_all, group-rank indexed pairs.
    pairs: list[tuple[str, str]] = []
    incomplete = len(presence) < group_size
    by_group_rank: dict[int, int] = {}
    for rank in presence:
        group_rank = per_rank[rank].group_rank
        if group_rank is not None:
            by_group_rank[group_rank] = rank
    if len(by_group_rank) < len(presence):
        incomplete = True
    if kind in ("gather", "scatter"):
        # gather: root's destination list slice[i] == member i's contribution.
        # scatter: member i's destination == root's contribution list slice[i].
        root_list_attr = "destination_digests" if kind == "gather" else "contribution_digests"
        leaf_attr = "contribution_digests" if kind == "gather" else "destination_digests"
        n_list_attr = "n_destination_roles" if kind == "gather" else "n_contribution_roles"
        roots = [r for r in presence if getattr(per_rank[r], n_list_attr) == group_size]
        expected_pairs = group_size
        if len(roots) != 1:
            incomplete = True
        else:
            root_list = getattr(per_rank[roots[0]], root_list_attr)
            if not root_list or len(root_list) != group_size:
                incomplete = True
            else:
                for group_rank in sorted(by_group_rank):
                    rank = by_group_rank[group_rank]
                    leaf = getattr(per_rank[rank], leaf_attr)
                    if not leaf:
                        incomplete = True
                        continue
                    pairs.append((root_list[group_rank], leaf[0]))
    else:  # all_to_all: rank j's destination[i] == rank i's contribution[j]
        expected_pairs = group_size * group_size
        for gr_j in sorted(by_group_rank):
            dest_j = per_rank[by_group_rank[gr_j]].destination_digests
            if not dest_j:
                incomplete = True
                continue
            for gr_i in sorted(by_group_rank):
                contrib_i = per_rank[by_group_rank[gr_i]].contribution_digests
                if not contrib_i or gr_i >= len(dest_j) or gr_j >= len(contrib_i):
                    incomplete = True
                    continue
                pairs.append((dest_j[gr_i], contrib_i[gr_j]))
    if any(left != right for left, right in pairs):
        return BoundaryConsistency.MISMATCHED
    if incomplete or len(pairs) < expected_pairs:
        return BoundaryConsistency.NOT_PRESENT
    return BoundaryConsistency.ATTESTED


def _check_order(
    evidence: Mapping[int, RankEvidence],
    joined_nodes: Mapping[tuple[int, int], JoinKey],
) -> MergedFinding | None:
    """Cycle check over rank-local order + join identifications (2.2)."""

    parent: dict[Any, Any] = {}

    def find(node: Any) -> Any:
        """Union-find representative of ``node``, with path compression."""

        root = node
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(node, node) != node:
            parent[node], node = root, parent[node]
        return root

    def union(a: Any, b: Any) -> None:
        """Merge the classes of ``a`` and ``b``."""

        parent[find(a)] = find(b)

    for (rank, index), key in joined_nodes.items():
        union((rank, index), ("join", key))

    edges: dict[Any, set[Any]] = {}
    indegree: dict[Any, int] = {}
    nodes: set[Any] = set()
    for rank in sorted(evidence):
        boundary_nodes = [
            find((rank, index)) if (rank, index) in joined_nodes else (rank, index)
            for index in range(len(evidence[rank].boundaries))
        ]
        for node in boundary_nodes:
            nodes.add(node)
        for left, right in zip(boundary_nodes, boundary_nodes[1:]):
            if left == right:
                continue
            if right not in edges.setdefault(left, set()):
                edges[left].add(right)
                indegree[right] = indegree.get(right, 0) + 1
                nodes.add(right)
    queue = [node for node in nodes if indegree.get(node, 0) == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for successor in edges.get(node, ()):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if visited != len(nodes):
        return MergedFinding(
            kind="order_contradiction",
            detail=(
                "the joined cross-rank order contradicts at least one rank's "
                "local issue order (a cycle exists through the join graph); "
                "the presented cores cannot describe one consistent execution."
            ),
            ranks=tuple(sorted(evidence)),
        )
    return None


def derive_merge(
    evidence: Mapping[int, RankEvidence],
    expected_ranks: Iterable[int] | None = None,
) -> MergeDerivation:
    """Derive the complete merge from per-rank evidence (the ONE function).

    Parameters
    ----------
    evidence:
        Mapping from global rank to validated rank evidence.
    expected_ranks:
        Optional declared world. Can only WIDEN presence expectations beyond
        the group memberships recorded inside the rank cores (4.3) -- a
        declared rank with no core is a presence gap; declaring fewer ranks
        than the recorded memberships never narrows anything.

    Returns
    -------
    MergeDerivation
        Joins, findings ledgers, audit verdicts, and stored verdicts.

    Raises
    ------
    MergeInputError
        On out-of-scope (p2p/pipeline/DTensor) boundaries.
    """

    _guard_scope(evidence)
    input_ranks = tuple(sorted(evidence))
    declared = None if expected_ranks is None else tuple(sorted({int(r) for r in expected_ranks}))
    findings: list[MergedFinding] = []

    # 1. PRE-JOIN membership-lineage audit, before ANY joining or gaps (1.3).
    epochs = cast(
        "Mapping[int, InstallEpoch]",
        {rank: ev.install_epoch for rank, ev in evidence.items()},
    )
    audit = audit_membership_lineages({rank: ev.ledger for rank, ev in evidence.items()}, epochs)
    conflicted_memberships = tuple(
        sorted(digest for digest, verdict in audit.items() if verdict.is_conflict)
    )
    for digest in conflicted_memberships:
        verdict = audit[digest]
        findings.append(
            MergedFinding(
                kind="group_lifetime_evidence_conflict",
                detail=verdict.detail,
                membership_digest=digest,
                ranks=verdict.presenting_ranks,
            )
        )

    # 2. Group table from the recorded memberships (rank cores are authority).
    group_members: dict[tuple[str, int], tuple[int, ...]] = {}
    group_backend: dict[tuple[str, int], str | None] = {}
    for rank in input_ranks:
        for index, entry in enumerate(evidence[rank].boundaries):
            correlation = entry["correlation"]
            uid = (correlation["membership_digest"], correlation["lifetime_ordinal"])
            members = tuple(int(r) for r in entry["group"]["global_ranks"])
            backend = entry["group"].get("backend")
            if uid in group_members and group_members[uid] != members:
                findings.append(
                    MergedFinding(
                        kind="relation_violation",
                        detail=(
                            f"rank {rank} records group {uid} with member order "
                            f"{members} but another rank recorded "
                            f"{group_members[uid]}; c10d group rank lists must be "
                            "identical on every member rank."
                        ),
                        membership_digest=uid[0],
                        ranks=input_ranks,
                    )
                )
            group_members.setdefault(uid, members)
            group_backend.setdefault(uid, backend)

    # 3. Correlation joining: seq DELTAS from each rank's first recorded key.
    per_channel: dict[tuple[str, int, str], dict[int, list[tuple[int, int]]]] = {}
    for rank in input_ranks:
        for index, entry in enumerate(evidence[rank].boundaries):
            correlation = entry["correlation"]
            digest = correlation["membership_digest"]
            if digest in conflicted_memberships:
                continue  # no join AND no gap for a conflicted membership (1.3)
            channel_key = (digest, correlation["lifetime_ordinal"], correlation["channel"])
            per_channel.setdefault(channel_key, {}).setdefault(rank, []).append(
                (int(correlation["seq"]), index)
            )

    joins: list[JoinRecord] = []
    joined_nodes: dict[tuple[int, int], JoinKey] = {}
    for channel_key in sorted(per_channel):
        digest, ordinal, channel = channel_key
        by_rank = per_channel[channel_key]
        deltas: dict[int, dict[int, tuple[int, int]]] = {}
        for rank, records in by_rank.items():
            ordered = sorted(records)
            base = ordered[0][0]
            for seq_abs, index in ordered:
                deltas.setdefault(seq_abs - base, {})[rank] = (seq_abs, index)
        membership = group_members[(digest, ordinal)]
        backend = group_backend[(digest, ordinal)]
        c10d_base: dict[int, int] = {}
        for delta in sorted(deltas):
            entries_at = deltas[delta]
            key: JoinKey = (digest, ordinal, channel, delta)
            entries = {
                rank: evidence[rank].boundaries[index] for rank, (_seq, index) in entries_at.items()
            }
            per_rank: dict[int, PerRankRef] = {}
            for rank, (seq_abs, index) in sorted(entries_at.items()):
                entry = entries[rank]
                contribution, destination = _role_digests(entry)
                per_rank[rank] = PerRankRef(
                    rank=rank,
                    boundary_index=index,
                    seq_abs=seq_abs,
                    op_labels_raw=tuple(entry["op_labels_raw"]),
                    group_rank=entry["group"].get("my_group_rank"),
                    witness_policy=entry["witness"]["policy_resolved"],
                    n_contribution_roles=len(
                        _roles_of(entry, ("contribution", "contribution_destination"))
                    ),
                    n_destination_roles=len(
                        _roles_of(entry, ("destination", "contribution_destination"))
                    ),
                    contribution_digests=contribution,
                    destination_digests=destination,
                    c10d_group_seq=entry.get("c10d_group_seq"),
                )
                joined_nodes[(rank, index)] = key
            presence = tuple(sorted(per_rank))
            missing = tuple(sorted(set(membership) - set(presence)))
            findings.extend(
                _relation_findings(key, entries[presence[0]]["kind"], membership, entries)
            )

            # Redundant c10d group-seq cross-check, as deltas from the first
            # joined key (1.6): a disagreement means c10d itself orders these
            # issues differently than the join claims.
            for rank in presence:
                value = per_rank[rank].c10d_group_seq
                if value is not None and rank not in c10d_base and delta == 0:
                    c10d_base[rank] = value
            c10d_deltas: dict[int, int] = {}
            for rank in presence:
                observed = per_rank[rank].c10d_group_seq
                if observed is not None and rank in c10d_base:
                    c10d_deltas[rank] = observed - c10d_base[rank]
            if len(set(c10d_deltas.values())) > 1:
                findings.append(
                    MergedFinding(
                        kind="correlation_delta_mismatch",
                        detail=(
                            f"c10d group-seq deltas disagree at key {key}: "
                            f"{dict(sorted(c10d_deltas.items()))}; the backend's own "
                            "sequence counter contradicts the join alignment."
                        ),
                        membership_digest=digest,
                        key=key,
                        ranks=tuple(sorted(c10d_deltas)),
                    )
                )

            if missing:
                findings.append(
                    MergedFinding(
                        kind="presence_gap",
                        detail=(
                            f"member rank(s) {list(missing)} of group {digest[:12]}… "
                            f"(ordinal {ordinal}) did not present key {key}."
                        ),
                        membership_digest=digest,
                        key=key,
                        ranks=missing,
                    )
                )
            kind = entries[presence[0]]["kind"]
            reduce_ops = sorted(
                {
                    str(entry.get("reduce_op"))
                    for entry in entries.values()
                    if entry.get("reduce_op") is not None
                }
            )
            consistency = _witness_consistency(kind, backend, membership, presence, per_rank)
            if consistency is BoundaryConsistency.MISMATCHED:
                findings.append(
                    MergedFinding(
                        kind="value_divergence",
                        detail=(
                            f"byte-exact witness digests disagree at key {key} "
                            f"({kind}); rank values are NOT interchangeable. The "
                            "divergence is recorded, never repaired (P3)."
                        ),
                        membership_digest=digest,
                        key=key,
                        ranks=presence,
                    )
                )
            joins.append(
                JoinRecord(
                    key=key,
                    kind=kind,
                    reduce_op=reduce_ops[0] if reduce_ops else None,
                    membership=membership,
                    backend=backend,
                    presence=presence,
                    missing=missing,
                    per_rank=per_rank,
                    consistency=consistency,
                )
            )

    # 4. Declared expected ranks can only WIDEN expectations (4.3).
    if declared is not None:
        absent = tuple(sorted(set(declared) - set(input_ranks)))
        if absent:
            findings.append(
                MergedFinding(
                    kind="presence_gap",
                    detail=(f"declared expected rank(s) {list(absent)} presented no rank core."),
                    ranks=absent,
                )
            )

    # 5. Partial-order consistency.
    order_finding = _check_order(evidence, joined_nodes)
    if order_finding is not None:
        findings.append(order_finding)

    # 6. Verdicts.
    structural = [f for f in findings if f.kind in _STRUCTURAL_KINDS]
    gaps = [f for f in findings if f.kind == "presence_gap"]
    if structural:
        alignment = MergeAlignment.CONFLICTED
    elif gaps:
        alignment = MergeAlignment.PARTIAL
    else:
        alignment = MergeAlignment.ALIGNED

    applicable = [j for j in joins if j.consistency is not BoundaryConsistency.NOT_APPLICABLE]
    attested = [j for j in applicable if j.consistency is BoundaryConsistency.ATTESTED]
    mismatched = [j for j in applicable if j.consistency is BoundaryConsistency.MISMATCHED]
    if mismatched:
        value_status = MergeValueStatus.DIVERGENT
    elif applicable and len(attested) == len(applicable):
        value_status = MergeValueStatus.ATTESTED_COMPLETE
    elif attested:
        value_status = MergeValueStatus.ATTESTED_PARTIAL
    else:
        value_status = MergeValueStatus.UNWITNESSED

    return MergeDerivation(
        ranks=input_ranks,
        expected_ranks=declared,
        joins=tuple(joins),
        findings=tuple(findings),
        audit=dict(sorted(audit.items())),
        stored_alignment=alignment,
        stored_value_status=value_status,
        conflicted_memberships=conflicted_memberships,
    )
