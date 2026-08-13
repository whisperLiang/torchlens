"""``MergedTrace``: a presenter over lazy rank handles + the join table.

Composition, never inheritance (design-merge-ranks-c v5, 2.1): a merged trace
is NOT a ``Trace`` and NOT a ``Bundle``. Rank cores stay the single truth
(P1); the presenter only references them. Structural access is authority --
``merged.ranks[r][label]`` and ``merged.super_op(...)``; the ``r{rank}/label``
string is validated sugar. Refused surfaces raise typed
(:class:`MergedSurfaceUnsupportedError`); merged replay does not exist.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._engine import JoinKey, JoinRecord, MergeDerivation, derive_merge
from ._enums import BoundaryConsistency, MergeAlignment, MergedErrorCode, MergeValueStatus
from ._errors import MergeConflictError, MergedFinding, MergedSurfaceUnsupportedError
from ._evidence import resolve_rank_inputs

__all__ = [
    "CollectiveJoin",
    "MergeReport",
    "MergedTrace",
    "merge_ranks",
    "merge_report",
]


class _RankHandle:
    """Lazy handle on one rank core: a live trace or an on-disk bundle."""

    def __init__(self, rank: int, trace: Any = None, path: str | None = None) -> None:
        self.rank = rank
        self.path = path
        self._trace = trace

    @property
    def trace(self) -> Any:
        """The rank's ``Trace``, loading from disk on first access."""

        if self._trace is None:
            from .._io.bundle import load as load_bundle

            assert self.path is not None, "a rank handle has a trace or a path"
            self._trace = load_bundle(self.path)
        return self._trace

    @property
    def is_loaded(self) -> bool:
        """Whether the trace is currently materialized in memory."""

        return self._trace is not None


class _RankMapping(Mapping[int, Any]):
    """Read-only ``rank -> Trace`` mapping with lazy loading."""

    def __init__(self, handles: Mapping[int, _RankHandle]) -> None:
        self._handles = dict(handles)

    def __getitem__(self, rank: int) -> Any:
        return self._handles[int(rank)].trace

    def __iter__(self) -> Iterator[int]:
        return iter(self._handles)

    def __len__(self) -> int:
        return len(self._handles)


@dataclass(frozen=True)
class CollectiveJoin:
    """Presenter view of one cross-rank join (references, never copies)."""

    record: JoinRecord

    @property
    def key(self) -> JoinKey:
        return self.record.key

    @property
    def kind(self) -> str:
        return self.record.kind

    @property
    def membership(self) -> tuple[int, ...]:
        return self.record.membership

    @property
    def presence(self) -> tuple[int, ...]:
        return self.record.presence

    @property
    def missing(self) -> tuple[int, ...]:
        return self.record.missing

    @property
    def consistency(self) -> BoundaryConsistency:
        return self.record.consistency

    def op_labels_raw(self, rank: int) -> tuple[str, ...]:
        """The rank core's raw op-label back-references for this join."""

        return self.record.per_rank[int(rank)].op_labels_raw

    def __repr__(self) -> str:
        digest, ordinal, channel, delta = self.record.key
        return (
            f"CollectiveJoin({self.record.kind} {digest[:12]}…/{ordinal}/"
            f"{channel}/{delta}, presence={list(self.record.presence)}, "
            f"consistency={self.record.consistency.value})"
        )


@dataclass(frozen=True)
class MergeReport:
    """Graph-free diagnostic summary of a merge derivation (3.2)."""

    ranks: tuple[int, ...]
    expected_ranks: tuple[int, ...] | None
    stored_alignment: MergeAlignment
    alignment: MergeAlignment
    value_status: MergeValueStatus
    n_joins: int
    findings: tuple[MergedFinding, ...]
    load_degradations: tuple[str, ...] = ()

    @property
    def gaps(self) -> tuple[MergedFinding, ...]:
        """The ``presence_gap`` findings: boundaries a rank never recorded."""

        return tuple(f for f in self.findings if f.kind == "presence_gap")

    @property
    def divergences(self) -> tuple[MergedFinding, ...]:
        """The ``value_divergence`` findings: ranks disagreeing on a witnessed value."""

        return tuple(f for f in self.findings if f.kind == "value_divergence")

    @property
    def conflicts(self) -> tuple[MergedFinding, ...]:
        """The structural conflict findings, which never join and never become gaps."""

        structural = {
            "group_lifetime_evidence_conflict",
            "relation_violation",
            "order_contradiction",
            "correlation_delta_mismatch",
        }
        return tuple(f for f in self.findings if f.kind in structural)

    def to_markdown(self) -> str:
        """Render a compact human-readable report."""

        lines = [
            "# Cross-rank merge report",
            f"- ranks: {list(self.ranks)}"
            + ("" if self.expected_ranks is None else f" (declared: {list(self.expected_ranks)})"),
            f"- alignment: {self.alignment.value}"
            + (
                f" (stored: {self.stored_alignment.value}; "
                f"{len(self.load_degradations)} load degradation(s))"
                if self.alignment is not self.stored_alignment
                else ""
            ),
            f"- value status: {self.value_status.value}",
            f"- joins: {self.n_joins}",
        ]
        for finding in self.findings:
            lines.append(f"- [{finding.kind}] {finding.detail}")
        return "\n".join(lines)


def _report_from(
    derivation: MergeDerivation, load_degradations: tuple[str, ...] = ()
) -> MergeReport:
    """Build the report projection of a derivation."""

    findings = derivation.findings
    if load_degradations:
        findings = findings + tuple(
            MergedFinding(kind="load_degradation", detail=detail) for detail in load_degradations
        )
    effective = derivation.stored_alignment
    if load_degradations and effective is MergeAlignment.ALIGNED:
        effective = MergeAlignment.PARTIAL
    return MergeReport(
        ranks=derivation.ranks,
        expected_ranks=derivation.expected_ranks,
        stored_alignment=derivation.stored_alignment,
        alignment=effective,
        value_status=derivation.stored_value_status,
        n_joins=len(derivation.joins),
        findings=findings,
        load_degradations=load_degradations,
    )


class MergedTrace:
    """N rank-local captures stitched at their collective boundaries.

    Never construct directly: use :func:`merge_ranks` (live/loaded inputs)
    or :func:`torchlens.merged.load` (a saved ``merged-directory`` artifact).
    """

    def __init__(
        self,
        derivation: MergeDerivation,
        handles: Mapping[int, _RankHandle],
        load_degradations: tuple[str, ...] = (),
    ) -> None:
        self._derivation = derivation
        self._handles = dict(handles)
        self._load_degradations = tuple(load_degradations)
        self._source_path: str | None = None

    # ------------------------------------------------------------------
    # Verdicts (3.2): stored vs effective are DISTINCT properties.
    # ------------------------------------------------------------------

    @property
    def stored_alignment(self) -> MergeAlignment:
        """The structural verdict derived at merge time (frozen in the descriptor)."""

        return self._derivation.stored_alignment

    @property
    def alignment(self) -> MergeAlignment:
        """The EFFECTIVE verdict: stored, lowered by load degradations."""

        if self._load_degradations and self.stored_alignment is MergeAlignment.ALIGNED:
            return MergeAlignment.PARTIAL
        return self.stored_alignment

    @property
    def value_status(self) -> MergeValueStatus:
        """Merge-level value verdict from the join ledger (3.3)."""

        return self._derivation.stored_value_status

    @property
    def load_degradations(self) -> tuple[str, ...]:
        """Environment degradations recorded at load (empty on a live merge)."""

        return self._load_degradations

    # ------------------------------------------------------------------
    # Structure.
    # ------------------------------------------------------------------

    @property
    def ranks(self) -> Mapping[int, Any]:
        """Lazy ``rank -> Trace`` mapping (structural access authority)."""

        return _RankMapping(self._handles)

    @property
    def rank_ids(self) -> tuple[int, ...]:
        """Global ranks present in the merge."""

        return self._derivation.ranks

    @property
    def expected_ranks(self) -> tuple[int, ...] | None:
        """Declared world, when one was given (can only widen expectations)."""

        return self._derivation.expected_ranks

    @property
    def joins(self) -> tuple[CollectiveJoin, ...]:
        """The cross-rank collective joins, in deterministic key order."""

        return tuple(CollectiveJoin(record) for record in self._derivation.joins)

    @property
    def findings(self) -> tuple[MergedFinding, ...]:
        """All merge findings (gaps, divergences, load degradations)."""

        return _report_from(self._derivation, self._load_degradations).findings

    @property
    def gaps(self) -> tuple[MergedFinding, ...]:
        """The typed presence-gap ledger."""

        return self._derivation.gap_findings

    @property
    def report(self) -> MergeReport:
        """The structured merge report."""

        return _report_from(self._derivation, self._load_degradations)

    # ------------------------------------------------------------------
    # Access surface (2.5).
    # ------------------------------------------------------------------

    def __getitem__(self, item: Any) -> Any:
        if not isinstance(item, str):
            raise MergedSurfaceUnsupportedError(
                "Merged-level selectors are not supported in this release; "
                "index a single rank core (merged.ranks[r][selector]) instead.",
                code=MergedErrorCode.MERGED_SELECTOR_UNSUPPORTED,
            )
        rank, label = self._split_sugar(item)
        if rank is not None:
            return self.ranks[rank][label]
        hits = {}
        for rank_id in self.rank_ids:
            try:
                hits[rank_id] = self.ranks[rank_id][item]
            except Exception:
                continue
        if not hits:
            raise KeyError(item)
        if len(hits) > 1:
            from .._errors import AmbiguousOpLookupError

            spellings = ", ".join(f"r{rank_id}/{item}" for rank_id in sorted(hits))
            raise AmbiguousOpLookupError(
                f"Label {item!r} resolves on {len(hits)} ranks; disambiguate "
                f"with a rank-qualified label ({spellings}) or use "
                "merged.super_op() for the cross-rank fan."
            )
        return next(iter(hits.values()))

    def _split_sugar(self, item: str) -> tuple[int | None, str]:
        """Parse the validated ``r{rank}/label`` sugar; real names win."""

        if "/" in item and item.startswith("r"):
            prefix, _, rest = item.partition("/")
            if prefix[1:].isdigit() and rest:
                rank = int(prefix[1:])
                if rank in self._handles:
                    return rank, rest
        return None, item

    def super_op(self, label: str) -> dict[int, Any]:
        """Return the SPMD fan: the ops matching ``label`` on every rank.

        Parameters
        ----------
        label:
            A rank-core op/layer label (unqualified).

        Returns
        -------
        dict[int, Any]
            Mapping from global rank to that rank's matching record; ranks
            where the label does not resolve are absent.
        """

        fan: dict[int, Any] = {}
        for rank_id in self.rank_ids:
            try:
                fan[rank_id] = self.ranks[rank_id][label]
            except Exception:
                continue
        if not fan:
            raise KeyError(label)
        return fan

    def join_ops(self, join: CollectiveJoin) -> dict[int, tuple[Any, ...]]:
        """Resolve a join's boundary ops on every presenting rank core.

        Uses each rank core's persisted raw-to-final label mapping; a
        tensorless boundary (barrier, object collectives) has no op node and
        yields an empty tuple for that rank.
        """

        resolved: dict[int, tuple[Any, ...]] = {}
        for rank in join.presence:
            trace = self.ranks[rank]
            mapping = getattr(trace, "_raw_to_final_op_labels", {}) or {}
            ops = []
            for raw in join.op_labels_raw(rank):
                final = mapping.get(raw, raw)
                try:
                    ops.append(trace[final])
                except Exception:
                    continue
            resolved[rank] = tuple(ops)
        return resolved

    def happens_before(self, left: JoinKey, right: JoinKey) -> bool | None:
        """Partial-order query over join keys (2.2).

        Returns ``True``/``False`` for provable order, ``None`` for
        genuinely concurrent joins (no path either way). Order flows only
        through rank-local issue order and the joins themselves; there is no
        global step numbering.
        """

        joins = {join.key: join for join in self.joins}
        if left not in joins or right not in joins:
            raise KeyError("happens_before takes keys of existing joins")
        successors: dict[JoinKey, set[JoinKey]] = {key: set() for key in joins}
        per_rank_sequence: dict[int, list[tuple[int, JoinKey]]] = {}
        for key, join in joins.items():
            for rank in join.presence:
                index = join.record.per_rank[rank].boundary_index
                per_rank_sequence.setdefault(rank, []).append((index, key))
        for sequence in per_rank_sequence.values():
            sequence.sort()
            for (_, earlier), (_, later) in zip(sequence, sequence[1:]):
                successors[earlier].add(later)

        def reachable(source: JoinKey, target: JoinKey) -> bool:
            """Whether ``target`` is reachable from ``source`` in the merged successor graph."""

            frontier, seen = [source], {source}
            while frontier:
                node = frontier.pop()
                for successor in successors[node]:
                    if successor == target:
                        return True
                    if successor not in seen:
                        seen.add(successor)
                        frontier.append(successor)
            return False

        if left == right:
            return False
        if reachable(left, right):
            return True
        if reachable(right, left):
            return False
        return None

    def to_pandas(self) -> Any:
        """Join table as a DataFrame with a ``(rank, rank_local_index)`` MultiIndex."""

        import pandas as pd

        rows = []
        index = []
        for join in self.joins:
            digest, ordinal, channel, delta = join.key
            for rank in join.presence:
                ref = join.record.per_rank[rank]
                index.append((rank, ref.boundary_index))
                rows.append(
                    {
                        "kind": join.kind,
                        "membership_digest": digest,
                        "lifetime_ordinal": ordinal,
                        "channel": channel,
                        "seq_delta": delta,
                        "seq_abs": ref.seq_abs,
                        "group_size": len(join.membership),
                        "presence": len(join.presence),
                        "consistency": join.consistency.value,
                    }
                )
        frame = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(index, names=["rank", "rank_local_index"]),
        )
        return frame.sort_index()

    # ------------------------------------------------------------------
    # Presentation.
    # ------------------------------------------------------------------

    def _witness_coverage_line(self) -> str:
        """The witness-coverage disclosure every ``aligned`` presentation carries."""

        joins = self._derivation.joins
        attested = sum(1 for j in joins if j.consistency is BoundaryConsistency.ATTESTED)
        applicable = sum(
            1 for j in joins if j.consistency is not BoundaryConsistency.NOT_APPLICABLE
        )
        return (
            f"witness coverage: {attested}/{applicable} applicable join(s) "
            f"attested, {len(joins) - applicable} not applicable "
            f"(value status: {self.value_status.value})"
        )

    def summary(self) -> str:
        """Human-readable merge summary."""

        alignment = self.alignment.value
        if self.alignment is not self.stored_alignment:
            alignment = (
                f"{alignment} (stored: {self.stored_alignment.value}; "
                f"{len(self._load_degradations)} load degradation(s))"
            )
        lines = [
            f"MergedTrace over ranks {list(self.rank_ids)}: alignment={alignment}",
            self._witness_coverage_line(),
            f"{len(self._derivation.joins)} collective join(s), "
            f"{len(self.gaps)} presence gap(s), "
            f"{len(self._derivation.divergence_findings)} value divergence(s)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MergedTrace(ranks={list(self.rank_ids)}, "
            f"alignment={self.alignment.value}, "
            f"value_status={self.value_status.value}, "
            f"joins={len(self._derivation.joins)})"
        )

    # ------------------------------------------------------------------
    # Refused surfaces (2.5 / 3.4): merged replay does not exist.
    # ------------------------------------------------------------------

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: merged replay does not exist (typed)."""

        raise MergedSurfaceUnsupportedError(
            "Merged replay does not exist: re-issuing collectives outside "
            "their communicator hangs or fabricates peer-dependent values. "
            "Run a single rank core instead (merged.ranks[r]).",
            code=MergedErrorCode.MERGE_RUN_UNSUPPORTED,
        )

    def validate(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: per-rank validation belongs to the rank cores (typed)."""

        raise MergedSurfaceUnsupportedError(
            "Merged validation does not exist; validate each rank core "
            "(noting that collective-crossing cores refuse forward replay "
            "typed while metadata invariants run in full).",
            code=MergedErrorCode.MERGED_SURFACE_UNSUPPORTED,
        )

    # Artifact save lives in _artifact.py; bound late to avoid an import cycle.
    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the merged artifact as a ``merged-directory`` bundle."""

        from ._artifact import save_merged

        save_merged(self, path, overwrite=overwrite)
        self._source_path = str(path)


def merge_ranks(
    inputs: Sequence[Any],
    *,
    expected_ranks: Iterable[int] | None = None,
) -> MergedTrace:
    """Merge N rank-local captures at their collective boundaries (rung C1).

    Parameters
    ----------
    inputs:
        Live/loaded ``Trace`` objects and/or rank-core ``.tlspec`` paths, one
        per rank, in any order. Every input must have been captured under the
        distributed opt-in.
    expected_ranks:
        Optional declared world. Widens presence expectations only: a
        declared rank without a core is a presence gap; declaring fewer
        ranks than the recorded group memberships never narrows anything.

    Returns
    -------
    MergedTrace
        The merged presenter over the rank cores and the join table.

    Raises
    ------
    MergeInputError
        On invalid inputs or out-of-C1-scope boundaries (p2p/pipeline: C3;
        DTensor topologies: C2).
    MergeConflictError
        When the presented cores structurally contradict each other
        (pre-join lineage-audit conflicts, relation violations, order
        contradictions, correlation cross-check disagreements). The
        graph-free diagnostic escape hatch is :func:`merge_report`.
    """

    resolved = resolve_rank_inputs(inputs)
    derivation = derive_merge(
        {rank: evidence for rank, (evidence, _trace) in resolved.items()},
        expected_ranks,
    )
    structural = derivation.structural_findings
    if structural:
        raise MergeConflictError(
            f"{len(structural)} structural conflict(s) between the presented "
            "rank cores; see fields['findings']. Diagnose without merging via "
            "torchlens.merged.merge_report().",
            code=MergedErrorCode.MERGE_CONFLICT
            if not all(f.kind == "group_lifetime_evidence_conflict" for f in structural)
            else MergedErrorCode.GROUP_LIFETIME_EVIDENCE_CONFLICT,
            findings=structural,
        )
    handles = {}
    for rank, (evidence, trace) in resolved.items():
        path = evidence.source if not evidence.source.startswith("live[") else None
        handles[rank] = _RankHandle(rank, trace=trace, path=path)
    return MergedTrace(derivation, handles)


def merge_report(
    inputs: Sequence[Any],
    *,
    expected_ranks: Iterable[int] | None = None,
) -> MergeReport:
    """Graph-free merge diagnostic: derive verdicts without constructing.

    Unlike :func:`merge_ranks` this never raises on structural conflicts --
    it is the escape hatch for diagnosing exactly those. Input validation
    (unloadable path, malformed core, out-of-scope boundary) still refuses
    typed.
    """

    resolved = resolve_rank_inputs(inputs)
    derivation = derive_merge(
        {rank: evidence for rank, (evidence, _trace) in resolved.items()},
        expected_ranks,
    )
    return _report_from(derivation)
