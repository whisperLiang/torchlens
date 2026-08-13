"""Rank-core evidence extraction for the merge engine.

One extraction path serves live traces, loaded traces, and rank-core bundle
paths, so merge time and load rederivation consume IDENTICAL evidence
(design-merge-ranks-c v5, 4.3: one derivation function called twice). The
extractor validates the ``collective_boundary_v1`` payloads against their
closed vocabularies at parse time; a malformed core refuses typed
(``merged_schema_invalid``) instead of degrading silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..distributed._ledger import GroupLifecycleLedger
from ._enums import MergedErrorCode
from ._errors import MergeInputError

__all__ = [
    "BOUNDARY_SCHEMA",
    "KNOWN_KINDS",
    "P2P_KINDS",
    "RankEvidence",
    "extract_rank_evidence",
    "resolve_rank_inputs",
]

BOUNDARY_SCHEMA = "collective_boundary_v1"

KNOWN_KINDS = frozenset(
    {
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "broadcast",
        "reduce",
        "all_to_all",
        "all_to_all_single",
        "gather",
        "scatter",
        "send",
        "recv",
        "barrier",
        "all_gather_object",
        "broadcast_object_list",
        "gather_object",
        "scatter_object_list",
    }
)
"""The closed C0 boundary-kind vocabulary; an unknown kind refuses typed."""

P2P_KINDS = frozenset({"send", "recv"})
"""Point-to-point kinds: out of C1 scope (pipeline pairing is rung C3)."""

_COMPLETION_BINDINGS = frozenset({"issue_sync", "unobserved"})
_WITNESS_POLICIES = frozenset({"none", "digest"})
_INSTALL_EPOCHS = frozenset({"armed_before_any_group", "seeded"})


@dataclass(frozen=True)
class RankEvidence:
    """Everything the merge engine reads from one rank core.

    Parameters
    ----------
    rank:
        The core's global rank, proven by its own boundary records.
    boundaries:
        The ordered boundary entries from the trace-level distributed
        journal (issue order; each carries the ``collective_boundary_v1``
        payload plus ``op_labels_raw`` back-references).
    ledger:
        The rank's group-lifecycle ledger, rebuilt from its portable payload.
    install_epoch:
        The rank's arming install-epoch record.
    source:
        Where the evidence came from (diagnostic): ``"live"``, ``"loaded"``,
        or the bundle path string.
    """

    rank: int
    boundaries: tuple[dict[str, Any], ...]
    ledger: GroupLifecycleLedger
    install_epoch: str
    source: str


def _refuse(detail: str, **payload: Any) -> MergeInputError:
    """Build the typed parse refusal for malformed rank evidence."""

    return MergeInputError(
        f"Rank-core evidence is not a valid collective_boundary_v1 journal: {detail}",
        code=MergedErrorCode.MERGED_SCHEMA_INVALID,
        **payload,
    )


def _validate_boundary(entry: dict[str, Any], index: int, source: str) -> None:
    """Validate one journal entry against the closed C0 vocabularies."""

    where = f"boundary {index} of {source}"
    if entry.get("schema") != BOUNDARY_SCHEMA:
        raise _refuse(f"{where} has schema {entry.get('schema')!r}", source=source)
    kind = entry.get("kind")
    if kind not in KNOWN_KINDS:
        raise _refuse(f"{where} has unknown kind {kind!r}", source=source)
    correlation = entry.get("correlation")
    if not isinstance(correlation, dict) or set(correlation) != {
        "membership_digest",
        "lifetime_ordinal",
        "channel",
        "seq",
    }:
        raise _refuse(f"{where} has a malformed correlation key", source=source)
    if not isinstance(correlation["membership_digest"], str):
        raise _refuse(f"{where} membership_digest is not a string", source=source)
    if not isinstance(correlation["lifetime_ordinal"], int) or isinstance(
        correlation["lifetime_ordinal"], bool
    ):
        raise _refuse(f"{where} lifetime_ordinal is not an integer", source=source)
    if not isinstance(correlation["seq"], int) or isinstance(correlation["seq"], bool):
        raise _refuse(f"{where} seq is not an integer", source=source)
    group = entry.get("group")
    if not isinstance(group, dict) or not isinstance(group.get("global_ranks"), list):
        raise _refuse(f"{where} has no group membership record", source=source)
    if not isinstance(group.get("my_global_rank"), int):
        raise _refuse(f"{where} has no my_global_rank", source=source)
    events = entry.get("events")
    if not isinstance(events, dict) or events.get("completion_binding") not in _COMPLETION_BINDINGS:
        raise _refuse(f"{where} has a malformed events record", source=source)
    witness = entry.get("witness")
    if not isinstance(witness, dict) or witness.get("policy_resolved") not in _WITNESS_POLICIES:
        raise _refuse(
            f"{where} has witness policy {witness.get('policy_resolved') if isinstance(witness, dict) else witness!r} "
            "outside the closed vocabulary",
            source=source,
        )
    if not isinstance(entry.get("op_labels_raw"), list):
        raise _refuse(f"{where} lacks op_labels_raw back-references", source=source)


def extract_rank_evidence(trace: Any, source: str) -> RankEvidence:
    """Extract and validate one rank core's merge evidence.

    Parameters
    ----------
    trace:
        A finished (live or loaded) ``Trace`` captured under the distributed
        opt-in.
    source:
        Diagnostic origin string recorded on the evidence.

    Returns
    -------
    RankEvidence
        Validated evidence for the engine.

    Raises
    ------
    MergeInputError
        If the trace carries no distributed record, its journal is malformed,
        or its boundary records disagree about the rank's own identity.
    """

    record = getattr(trace, "annotations", {}).get("distributed")
    if not isinstance(record, dict) or not record.get("boundaries"):
        raise MergeInputError(
            f"Merge input {source} carries no distributed boundary evidence "
            "(trace.annotations['distributed'] is absent or empty). Only rank "
            "captures taken under the distributed opt-in "
            "(torchlens.distributed.arm() or SPMD lazy arming) can be merged.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            source=source,
        )
    boundaries = record["boundaries"]
    ranks: set[int] = set()
    for index, entry in enumerate(boundaries):
        if not isinstance(entry, dict):
            raise _refuse(f"boundary {index} of {source} is not a mapping", source=source)
        _validate_boundary(entry, index, source)
        ranks.add(int(entry["group"]["my_global_rank"]))
    if len(ranks) != 1:
        raise MergeInputError(
            f"Merge input {source} claims multiple global ranks {sorted(ranks)}; "
            "a rank core is a single-rank capture.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            source=source,
        )
    install_epoch = record.get("install_epoch")
    if install_epoch not in _INSTALL_EPOCHS:
        raise _refuse(
            f"install_epoch {install_epoch!r} outside the closed vocabulary",
            source=source,
        )
    ledger_payload = record.get("group_lifecycle_ledger")
    if not isinstance(ledger_payload, list) or not ledger_payload:
        raise _refuse("group_lifecycle_ledger is absent or empty", source=source)
    try:
        ledger = GroupLifecycleLedger.from_payload(ledger_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(f"group_lifecycle_ledger does not parse ({exc})", source=source) from exc
    # ``lineage_vectors()`` stamps each vector's install epoch from the EVENTS, while
    # the audit's completeness reasoning also reads the record-level epoch. A rank
    # has exactly one epoch, so a disagreement is a forged/corrupt sidecar trying to
    # promote a ``seeded`` rank to a complete witness; refuse rather than let the two
    # readings diverge.
    event_epochs = {event.install_epoch for event in ledger.events}
    if event_epochs != {install_epoch}:
        raise _refuse(
            f"group_lifecycle_ledger event install_epochs {sorted(event_epochs)} disagree "
            f"with the record install_epoch {install_epoch!r}",
            source=source,
        )
    return RankEvidence(
        rank=ranks.pop(),
        boundaries=tuple(boundaries),
        ledger=ledger,
        install_epoch=str(install_epoch),
        source=source,
    )


def resolve_rank_inputs(inputs: Sequence[Any]) -> dict[int, tuple[RankEvidence, Any]]:
    """Resolve merge inputs (traces or bundle paths) into per-rank evidence.

    Parameters
    ----------
    inputs:
        Live/loaded ``Trace`` objects or ``.tlspec`` rank-core paths, in any
        mix and order. Paths are loaded for analysis.

    Returns
    -------
    dict[int, tuple[RankEvidence, Any]]
        Mapping from global rank to ``(evidence, trace)``, ordered by rank.

    Raises
    ------
    MergeInputError
        On empty input, unloadable paths, duplicate ranks, or invalid cores.
    """

    if not inputs:
        raise MergeInputError(
            "merge_ranks requires at least one rank capture or rank-core path.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
        )
    resolved: dict[int, tuple[RankEvidence, Any]] = {}
    for position, item in enumerate(inputs):
        if isinstance(item, (str, Path)):
            from .._io.bundle import load as load_bundle

            source = str(item)
            try:
                trace = load_bundle(source)
            except Exception as exc:
                raise MergeInputError(
                    f"Merge input {source} failed to load as a rank core: {exc}",
                    code=MergedErrorCode.MERGE_INPUT_INVALID,
                    source=source,
                ) from exc
        else:
            trace = item
            source = f"live[{position}]"
        evidence = extract_rank_evidence(trace, source)
        if evidence.rank in resolved:
            raise MergeInputError(
                f"Merge inputs contain global rank {evidence.rank} twice "
                f"({resolved[evidence.rank][0].source} and {source}); every rank "
                "core must come from a distinct rank of one run.",
                code=MergedErrorCode.MERGE_INPUT_INVALID,
            )
        resolved[evidence.rank] = (evidence, trace)
    return dict(sorted(resolved.items()))
