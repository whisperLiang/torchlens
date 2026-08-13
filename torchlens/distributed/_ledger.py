"""Group-lifecycle ledger: per-rank evidence for cross-rank group identity.

C0 of the merge-ranks tier records, on every rank core captured under the
distributed opt-in, an ordered ledger of every observed process-group
lifecycle event: wrapped creations (with their assigned lifetime ordinal),
wrapped destroys/aborts, and restricted seeding events. Per-membership
*lineage vectors* derive from the ledger; they are the evidence the merge
engine's PRE-JOIN membership-lineage audit (:mod:`torchlens.distributed._audit`)
reads before any correlation-key joining or presence-gap derivation.

The ledger is rank-local truth, never authority about other ranks: it exists
so a later offline merge can *prove* that the group generations two ranks
captured are the same communicator lineage, or refuse structurally when that
cannot be proven. Display strings (torch's ``pg.group_name``) are recorded as
diagnostics only and never participate in identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "GroupLifecycleEvent",
    "GroupLifecycleLedger",
    "InstallEpoch",
    "LineageEntry",
    "LineageVector",
    "membership_digest_for_ranks",
]

InstallEpoch = Literal["armed_before_any_group", "seeded"]
"""Arming wall-position relative to this rank's group-creation history.

``armed_before_any_group`` makes this rank a COMPLETE WITNESS for every
membership it joins: c10d requires identical creation/destruction program
order across member ranks, so an armed member rank cannot miss a generation.
``seeded`` means group history may predate arming, so this rank's ordinals are
provable only through the merge-time audit.
"""

EventKind = Literal["create", "destroy", "seed"]

OrdinalSource = Literal["wrapped", "seeded"]


def membership_digest_for_ranks(global_ranks: Any) -> str:
    """Return the membership digest for a process group's global ranks.

    Parameters
    ----------
    global_ranks:
        Iterable of the group's member ranks in GLOBAL (world) numbering.

    Returns
    -------
    str
        Hex SHA-256 over the sorted rank tuple. Identical on every member
        rank by construction, so it is the cross-rank membership key.
    """

    ranks = sorted(int(rank) for rank in global_ranks)
    encoded = ",".join(str(rank) for rank in ranks).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GroupLifecycleEvent:
    """One observed group-lifecycle event on this rank.

    Parameters
    ----------
    event_index:
        Rank-local monotone position of this event in the ledger.
    kind:
        ``"create"`` (wrapped creation), ``"destroy"`` (wrapped destroy or
        abort, observational), or ``"seed"`` (restricted registry seeding of a
        group that predates arming).
    membership_digest:
        Membership key from :func:`membership_digest_for_ranks`.
    ordinal:
        Lifetime ordinal assigned to the generation this event concerns.
    ordinal_source:
        ``"wrapped"`` when the ordinal was assigned at wrapped creation,
        ``"seeded"`` when assigned by restricted seeding. Diagnostic on
        destroy events (echoes the generation's source).
    install_epoch:
        The rank's install-epoch record at the time of the event.
    local_creation_index:
        Diagnostic: how many groups (any membership) this rank had observed
        created/seeded before this one. ``None`` for destroy events.
    group_name:
        Diagnostic display string (torch ``pg.group_name``); never identity.
    name_scheme:
        Diagnostic: how ``group_name`` was obtained (``"pg.group_name"`` or
        ``None`` when unavailable).
    """

    event_index: int
    kind: EventKind
    membership_digest: str
    ordinal: int
    ordinal_source: OrdinalSource
    install_epoch: InstallEpoch
    local_creation_index: int | None = None
    group_name: str | None = None
    name_scheme: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for portable artifacts."""

        return {
            "event_index": self.event_index,
            "kind": self.kind,
            "membership_digest": self.membership_digest,
            "ordinal": self.ordinal,
            "ordinal_source": self.ordinal_source,
            "install_epoch": self.install_epoch,
            "local_creation_index": self.local_creation_index,
            "group_name": self.group_name,
            "name_scheme": self.name_scheme,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> GroupLifecycleEvent:
        """Rebuild an event from :meth:`to_payload` output."""

        return cls(
            event_index=int(payload["event_index"]),
            kind=payload["kind"],
            membership_digest=payload["membership_digest"],
            ordinal=int(payload["ordinal"]),
            ordinal_source=payload["ordinal_source"],
            install_epoch=payload["install_epoch"],
            local_creation_index=payload.get("local_creation_index"),
            group_name=payload.get("group_name"),
            name_scheme=payload.get("name_scheme"),
        )


@dataclass(frozen=True)
class LineageEntry:
    """One generation of a membership as this rank evidences it."""

    ordinal: int
    source: OrdinalSource
    destroyed: bool

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {"ordinal": self.ordinal, "source": self.source, "destroyed": self.destroyed}


@dataclass(frozen=True)
class LineageVector:
    """This rank's ordered generation evidence for one membership.

    Parameters
    ----------
    membership_digest:
        Membership key the vector describes.
    entries:
        Ordered ``(ordinal, source, destroyed)`` generation entries.
    install_epoch:
        The presenting rank's install-epoch record.
    """

    membership_digest: str
    entries: tuple[LineageEntry, ...]
    install_epoch: InstallEpoch

    @property
    def generations_created(self) -> int:
        """Number of generations of this membership the rank ever evidenced."""

        return len(self.entries)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "membership_digest": self.membership_digest,
            "entries": [entry.to_payload() for entry in self.entries],
            "install_epoch": self.install_epoch,
        }


class GroupLifecycleLedger:
    """Ordered per-rank record of observed group-lifecycle events.

    The ledger is append-only during a process's lifetime; lineage vectors
    are derived views. It serializes into rank cores as a portable field and
    round-trips through :meth:`to_payload` / :meth:`from_payload`.
    """

    def __init__(self, events: list[GroupLifecycleEvent] | None = None) -> None:
        self._events: list[GroupLifecycleEvent] = list(events or [])

    @property
    def events(self) -> tuple[GroupLifecycleEvent, ...]:
        """Immutable view of the recorded events in order."""

        return tuple(self._events)

    def append(self, event: GroupLifecycleEvent) -> None:
        """Append one event, enforcing the monotone ``event_index`` contract."""

        if self._events and event.event_index <= self._events[-1].event_index:
            raise ValueError(
                "GroupLifecycleLedger events must carry strictly increasing "
                f"event_index; got {event.event_index} after "
                f"{self._events[-1].event_index}"
            )
        self._events.append(event)

    def next_event_index(self) -> int:
        """Return the ``event_index`` the next appended event should carry."""

        return self._events[-1].event_index + 1 if self._events else 0

    def next_ordinal(self, membership_digest: str) -> int:
        """Return the next lifetime ordinal for a membership.

        Ordinals are ever-created and never reused: the next ordinal is the
        count of create/seed events already recorded for the membership,
        regardless of destroys.
        """

        return sum(
            1
            for event in self._events
            if event.membership_digest == membership_digest and event.kind in ("create", "seed")
        )

    def creation_count(self) -> int:
        """Return how many creations/seeds (any membership) were observed."""

        return sum(1 for event in self._events if event.kind in ("create", "seed"))

    def lineage_vectors(self) -> dict[str, LineageVector]:
        """Derive the per-membership lineage vectors the pre-join audit reads.

        Returns
        -------
        dict[str, LineageVector]
            Mapping from membership digest to that membership's ordered
            ``(ordinal, source, destroyed)`` entries. The install epoch
            stamped on the vector is the one carried by the membership's
            events (a rank has exactly one epoch; it is stamped per event so
            the ledger stays self-describing after serialization).
        """

        generations: dict[str, dict[int, dict[str, Any]]] = {}
        epochs: dict[str, InstallEpoch] = {}
        for event in self._events:
            digest = event.membership_digest
            epochs.setdefault(digest, event.install_epoch)
            rows = generations.setdefault(digest, {})
            if event.kind in ("create", "seed"):
                rows[event.ordinal] = {
                    "source": event.ordinal_source,
                    "destroyed": False,
                }
            elif event.kind == "destroy" and event.ordinal in rows:
                rows[event.ordinal]["destroyed"] = True
        vectors: dict[str, LineageVector] = {}
        for digest, rows in generations.items():
            entries = tuple(
                LineageEntry(
                    ordinal=ordinal,
                    source=rows[ordinal]["source"],
                    destroyed=rows[ordinal]["destroyed"],
                )
                for ordinal in sorted(rows)
            )
            vectors[digest] = LineageVector(
                membership_digest=digest,
                entries=entries,
                install_epoch=epochs[digest],
            )
        return vectors

    def to_payload(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable payload for portable artifacts."""

        return [event.to_payload() for event in self._events]

    @classmethod
    def from_payload(cls, payload: list[dict[str, Any]]) -> GroupLifecycleLedger:
        """Rebuild a ledger from :meth:`to_payload` output."""

        return cls([GroupLifecycleEvent.from_payload(entry) for entry in payload])
