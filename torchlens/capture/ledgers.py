"""Derived fact, decision, and payload views over one capture event spine."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

from ..ir.events import OpEvent, OutputRef


@dataclass(frozen=True, slots=True, order=True)
class EventId:
    """Stable identity for one operation event within a capture session.

    Parameters
    ----------
    raw_index
        Monotonic raw operation index reserved by the producer.
    label_raw
        Producer-stable raw label for the operation output.
    """

    raw_index: int
    label_raw: str

    @classmethod
    def from_event(cls, event: OpEvent) -> "EventId":
        """Build the stable session identity for an emitted operation event.

        Parameters
        ----------
        event
            Immutable operation event emitted by an existing producer.

        Returns
        -------
        EventId
            Identity keyed by the producer's existing raw reservation.
        """

        return cls(raw_index=event.raw_index, label_raw=event.label_raw)


@dataclass(frozen=True, slots=True)
class EventFact:
    """One immutable event fact stored in the journal.

    Parameters
    ----------
    event_id
        Stable event identity.
    event
        Existing frozen backend event; the adapter does not transform it.
    """

    event_id: EventId
    event: OpEvent


def _event_for_id(events: Sequence[OpEvent], event_id: EventId) -> OpEvent:
    """Resolve an event from a dense producer-ordered spine.

    Parameters
    ----------
    events
        Canonical producer-ordered event spine.
    event_id
        Stable event identity to resolve.

    Returns
    -------
    OpEvent
        Event matching ``event_id``.

    Raises
    ------
    KeyError
        If the canonical spine does not contain ``event_id``.
    """

    if events:
        index = event_id.raw_index - events[0].raw_index
        if 0 <= index < len(events):
            candidate = events[index]
            if (
                candidate.raw_index == event_id.raw_index
                and candidate.label_raw == event_id.label_raw
            ):
                return candidate
    for event in events:
        if event.raw_index == event_id.raw_index and event.label_raw == event_id.label_raw:
            return event
    raise KeyError(event_id)


class EventFactSequence(Sequence[EventFact]):
    """Immutable fact projection over a sealed canonical event tuple."""

    def __init__(self, events: tuple[OpEvent, ...]) -> None:
        """Bind a sealed event tuple without copying per-event records.

        Parameters
        ----------
        events
            Canonical sealed events in producer order.
        """

        self._events = events

    @overload
    def __getitem__(self, index: int) -> EventFact: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[EventFact]: ...

    def __getitem__(self, index: int | slice) -> EventFact | Sequence[EventFact]:
        """Return derived facts for one position or slice.

        Parameters
        ----------
        index
            Integer position or slice in producer order.

        Returns
        -------
        EventFact | Sequence[EventFact]
            Derived immutable fact or tuple of facts.
        """

        if isinstance(index, slice):
            return tuple(
                EventFact(event_id=EventId.from_event(event), event=event)
                for event in self._events[index]
            )
        event = self._events[index]
        return EventFact(event_id=EventId.from_event(event), event=event)

    def __len__(self) -> int:
        """Return the number of canonical events."""

        return len(self._events)


class _DerivedEventMapping(Mapping[EventId, Any]):
    """Base mapping that derives sidecar values from canonical events."""

    def __init__(self, events: Sequence[OpEvent]) -> None:
        """Bind a canonical event sequence.

        Parameters
        ----------
        events
            Producer-ordered operation events.
        """

        self._events = events

    def __iter__(self) -> Iterator[EventId]:
        """Yield stable event identities in producer order."""

        return (EventId.from_event(event) for event in self._events)

    def __len__(self) -> int:
        """Return the number of canonical events."""

        return len(self._events)


class EventJournal:
    """Append-only journal of immutable operation facts.

    The mutable journal owns only indexing and ordering.  Its public snapshots
    expose frozen :class:`EventFact` objects, so a later projector cannot alter
    producer facts through this adapter.
    """

    def __init__(self) -> None:
        """Initialize an empty journal."""

        self._events: list[OpEvent] = []
        self._owns_events = True

    def bind(self, events: list[OpEvent]) -> None:
        """Use ``events`` as the journal's canonical mutable spine.

        Parameters
        ----------
        events
            ``CaptureEvents.op_events`` list owned by the active run.
        """

        self._events = events
        self._owns_events = False

    def append(self, event: OpEvent) -> EventId:
        """Append an immutable producer fact and return its stable identity.

        Parameters
        ----------
        event
            Existing event to journal without copying or enriching it.

        Returns
        -------
        EventId
            Stable key for all event sidecars.

        Raises
        ------
        ValueError
            If the producer attempts to reuse a stable event identity.
        """

        event_id = EventId.from_event(event)
        if any(
            existing.raw_index == event.raw_index and existing.label_raw == event.label_raw
            for existing in self._events
        ):
            raise ValueError(f"Duplicate stable capture event id: {event_id!r}")
        self._events.append(event)
        return event_id

    def replace(self, event: OpEvent) -> EventId:
        """Replace a fact with its immutable producer-updated event.

        Parameters
        ----------
        event
            Updated frozen event with the same stable identity.

        Returns
        -------
        EventId
            Stable identity retained by the replacement.

        Raises
        ------
        KeyError
            If no previously journaled fact owns the event identity.
        """

        event_id = EventId.from_event(event)
        for index, existing in enumerate(self._events):
            if existing.raw_index == event.raw_index and existing.label_raw == event.label_raw:
                self._events[index] = event
                return event_id
        raise KeyError(event_id)

    def clear(self) -> None:
        """Release all journaled event references.

        Returns
        -------
        None
            Removes the session-local mirror once legacy forward cleanup has
            reached its historical release point.
        """

        if self._owns_events:
            self._events.clear()
        else:
            self._events = []
            self._owns_events = True

    @property
    def facts(self) -> tuple[EventFact, ...]:
        """Return journaled facts in producer order."""

        return tuple(
            EventFact(event_id=EventId.from_event(event), event=event) for event in self._events
        )

    @property
    def by_id(self) -> Mapping[EventId, EventFact]:
        """Return a read-only stable-id lookup of journaled facts."""

        return _EventFactMapping(self._events)

    @property
    def events(self) -> Sequence[OpEvent]:
        """Return the canonical event sequence without per-fact allocation."""

        return self._events


class _EventFactMapping(_DerivedEventMapping):
    """Stable-id fact lookup derived from canonical events."""

    def __getitem__(self, event_id: EventId) -> EventFact:
        """Return one derived immutable event fact.

        Parameters
        ----------
        event_id
            Stable event identity.

        Returns
        -------
        EventFact
            Derived event fact.
        """

        event = _event_for_id(self._events, event_id)
        return EventFact(event_id=event_id, event=event)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Selection and intervention decisions for one operation event.

    Parameters
    ----------
    predicate_matched
        Existing producer predicate decision.
    intervention_fired
        Whether an intervention fired.
    intervention_replaced
        Whether an intervention returned a replacement output.
    fire_results
        Existing intervention sidecar facts.
    """

    predicate_matched: bool
    intervention_fired: bool
    intervention_replaced: bool
    fire_results: tuple[Any, ...]


class DecisionLedger:
    """Mutable sidecar ledger for event-local selection decisions."""

    def __init__(self) -> None:
        """Initialize an empty decision ledger."""

        self._journal: EventJournal | None = None

    def bind(self, journal: EventJournal) -> None:
        """Bind decisions to the journal's canonical event spine.

        Parameters
        ----------
        journal
            Event journal sharing the active ``CaptureEvents`` spine.
        """

        self._journal = journal

    def append_from_event(self, event_id: EventId, event: OpEvent) -> None:
        """Record the existing event decision fields without re-evaluating them.

        Parameters
        ----------
        event_id
            Stable identity for ``event``.
        event
            Existing producer event.
        """

        if self._journal is None:
            raise RuntimeError("DecisionLedger must be bound before recording decisions.")
        _event_for_id(self._journal.events, event_id)

    def clear(self) -> None:
        """Release all decision sidecars.

        Returns
        -------
        None
            Removes session-local references after the active run ends.
        """

        self._journal = None

    @property
    def records(self) -> Mapping[EventId, DecisionRecord]:
        """Return a read-only stable-id lookup of decisions."""

        events: Sequence[OpEvent] = () if self._journal is None else self._journal.events
        return DecisionMapping(events)


class DecisionMapping(_DerivedEventMapping):
    """Selection decision mapping derived from canonical event fields."""

    def __getitem__(self, event_id: EventId) -> DecisionRecord:
        """Return the decision fields for one event.

        Parameters
        ----------
        event_id
            Stable event identity.

        Returns
        -------
        DecisionRecord
            Immutable decision projection.
        """

        event = _event_for_id(self._events, event_id)
        return DecisionRecord(
            predicate_matched=event.predicate_matched,
            intervention_fired=event.intervention_fired,
            intervention_replaced=event.intervention_replaced,
            fire_results=event.fire_results,
        )


@dataclass(frozen=True, slots=True)
class PayloadRecord:
    """Payload lease sidecar for one event.

    Parameters
    ----------
    output
        Existing event output reference.  The adapter creates no tensor copy,
        storage write, or additional payload retention.
    """

    output: OutputRef


class PayloadLedger:
    """Mutable sidecar ledger for producer-owned payload leases."""

    def __init__(self) -> None:
        """Initialize an empty payload ledger."""

        self._journal: EventJournal | None = None

    def bind(self, journal: EventJournal) -> None:
        """Bind payload leases to the journal's canonical event spine.

        Parameters
        ----------
        journal
            Event journal sharing the active ``CaptureEvents`` spine.
        """

        self._journal = journal

    def append_from_event(self, event_id: EventId, event: OpEvent) -> None:
        """Reference an existing event payload without retaining a new copy.

        Parameters
        ----------
        event_id
            Stable identity for ``event``.
        event
            Existing producer event.
        """

        if self._journal is None:
            raise RuntimeError("PayloadLedger must be bound before recording payloads.")
        _event_for_id(self._journal.events, event_id)

    def clear(self) -> None:
        """Release all payload leases.

        Returns
        -------
        None
            Drops activation references at the legacy forward-memory release
            point rather than extending their lifetime through the session.
        """

        self._journal = None

    @property
    def records(self) -> Mapping[EventId, PayloadRecord]:
        """Return a read-only stable-id lookup of payload sidecars."""

        events: Sequence[OpEvent] = () if self._journal is None else self._journal.events
        return PayloadMapping(events)


class PayloadMapping(_DerivedEventMapping):
    """Payload lease mapping derived from canonical event output fields."""

    def __getitem__(self, event_id: EventId) -> PayloadRecord:
        """Return the output payload lease for one event.

        Parameters
        ----------
        event_id
            Stable event identity.

        Returns
        -------
        PayloadRecord
            Immutable payload projection.
        """

        return PayloadRecord(output=_event_for_id(self._events, event_id).output)
