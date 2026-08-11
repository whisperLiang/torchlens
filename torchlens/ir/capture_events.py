"""Mutable capture event accumulator for one forward pass."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, NoReturn
import weakref

from .events import (
    BackwardPassEnd,
    BackwardPassStart,
    GradFnDiscovered,
    GradFnFired,
    ModuleEnterEvent,
    ModuleExitEvent,
    ModulePrepEvent,
    OpGradObserved,
    ParamGradObserved,
    OpEvent,
    OutputVersionEvent,
    PreHookProvenanceEvent,
)
from .live_index import LiveIndex
from .predicate import RecordContext
from .refs import ParamRef, ReservedLabel

if TYPE_CHECKING:
    import torch

    from .intervention import FireResult


def _clone_op_event_for_replay(event: OpEvent) -> OpEvent:
    """Return a projection copy of ``event`` with independent mutable state.

    ``OpEvent`` is a frozen dataclass, but two of its fields are live dicts:
    ``transform_config`` and ``parent_arg_positions``. A ``copy_for_replay``
    projection must not be able to mutate those dicts on the sealed source
    stream, so they are duplicated here. Every other field is immutable (or an
    intentionally shared tensor payload / opaque handle), so the clone stays
    cheap and never copies activations.

    Parameters
    ----------
    event
        Sealed source operation event.

    Returns
    -------
    OpEvent
        Event with fresh, independent ``transform_config`` and
        ``parent_arg_positions`` containers.
    """

    return replace(
        event,
        parent_arg_positions={
            domain: dict(positions) for domain, positions in event.parent_arg_positions.items()
        },
        transform_config=dict(event.transform_config),
    )


@dataclass(slots=False)
class CaptureEvents:
    """Mutable event buffer allocated once per capture."""

    op_events: list[OpEvent] = field(default_factory=list)
    module_prep_events: list[ModulePrepEvent] = field(default_factory=list)
    module_enter_events: list[ModuleEnterEvent] = field(default_factory=list)
    module_exit_events: list[ModuleExitEvent] = field(default_factory=list)
    pre_hook_events: list[PreHookProvenanceEvent] = field(default_factory=list)
    output_version_events: list[OutputVersionEvent] = field(default_factory=list)
    backward_events: list[
        BackwardPassStart
        | OpGradObserved
        | ParamGradObserved
        | BackwardPassEnd
        | GradFnDiscovered
        | GradFnFired
    ] = field(default_factory=list)
    param_refs: dict[str, ParamRef] = field(default_factory=dict)
    raw_layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=dict)
    func_call_id_counter: int = 0
    recent_events: deque[RecordContext] = field(default_factory=deque)
    backend_session: object | None = None
    live_index: LiveIndex = field(default_factory=LiveIndex)
    grad_fn_handles_by_label_raw: dict[str, Any] = field(default_factory=dict)
    backward_event_seq: int = 0
    backward_revision: int = 0

    @property
    def op_event_by_label_raw(self) -> dict[str, OpEvent]:
        """Return the label lookup derived from the shared live index.

        Returns
        -------
        dict[str, OpEvent]
            Canonical live label-to-event mapping.
        """

        return self.live_index.by_raw_label

    @op_event_by_label_raw.setter
    def op_event_by_label_raw(self, events_by_label: dict[str, OpEvent]) -> None:
        """Replace the canonical label lookup and synchronize live edges.

        Parameters
        ----------
        events_by_label
            Replacement mapping, normally derived from ``op_events`` by a
            compatibility projector.
        """

        self.live_index.by_raw_label = events_by_label
        self.live_index.labels = list(events_by_label)
        self.live_index.rebuild_edges()

    def _event_position(self, event: OpEvent) -> int | None:
        """Return one event's canonical list position without a retained index.

        Parameters
        ----------
        event
            Existing operation event to locate.

        Returns
        -------
        int | None
            Producer-order position, or ``None`` when absent.
        """

        if self.op_events:
            position = event.raw_index - self.op_events[0].raw_index
            if 0 <= position < len(self.op_events):
                candidate = self.op_events[position]
                if (
                    candidate.raw_index == event.raw_index
                    and candidate.label_raw == event.label_raw
                ):
                    return position
        return next(
            (
                index
                for index, candidate in enumerate(self.op_events)
                if candidate.raw_index == event.raw_index and candidate.label_raw == event.label_raw
            ),
            None,
        )

    def copy_for_replay(
        self,
        *,
        projected_op_events: Iterable[OpEvent] | None = None,
    ) -> "CaptureEvents":
        """Return a structural working projection for postprocess mutation.

        Later postprocess steps replace entries in ``op_events`` and
        ``op_event_by_label_raw`` in place. A projector must therefore mutate
        an independent container projection instead of the sealed capture
        lanes. This is also used when a long-lived, frozen ``Recording`` cooks
        itself into a ``Trace`` so repeated projections cannot alter its event
        stream.

        Every mutable container is duplicated into a fresh object (nested list
        values included where they are rebuilt in place). The ``OpEvent`` objects
        are re-created with independent copies of their mutable dict fields
        (``transform_config`` and ``parent_arg_positions``) so a projection can
        never mutate those dicts on the sealed source stream; the same cloned
        events back ``op_events``, ``op_event_by_label_raw``, and the projected
        ``live_index`` so the projection stays internally consistent. Tensor
        payloads and every other (immutable) event field are shared by
        reference, so the copy is cheap and does not clone activations. Scalars
        and the opaque ``backend_session`` are copied by value / reference.

        A caller that already projected independent ``OpEvent`` objects may
        provide them through ``projected_op_events``. Those events are installed
        directly while every mutable container and live-index lane is still
        copied. The caller must own independent ``transform_config`` and
        ``parent_arg_positions`` dictionaries on each supplied event.

        Parameters
        ----------
        projected_op_events
            Already-projected operation events whose mutable dictionaries are
            independent from this source stream. ``None`` clones this source's
            operation events as usual.

        Returns
        -------
        CaptureEvents
            Independent event buffer over the same underlying events.
        """

        if projected_op_events is None:
            replay_op_events = [_clone_op_event_for_replay(event) for event in self.op_events]
        else:
            replay_op_events = list(projected_op_events)
        replay_by_label = {event.label_raw: event for event in replay_op_events}
        projected_index = self.live_index.copy()
        if projected_op_events is None:
            projected_index.by_raw_label = {
                label: replay_by_label.get(label, event)
                for label, event in projected_index.by_raw_label.items()
            }
        else:
            projected_index.by_raw_label = dict(replay_by_label)
            projected_index.labels = list(replay_by_label)
            projected_index.rebuild_edges()

        return CaptureEvents(
            op_events=replay_op_events,
            module_prep_events=list(self.module_prep_events),
            module_enter_events=list(self.module_enter_events),
            module_exit_events=list(self.module_exit_events),
            pre_hook_events=list(self.pre_hook_events),
            output_version_events=list(self.output_version_events),
            backward_events=list(self.backward_events),
            param_refs=dict(self.param_refs),
            raw_layer_counter=self.raw_layer_counter,
            raw_layer_type_counter=dict(self.raw_layer_type_counter),
            func_call_id_counter=self.func_call_id_counter,
            recent_events=deque(self.recent_events),
            backend_session=self.backend_session,
            live_index=projected_index,
            grad_fn_handles_by_label_raw=dict(self.grad_fn_handles_by_label_raw),
            backward_event_seq=self.backward_event_seq,
            backward_revision=self.backward_revision,
        )

    def release_working_projection(self) -> None:
        """Release mutable projector lanes without touching the sealed source.

        Returns
        -------
        None
            Drops working-container and runtime-handle references after Step 0.
        """

        self.op_events.clear()
        self.module_prep_events.clear()
        self.module_enter_events.clear()
        self.module_exit_events.clear()
        self.pre_hook_events.clear()
        self.output_version_events.clear()
        self.live_index.clear()
        self.grad_fn_handles_by_label_raw.clear()

    def release_runtime_sidecars(self) -> None:
        """Detach payload and runtime handles while retaining structural facts.

        Operation entries are rebuilt as payload-free immutable facts, and every
        runtime-handle sidecar the buffer holds is dropped so the advertised
        release boundary really frees backend / autograd / runtime-context object
        graphs: the backend session (``backend_session``), the per-label autograd
        ``grad_fn`` handles (``grad_fn_handles_by_label_raw``, also cleared by the
        sibling :meth:`release_working_projection`), and the runtime record-context
        deque (``recent_events``). Structural event facts (op/module/prep/enter/
        exit/output-version lanes with payloads stripped) are retained.

        Returns
        -------
        None
            Replaces operation entries with payload-free immutable facts and
            detaches all runtime-handle sidecars.
        """

        structural_events: list[OpEvent] = []
        for event in self.op_events:
            tensor = replace(event.output.tensor, payload=None)
            transformed = event.output.transformed_tensor
            if transformed is not None:
                transformed = replace(transformed, payload=None)
            child_versions = tuple(
                (label, replace(child_tensor, payload=None))
                for label, child_tensor in event.output.child_versions
            )
            output = replace(
                event.output,
                tensor=tensor,
                transformed_tensor=transformed,
                child_versions=child_versions,
                activation_transform=None,
            )
            templates = event.templates
            if templates is not None:
                templates = replace(
                    templates,
                    saved_args=None,
                    saved_kwargs=None,
                    args_template=None,
                    kwargs_template=None,
                )
            structural_events.append(
                replace(
                    event,
                    output=output,
                    templates=templates,
                    source_trace=None,
                )
            )
        self.op_events = structural_events
        self.module_prep_events = [
            replace(
                event,
                forward_pre_hooks=None,
                forward_hooks=None,
                backward_pre_hooks=None,
                backward_hooks=None,
                full_backward_pre_hooks=None,
                full_backward_hooks=None,
            )
            for event in self.module_prep_events
        ]
        self.module_enter_events = [
            replace(
                event,
                forward_args=None,
                forward_kwargs=None,
                forward_args_template=None,
                forward_kwargs_template=None,
            )
            for event in self.module_enter_events
        ]
        self.pre_hook_events = [
            replace(
                event,
                inputs_before_pre_hooks=None,
                inputs_after_pre_hooks=None,
            )
            for event in self.pre_hook_events
        ]
        self.output_version_events = [
            replace(event, payload=None, transform_state=None)
            for event in self.output_version_events
        ]
        self.live_index.clear()
        self.live_index.by_raw_label = {event.label_raw: event for event in structural_events}
        self.backend_session = None
        self.grad_fn_handles_by_label_raw.clear()
        self.recent_events.clear()

    def next_backward_seq(self) -> int:
        """Return the next monotonic backward event sequence number."""

        self.backward_event_seq += 1
        return self.backward_event_seq

    def append(self, event: OpEvent) -> None:
        """Append a single operation event."""
        self.op_events.append(event)
        self.live_index.append(event)

    def append_backward(
        self,
        event: BackwardPassStart
        | OpGradObserved
        | ParamGradObserved
        | BackwardPassEnd
        | GradFnDiscovered
        | GradFnFired,
    ) -> None:
        """Append a backward sidecar event, stamping the global backward seq.

        The append path is the single writer for the backward stream, so it is
        also the single sequencing authority: every appended event of every
        kind receives the next value of one run-monotonic counter, making
        cross-kind ordering an exact recorded fact rather than an inference
        from timestamps or list positions.
        """

        object.__setattr__(event, "seq", self.next_backward_seq())
        self.backward_events.append(event)
        self.backward_revision += 1

    def note_backward_event_removal(self) -> None:
        """Advance the backward revision after a sanctioned event removal.

        Event count alone cannot distinguish an add-then-remove from an
        unchanged stream, so every mutation of ``backward_events`` must move
        the revision forward for the projection guard to stay sound.
        """

        self.backward_revision += 1

    def extend(self, events: tuple[OpEvent, ...] | list[OpEvent]) -> None:
        """Append multiple operation events in order."""
        for event in events:
            self.append(event)

    def append_output_version(self, event: OutputVersionEvent) -> None:
        """Append a parent output-version sibling event."""
        self.output_version_events.append(event)

    def reserve_label(self, layer_type: str) -> ReservedLabel:
        """Reserve the next raw label for a single output site."""
        return self.reserve_label_block(layer_type, 1)[0]

    def reserve_label_block(self, layer_type: str, n: int) -> tuple[ReservedLabel, ...]:
        """Reserve a contiguous block of raw labels for output sites."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return ()

        type_counter = self.raw_layer_type_counter.get(layer_type, 0)
        labels: list[ReservedLabel] = []
        for _ in range(n):
            self.raw_layer_counter += 1
            type_counter += 1
            label_raw = f"{layer_type}_{type_counter}_{self.raw_layer_counter}_raw"
            labels.append(
                ReservedLabel(
                    label=label_raw,
                    label_raw=label_raw,
                    raw_index=self.raw_layer_counter,
                    type_index=type_counter,
                    layer_type=layer_type,
                    site=label_raw,
                )
            )
        self.raw_layer_type_counter[layer_type] = type_counter
        return tuple(labels)


@dataclass(slots=True)
class LiveOpRecord:
    """Mutable capture-time projection for one raw op label.

    Parameters
    ----------
    event
        Capture event for this operation, if emitted.
    fields
        Mutable pre-postprocess field mapping used by live capture consumers.
    tensor_ref
        Weak reference to the live output tensor, when weak-referenceable.
    t_args
        Positional call arguments used for activation saving.
    t_kwargs
        Keyword call arguments used for activation saving.
    fire_results
        Intervention hook results recorded for this operation.
    """

    event: OpEvent | None
    fields: dict[str, Any]
    tensor_ref: "weakref.ReferenceType[torch.Tensor] | None"
    t_args: tuple[Any, ...]
    t_kwargs: dict[str, Any]
    fire_results: "tuple[FireResult, ...]" = ()


def register_live_event(trace: Any, event: OpEvent, live_record: LiveOpRecord) -> None:
    """Register an emitted operation event on a trace.

    Appends ``event`` to ``trace.capture_events`` (allocating the buffer on
    first use) and records its grad-fn handle when present. This function has no
    callers in the tree; the live hot path appends events directly through
    :meth:`CaptureEvents.append`.

    Parameters
    ----------
    trace
        Active trace receiving capture events.
    event
        Operation event emitted for the new raw label.
    live_record
        Accepted only for historical signature compatibility and intentionally
        ignored: the mutable live-record projection lane is retired, so no
        ``LiveOpRecord`` is stored. Dropping this parameter is an owner-reserved
        signature change.

    Returns
    -------
    None
        Mutates ``trace.capture_events``.
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        events = CaptureEvents()
        trace.capture_events = events
    events.append(event)
    if event.grad_fn_handle is not None:
        events.grad_fn_handles_by_label_raw[event.label_raw] = event.grad_fn_handle


def replace_op_event(trace: Any, label_raw: str, **updates: Any) -> OpEvent | None:
    """Replace one emitted operation event with updated field values.

    Parameters
    ----------
    trace
        Active trace carrying the capture event buffer.
    label_raw
        Raw label identifying the operation event.
    **updates
        Dataclass field updates to apply to the frozen event.

    Returns
    -------
    OpEvent | None
        Updated event when found, otherwise ``None``.
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        return None
    event = events.op_event_by_label_raw.get(label_raw)
    if event is None:
        return None
    updated_event = replace(event, **updates)
    index = events._event_position(event)
    if index is None:
        return updated_event
    events.op_events[index] = updated_event
    events.live_index.replace(updated_event)
    return updated_event


def live_record_for_label(trace: Any, label_raw: str) -> NoReturn:
    """Always raise: the mutable per-label live-record lane is retired.

    Capture no longer materializes a mutable :class:`LiveOpRecord` per raw
    label; capture-time consumers read the event-backed
    :class:`~torchlens.ir.live_index.LiveIndex` instead. This function is a
    retained compatibility stub with no callers in the tree and never returns a
    record. It is intentionally kept off the mutable-live-record hot path (see
    ``tests/test_capture_unification_p2.py``); removing it or its
    ``torchlens.ir`` export is an owner-reserved public-surface change.

    Parameters
    ----------
    trace
        Active trace (unused).
    label_raw
        Raw operation label included in the raised message.

    Raises
    ------
    KeyError
        Always, because no mutable live record exists for any label.
    """

    raise KeyError(
        f"{label_raw!r} has no mutable live record; use CaptureEvents.live_index instead."
    )
