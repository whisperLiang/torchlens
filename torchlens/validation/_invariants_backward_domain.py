"""Backward pass domains and journal sequencing."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .invariants import MetadataInvariantError


__all__ = (
    "_check_backward_pass_domain_invariants",
    "_layer_postdates_all_backward_triggers",
    "_backward_trigger_forward_positions",
    "_backward_pass_root_forward_position",
    "_backward_pass_observed_forward_position",
    "_resolve_op_grad_event_label",
    "_check_journal_seq_invariants",
)


def _check_backward_pass_domain_invariants(
    trace: "Trace",
    name: str,
    valid_pass_indices: set[int],
) -> None:
    """Check backward-pass domain fields and root coverage.

    The precondition contract is backward-capture only: this helper runs only
    after ``grad_fn_logs`` and dense ``backward_pass_logs`` have been proven
    present. It validates projected pass records against the event-domain
    literals used by the torch backward capture path.

    Parameters
    ----------
    trace:
        Trace with materialized backward-pass projections.
    name:
        Invariant check name to use in raised errors.
    valid_pass_indices:
        Dense set of known backward pass indices.

    Raises
    ------
    MetadataInvariantError
        If a BackwardPass field is outside its recorded domain or references a
        missing pass/GradFn.
    """

    valid_triggers = {
        "autograd_backward",
        "autograd_grad",
        "backward",
        "implicit",
        "recording_backward",
        "replay",
    }
    valid_statuses = {"error", "ok"}
    global_root_ids = set(trace.backward_root_grad_fn_object_ids)
    roots_seen_by_pass: set[int] = set()
    backward_pass_logs = getattr(trace, "backward_pass_logs", {})

    for pass_index, backward_pass in backward_pass_logs.items():
        if backward_pass.trigger not in valid_triggers:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid trigger {backward_pass.trigger!r}",
            )
        if backward_pass.status not in valid_statuses:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid status {backward_pass.status!r}",
            )
        if backward_pass.save_grads_policy is not None and not isinstance(
            backward_pass.save_grads_policy,
            str,
        ):
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid save_grads_policy "
                f"{backward_pass.save_grads_policy!r}",
            )
        if backward_pass.duration is not None and float(backward_pass.duration) < 0:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has negative duration {backward_pass.duration!r}",
            )
        if backward_pass.peak_memory is not None and backward_pass.peak_memory < 0:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has negative peak_memory {backward_pass.peak_memory!r}",
            )
        coverage = backward_pass.order_attribution_coverage
        if coverage is not None and not 0.0 <= coverage <= 1.0:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid order_attribution_coverage {coverage!r}",
            )
        origin_pass = backward_pass.origin_backward_pass
        if origin_pass is not None and origin_pass not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} references missing origin_backward_pass "
                f"{origin_pass!r}",
            )

        missing_root_ids = [
            root_id
            for root_id in backward_pass.root_grad_fn_ids
            if root_id not in trace.grad_fn_logs
        ]
        if missing_root_ids:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} root_grad_fn_ids are missing from grad_fn_logs: "
                f"{missing_root_ids!r}",
            )
        roots_seen_by_pass.update(backward_pass.root_grad_fn_ids)

    if len(backward_pass_logs) == 1:
        pass_index, backward_pass = next(iter(backward_pass_logs.items()))
        if set(backward_pass.root_grad_fn_ids) != global_root_ids:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} root_grad_fn_ids do not match trace roots",
            )
    elif roots_seen_by_pass != global_root_ids:
        raise MetadataInvariantError(
            name,
            "BackwardPass root_grad_fn_ids union does not match trace roots",
        )


def _layer_postdates_all_backward_triggers(trace: "Trace", layer: "Layer | Op") -> bool:
    """Return whether a layer was created after every recorded backward trigger.

    Parameters
    ----------
    trace:
        Trace containing backward sidecar events.
    layer:
        Layer whose forward position is being checked.

    Returns
    -------
    bool
        True only when every recorded ``BackwardPassStart`` has a structural
        forward-op boundary and ``layer.step_index`` is beyond all of them.
    """

    layer_step_index = getattr(layer, "step_index", None)
    if not isinstance(layer_step_index, int):
        return False
    trigger_positions = _backward_trigger_forward_positions(trace)
    return bool(trigger_positions) and layer_step_index > max(trigger_positions)


def _backward_trigger_forward_positions(trace: "Trace") -> list[int]:
    """Return strict forward boundaries for recorded backward triggers.

    Parameters
    ----------
    trace:
        Trace containing backward sidecar events and projections.

    Returns
    -------
    list[int]
        One forward boundary per trigger with enough structural metadata. Root
        GradFn pairings refine active mid-forward markers because the root's
        forward op is the last layer guaranteed to exist when graph walking ran.
    """

    from ..ir.events import BackwardPassStart

    event_positions = {
        event.pass_index: event.forward_op_count_at_trigger
        for event in getattr(getattr(trace, "_capture_events", None), "backward_events", ())
        if isinstance(event, BackwardPassStart)
        and isinstance(event.forward_op_count_at_trigger, int)
    }
    positions: list[int] = []
    for pass_index, event_position in event_positions.items():
        structural_positions = [
            position
            for position in (
                _backward_pass_root_forward_position(trace, pass_index),
                _backward_pass_observed_forward_position(trace, pass_index),
            )
            if position is not None
        ]
        if not structural_positions:
            positions.append(event_position)
        else:
            positions.append(min(event_position, *structural_positions))
    return positions


def _backward_pass_root_forward_position(trace: "Trace", pass_index: int) -> int | None:
    """Return the highest paired forward position among a backward pass's roots.

    Parameters
    ----------
    trace:
        Trace with materialized backward projections.
    pass_index:
        One-based backward pass index.

    Returns
    -------
    int | None
        Highest root-paired forward ``step_index`` for the pass, when available.
    """

    backward_pass = getattr(trace, "backward_pass_logs", {}).get(pass_index)
    if backward_pass is None:
        return None
    root_steps = [
        step_index
        for root_id in getattr(backward_pass, "root_grad_fn_ids", ())
        if isinstance(
            (
                step_index := getattr(
                    getattr(getattr(trace, "grad_fn_logs", {}).get(root_id), "op", None),
                    "step_index",
                    None,
                )
            ),
            int,
        )
    ]
    return max(root_steps) if root_steps else None


def _backward_pass_observed_forward_position(trace: "Trace", pass_index: int) -> int | None:
    """Return the highest forward position with an observed gradient in a pass.

    Parameters
    ----------
    trace:
        Trace with backward sidecar events.
    pass_index:
        One-based backward pass index.

    Returns
    -------
    int | None
        Highest observed forward ``step_index`` for the pass, when available.
    """

    from ..ir.events import OpGradObserved

    layer_lookup = getattr(trace, "layer_dict_all_keys", {})
    observed_steps = []
    for event in getattr(getattr(trace, "_capture_events", None), "backward_events", ()):
        if not isinstance(event, OpGradObserved) or event.pass_index != pass_index:
            continue
        final_label = _resolve_op_grad_event_label(trace, event.op_label)
        layer = layer_lookup.get(final_label)
        step_index = getattr(layer, "step_index", None)
        if isinstance(step_index, int):
            observed_steps.append(step_index)
    return max(observed_steps) if observed_steps else None


def _resolve_op_grad_event_label(trace: "Trace", op_label: str) -> str:
    """Return the final lookup label for an ``OpGradObserved`` label.

    Delegates to the single implementation next to the event emitter so the
    validation-side resolution can never drift from the projection-side one.
    """

    from ..backends.torch.backward import _resolve_op_grad_event_label as _impl

    return _impl(trace, op_label)


def _check_journal_seq_invariants(trace: "Trace", name: str) -> None:
    """Check one-journal sequencing across every retained event lane.

    The event writer stamps ONE run-monotonic ``seq`` on every event of every
    kind (forward ops, module prep/enter/exit, pre-hook provenance, output
    versions, buffer writes, and the whole backward family), so a torch live
    stream must show writer-stamped (>= 1), lane-monotonic, journal-unique seq
    values. Preview backends do not yet route every lane through the writer;
    they join this check in the ports phase.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.
    name:
        Invariant check name to use in raised errors.

    Raises
    ------
    MetadataInvariantError
        If any lane holds an unstamped, reordered, or duplicated seq value.
    """

    if getattr(trace, "backend", "torch") != "torch":
        return
    capture_events = getattr(trace, "_capture_events", None)
    if capture_events is None:
        return
    lane_names = (
        "op_events",
        "module_prep_events",
        "module_enter_events",
        "module_exit_events",
        "pre_hook_events",
        "output_version_events",
        "buffer_write_events",
        "intervention_events",
        "backward_events",
    )
    seen_lane_by_seq: dict[int, str] = {}
    for lane_name in lane_names:
        previous_seq = 0
        for event in getattr(capture_events, lane_name, ()) or ():
            seq = getattr(event, "seq", None)
            if not isinstance(seq, int) or seq < 1:
                raise MetadataInvariantError(
                    name,
                    f"{lane_name} event {event!r:.120} is missing a writer-stamped seq",
                )
            if seq <= previous_seq:
                raise MetadataInvariantError(
                    name,
                    f"{lane_name} seq {seq} does not increase past {previous_seq}",
                )
            previous_seq = seq
            duplicate_lane = seen_lane_by_seq.get(seq)
            if duplicate_lane is not None:
                raise MetadataInvariantError(
                    name,
                    f"journal seq {seq} appears in both {duplicate_lane} and {lane_name}",
                )
            seen_lane_by_seq[seq] = lane_name
    if seen_lane_by_seq:
        counter = int(getattr(capture_events, "event_seq", 0) or 0)
        max_seen = max(seen_lane_by_seq)
        if max_seen > counter:
            raise MetadataInvariantError(
                name,
                f"journal seq {max_seen} exceeds the writer counter {counter}: "
                "an event bypassed the single-writer append path",
            )
