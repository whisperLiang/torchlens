"""Certified mutable buffer operands and publication after device transport."""

from __future__ import annotations

from typing import Any

from .._state import pause_logging
from .graph import ReplayValueRef
from .planner import _partition_call_arguments
from .state import SegmentState


def _mutable_boundary_buffers(runtime: Any) -> frozenset[str]:
    """Find state sources written by suffix operations already certified by the planner."""
    graph, plan = runtime.trace_graph, runtime.plan
    written_sources: set[str] = set()
    for node_id in plan.ordered_mutation_node_ids & plan.suffix_node_ids:
        node = graph.node_by_id[node_id]
        args, kwargs = _partition_call_arguments(node)
        if "batchnorm" in node.op_type.replace("_", ""):
            # Functional and native BatchNorm have different running-state
            # positions. Affine weights and read-only operands are not writes.
            positions = (
                (1, 2)
                if getattr(node.target, "__module__", "") == "torch.nn.functional"
                else (3, 4)
            )
            operands = tuple(
                kwargs.get(name, args[index] if len(args) > index else None)
                for name, index in zip(("running_mean", "running_var"), positions, strict=True)
            )
        else:
            output = kwargs.get("out")
            operands = (output if output is not None else args[0] if args else kwargs.get("input"),)
        written_sources.update(
            graph.node_id_by_alias.get(value.value_id, value.value_id)
            for value in operands
            if isinstance(value, ReplayValueRef)
        )
    return frozenset(
        key
        for key, node_id in plan.boundary_bindings.items()
        if node_id in written_sources and graph.node_by_id[node_id].is_buffer
    )


def publish_transported_buffers(runtime: Any, boundary: Any) -> None:
    """Publish certified device-local writes to all effective buffer bindings.

    New owned snapshots leave any original storage saved by prefix/suffix
    autograd untouched. Stable source identities still drive later recuts.
    """
    if not runtime.placement.is_heterogeneous:
        return
    keys = _mutable_boundary_buffers(runtime)
    if not keys:
        return
    updates: dict[int, Any] = {}
    for key in keys:
        node = runtime.trace_graph.node_by_id[runtime.plan.boundary_bindings[key]]
        ref = node.buffer_refs[0]
        updates[getattr(ref, "source_id", id(ref))] = boundary.tensors[key]
    with pause_logging():
        # Bind declared occurrences before staging, so a later ownership query
        # cannot lazily resurrect a stale replica from the original model.
        runtime._segment_state_snapshots()
        states: dict[int, SegmentState] = {}
        for name in ("prefix", "training_prefix", "suffix"):
            state = getattr(getattr(runtime.segments, name), "_state", None)
            if isinstance(state, SegmentState):
                states[id(state)] = state
        staged = [(state, state._stage_buffer_updates(updates)) for state in states.values()]
        for state, entries in staged:
            state._entries.update(entries)
