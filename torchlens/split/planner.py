"""Split target resolution and live-frontier planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import floor
from typing import Literal

from ..intervention.types import CapturedArgTemplate
from .errors import SplitRequestError
from .frontier import (
    boundary_key_for_node,
    classify_boundary_role,
    make_boundary_schema,
)
from .graph import ReplayValueRef, SplitTraceGraph, SplitTraceNode
from .ir import BoundarySchema, SplitRequest

BoundaryKind = Literal["after", "before"]


@dataclass(frozen=True)
class SplitPlan:
    """Resolved split plan."""

    split_id: str
    boundary_kind: BoundaryKind
    target_node_id: str
    prefix_node_ids: frozenset[str]
    suffix_node_ids: frozenset[str]
    boundary_node_ids: tuple[str, ...]
    boundary_spec: dict[str, BoundarySchema]
    boundary_bindings: dict[str, str] = field(default_factory=dict)


def _parse_boundary(boundary: str) -> tuple[BoundaryKind | Literal["percent"], str]:
    """Parse a boundary string."""

    if boundary.startswith("after:"):
        return "after", boundary.split(":", 1)[1]
    if boundary.startswith("before:"):
        return "before", boundary.split(":", 1)[1]
    if boundary.startswith("percent:"):
        return "percent", boundary.split(":", 1)[1]
    if boundary.endswith("%"):
        return "percent", boundary[:-1]
    raise SplitRequestError(
        "Split boundary must be 'after:<target>', 'before:<target>', 'percent:<N>', or '<N>%'."
    )


def _resolve_percent_target(graph: SplitTraceGraph, percent_text: str) -> SplitTraceNode:
    """Resolve a percent target among eligible compute nodes."""

    try:
        percent = float(percent_text)
    except ValueError as exc:
        raise SplitRequestError(f"Invalid percent split {percent_text!r}.") from exc
    if not 0 < percent < 100:
        raise SplitRequestError("Percent split must be strictly between 0 and 100.")
    eligible = graph.compute_nodes
    if not eligible:
        raise SplitRequestError(
            "Cannot resolve percent split: graph has no eligible compute nodes."
        )
    index = floor((percent / 100.0) * (len(eligible) - 1))
    return eligible[index]


def _unique_match(candidates: list[SplitTraceNode], target: str) -> SplitTraceNode:
    """Return a unique target match or raise."""

    if not candidates:
        raise SplitRequestError(f"No split target matched {target!r}.")
    deduped = {candidate.canonical_id: candidate for candidate in candidates}
    if len(deduped) > 1:
        labels = ", ".join(sorted(node.label for node in deduped.values()))
        raise SplitRequestError(f"Split target {target!r} is ambiguous: {labels}.")
    return next(iter(deduped.values()))


def _targetable_candidates(candidates: list[SplitTraceNode]) -> list[SplitTraceNode]:
    """Return candidates that are valid split targets."""

    return [
        node
        for node in candidates
        if not (node.is_input or node.is_output or node.is_buffer or node.is_buffer_only_source)
    ]


def _resolve_named_target(graph: SplitTraceGraph, target: str) -> SplitTraceNode:
    """Resolve a target by exact label/raw/canonical/module, then substring."""

    for field_name in ("label", "raw_label", "canonical_id", "module_path"):
        exact = [
            node
            for node in graph.nodes
            if getattr(node, field_name, None) is not None
            and str(getattr(node, field_name)) == target
        ]
        if exact:
            targetable = _targetable_candidates(exact)
            return _unique_match(targetable or exact, target)
    substring = [
        node
        for node in graph.nodes
        if any(
            value is not None and target in str(value)
            for value in (node.label, node.raw_label, node.canonical_id, node.module_path)
        )
    ]
    targetable = _targetable_candidates(substring)
    return _unique_match(targetable or substring, target)


def _reject_invalid_target(node: SplitTraceNode) -> None:
    """Reject source/sink split targets."""

    if node.is_input:
        raise SplitRequestError(f"Split target {node.label!r} is an input node.")
    if node.is_output:
        raise SplitRequestError(f"Split target {node.label!r} is an output node.")
    if node.is_buffer:
        raise SplitRequestError(f"Split target {node.label!r} is a buffer node.")
    if node.is_buffer_only_source:
        raise SplitRequestError(f"Split target {node.label!r} is a buffer-only source node.")


def _call_group_ids(graph: SplitTraceGraph, node: SplitTraceNode) -> set[str]:
    """Return all nodes in the same function-call group as ``node``."""

    call = graph.replay_call_by_output_id.get(node.canonical_id)
    return {node.canonical_id} if call is None else set(call.output_node_ids)


def _prefix_suffix_sets(
    graph: SplitTraceGraph,
    target: SplitTraceNode,
    *,
    boundary_kind: BoundaryKind,
) -> tuple[frozenset[str], frozenset[str]]:
    """Compute prefix and suffix node ID sets."""

    order = graph.order_by_id
    target_order = order[target.canonical_id]
    group_ids = _call_group_ids(graph, target)
    if boundary_kind == "after":
        prefix = {
            node.canonical_id
            for node in graph.nodes
            if order[node.canonical_id] <= target_order or node.canonical_id in group_ids
        }
    else:
        prefix = {
            node.canonical_id
            for node in graph.nodes
            if order[node.canonical_id] < target_order and node.canonical_id not in group_ids
        }
    suffix = {node.canonical_id for node in graph.nodes} - prefix
    return frozenset(prefix), frozenset(suffix)


def _frontier_node_ids(
    graph: SplitTraceGraph,
    *,
    prefix_node_ids: frozenset[str],
    suffix_node_ids: frozenset[str],
) -> tuple[str, ...]:
    """Return live crossing dependencies from prefix to suffix/output."""

    order = graph.order_by_id
    frontier: set[str] = set()
    for node in graph.nodes:
        if node.canonical_id not in suffix_node_ids and not node.is_output:
            continue
        # A branched model can finish one output before the requested split
        # point while another output branch continues into the suffix.  That
        # completed output has no suffix child to expose it as a normal
        # crossing parent, so carry the output node itself through the
        # boundary ABI.
        if node.is_output and node.canonical_id in prefix_node_ids:
            frontier.add(node.canonical_id)
        for parent in node.parents:
            parent_node = graph.node_for_label(parent)
            if parent_node is not None and parent_node.canonical_id in prefix_node_ids:
                frontier.add(parent_node.canonical_id)
    for node_id in graph.input_node_ids:
        if node_id in suffix_node_ids:
            frontier.add(node_id)
    if not frontier and not suffix_node_ids and not graph.output_node_ids:
        terminal_prefix_nodes = [
            node
            for node in graph.nodes
            if node.canonical_id in prefix_node_ids
            and not (node.is_input or node.is_buffer or node.is_buffer_only_source)
        ]
        if terminal_prefix_nodes:
            frontier.add(terminal_prefix_nodes[-1].canonical_id)
    return tuple(sorted(frontier, key=lambda node_id: order[node_id]))


def _boundary_specs(
    graph: SplitTraceGraph,
    *,
    spec: SplitRequest,
    target: SplitTraceNode,
    boundary_kind: BoundaryKind,
    boundary_node_ids: tuple[str, ...],
) -> dict[str, BoundarySchema]:
    """Build public boundary specs for crossing node IDs."""

    node_by_id = graph.node_by_id
    direct_parent_ids = {
        parent_node.canonical_id
        for parent in target.parents
        if (parent_node := graph.node_for_label(parent)) is not None
    }
    spatial_shapes: set[tuple[int, ...]] = set()
    for node_id in boundary_node_ids:
        output_shape = node_by_id[node_id].output_shape
        if output_shape is not None and len(output_shape) >= 4:
            spatial_shapes.add(tuple(output_shape[-2:]))
    specs: dict[str, BoundarySchema] = {}
    for node_id in boundary_node_ids:
        node = node_by_id[node_id]
        key = boundary_key_for_node(node.canonical_id, node.output_container_path)
        role = classify_boundary_role(
            node_id=node.canonical_id,
            target_node_id=target.canonical_id,
            boundary_kind=boundary_kind,
            direct_target_parent_ids=direct_parent_ids,
            is_source=node.is_input or node.is_buffer,
            dtype=node.dtype,
            shape=node.output_shape,
            spatial_shapes=spatial_shapes,
        )
        specs[key] = make_boundary_schema(
            key,
            node=node,
            role=role,
            device_policy=spec.device_policy,
        )
    return specs


def _boundary_bindings(
    graph: SplitTraceGraph,
    boundary_node_ids: tuple[str, ...],
) -> dict[str, str]:
    """Bind public boundary keys to canonical graph value IDs."""

    node_by_id = graph.node_by_id
    bindings = {
        boundary_key_for_node(node_id, node_by_id[node_id].output_container_path): node_id
        for node_id in boundary_node_ids
    }
    if len(bindings) != len(boundary_node_ids):
        raise SplitRequestError("Split boundary contains duplicate canonical value bindings.")
    return bindings


def _validate_replay_dependencies(graph: SplitTraceGraph) -> None:
    """Require every canonical replay argument dependency to be a graph edge."""

    for node in graph.compute_nodes:
        parent_ids = {
            parent.canonical_id
            for label in node.parents
            if (parent := graph.node_for_label(label)) is not None
        }
        template_ids = {
            ref.value_id
            for component in (node.args_template, node.kwargs_template)
            for ref in _walk_replay_value_refs(component)
        }
        missing = template_ids - parent_ids
        if missing:
            raise SplitRequestError(
                f"Replay node {node.canonical_id!r} has untracked canonical dependencies "
                f"{tuple(sorted(missing))!r}."
            )


def _walk_replay_value_refs(component: object) -> tuple[ReplayValueRef, ...]:
    """Return replay value references from one nested template component."""

    if isinstance(component, ReplayValueRef):
        return (component,)
    if isinstance(component, CapturedArgTemplate):
        return tuple(
            ref
            for item in (*component.args, *(value for _key, value in component.kwargs))
            for ref in _walk_replay_value_refs(item)
        )
    if isinstance(component, (tuple, list)):
        return tuple(ref for item in component for ref in _walk_replay_value_refs(item))
    if isinstance(component, dict):
        return tuple(ref for item in component.values() for ref in _walk_replay_value_refs(item))
    return ()


def _split_id(graph: SplitTraceGraph, spec: SplitRequest, target: SplitTraceNode) -> str:
    """Compute a stable split identifier."""

    payload = repr(
        (
            graph.backend,
            graph.graph_shape_hash,
            spec.boundary,
            spec.batch_symbol,
            spec.dynamic_batch,
            spec.validation,
            spec.features.training,
            target.canonical_id,
        )
    ).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def plan_split(graph: SplitTraceGraph, spec: SplitRequest) -> SplitPlan:
    """Resolve a :class:`SplitRequest` into executable split plan."""

    _validate_replay_dependencies(graph)
    parsed_kind, target_text = _parse_boundary(spec.boundary)
    if parsed_kind == "percent":
        boundary_kind: BoundaryKind = "after"
        target = _resolve_percent_target(graph, target_text)
    else:
        boundary_kind = parsed_kind
        target = _resolve_named_target(graph, target_text)
    _reject_invalid_target(target)
    prefix_node_ids, suffix_node_ids = _prefix_suffix_sets(
        graph,
        target,
        boundary_kind=boundary_kind,
    )
    for call in graph.replay_calls:
        placements = {node_id in prefix_node_ids for node_id in call.output_node_ids}
        if len(placements) != 1:
            raise SplitRequestError(
                f"Replay call {call.call_id!r} has outputs in different segments."
            )
    boundary_node_ids = _frontier_node_ids(
        graph,
        prefix_node_ids=prefix_node_ids,
        suffix_node_ids=suffix_node_ids,
    )
    boundary_spec = _boundary_specs(
        graph,
        spec=spec,
        target=target,
        boundary_kind=boundary_kind,
        boundary_node_ids=boundary_node_ids,
    )
    boundary_bindings = _boundary_bindings(graph, boundary_node_ids)
    if boundary_spec.keys() != boundary_bindings.keys():
        raise SplitRequestError("Split boundary schema and canonical bindings disagree.")
    return SplitPlan(
        split_id=_split_id(graph, spec, target),
        boundary_kind=boundary_kind,
        target_node_id=target.canonical_id,
        prefix_node_ids=prefix_node_ids,
        suffix_node_ids=suffix_node_ids,
        boundary_node_ids=boundary_node_ids,
        boundary_spec=boundary_spec,
        boundary_bindings=boundary_bindings,
    )


__all__ = ["BoundaryKind", "SplitPlan", "plan_split"]
