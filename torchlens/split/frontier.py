"""Boundary frontier derivation for split replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..intervention.types import ParentRef
from .boundary import BoundaryTensorSpec
from .trace_graph import TraceGraph


BoundaryRole = str


@dataclass(frozen=True)
class SplitPlan:
    """Concrete prefix/suffix partition and boundary schema."""

    split_id: str
    split_label: str
    use_live_param_sources: bool
    prefix_nodes: tuple[str, ...]
    suffix_nodes: tuple[str, ...]
    boundary_nodes: tuple[str, ...]
    boundary_specs: dict[str, BoundaryTensorSpec]


def build_frontier(
    graph: TraceGraph,
    *,
    split_id: str,
    split_label: str,
    use_live_param_sources: bool = False,
    prefix_nodes: tuple[str, ...],
    suffix_nodes: tuple[str, ...],
) -> SplitPlan:
    """Compute all live prefix tensors consumed by suffix/output nodes."""

    prefix_set = set(prefix_nodes)
    suffix_or_output = set(suffix_nodes) | set(graph.output_nodes)
    boundary_nodes: list[str] = []
    for child_label in graph.nodes:
        if child_label not in suffix_or_output:
            continue
        child = graph.get(child_label)
        for parent_label in _node_dependency_labels(graph, child):
            if parent_label in prefix_set and parent_label not in boundary_nodes:
                boundary_nodes.append(parent_label)

    specs = {
        label: BoundaryTensorSpec.from_trace_node(graph.get(label), role=_infer_role(graph, label))
        for label in boundary_nodes
    }
    return SplitPlan(
        split_id=split_id,
        split_label=split_label,
        use_live_param_sources=use_live_param_sources,
        prefix_nodes=prefix_nodes,
        suffix_nodes=suffix_nodes,
        boundary_nodes=tuple(boundary_nodes),
        boundary_specs=specs,
    )


def _node_dependency_labels(graph: TraceGraph, node: Any) -> tuple[str, ...]:
    labels: list[str] = []
    for label in node.parents:
        _append_label(graph, labels, label)
    _walk_parent_refs(graph, labels, node.args)
    _walk_parent_refs(graph, labels, node.kwargs)
    return tuple(labels)


def _walk_parent_refs(graph: TraceGraph, labels: list[str], value: Any) -> None:
    if isinstance(value, ParentRef):
        _append_label(graph, labels, value.parent_label)
        return
    if isinstance(value, dict):
        for item in value.values():
            _walk_parent_refs(graph, labels, item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _walk_parent_refs(graph, labels, item)


def _append_label(graph: TraceGraph, labels: list[str], label: str) -> None:
    final_label = str(getattr(graph.model_log, "_raw_to_final_layer_labels", {}).get(label, label))
    if final_label not in labels:
        labels.append(final_label)


def _infer_role(graph: TraceGraph, label: str) -> BoundaryRole:
    node = graph.get(label)
    children = [graph.get(child) for child in node.children if child in graph.nodes]
    if node.is_input:
        return "passthrough"
    if len(children) > 1:
        return "skip"
    if node.output_shape and len(node.output_shape) >= 4:
        return "multi_scale_feature"
    if node.dtype is not None and "int" in str(node.dtype):
        return "index"
    if node.output_shape == ():
        return "shape_value"
    return "primary"


__all__ = ["BoundaryRole", "SplitPlan", "build_frontier"]
