"""Boundary frontier derivation for split replay."""

from __future__ import annotations

from dataclasses import dataclass

from .boundary import BoundaryTensorSpec
from .trace_graph import TraceGraph


BoundaryRole = str


@dataclass(frozen=True)
class SplitPlan:
    """Concrete prefix/suffix partition and boundary schema."""

    split_id: str
    split_label: str
    prefix_nodes: tuple[str, ...]
    suffix_nodes: tuple[str, ...]
    boundary_nodes: tuple[str, ...]
    boundary_specs: dict[str, BoundaryTensorSpec]


def build_frontier(
    graph: TraceGraph,
    *,
    split_id: str,
    split_label: str,
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
        for parent_label in child.parents:
            if parent_label in prefix_set and parent_label not in boundary_nodes:
                boundary_nodes.append(parent_label)

    specs = {
        label: BoundaryTensorSpec.from_trace_node(graph.get(label), role=_infer_role(graph, label))
        for label in boundary_nodes
    }
    return SplitPlan(
        split_id=split_id,
        split_label=split_label,
        prefix_nodes=prefix_nodes,
        suffix_nodes=suffix_nodes,
        boundary_nodes=tuple(boundary_nodes),
        boundary_specs=specs,
    )


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
