"""Resolve user split points and create split plans."""

from __future__ import annotations

import re

from .errors import SplitSpecError
from .frontier import SplitPlan, build_frontier
from .spec import SplitSpec
from .trace_graph import TraceGraph, TraceNode


def plan_split(graph: TraceGraph, spec: SplitSpec) -> SplitPlan:
    """Create a concrete split plan from a TraceGraph and SplitSpec."""

    ordered = graph.ordered_nodes()
    if not ordered:
        raise SplitSpecError("Cannot split an empty trace graph.")
    split_index, split_label, split_id = _resolve_split_index(graph, spec.boundary)
    prefix: list[str] = []
    suffix: list[str] = []
    for index, node in enumerate(ordered):
        if node.is_output:
            continue
        if index <= split_index:
            prefix.append(node.torchlens_label)
        elif not node.is_input:
            suffix.append(node.torchlens_label)
    return build_frontier(
        graph,
        split_id=split_id,
        split_label=split_label,
        prefix_nodes=tuple(prefix),
        suffix_nodes=tuple(suffix),
    )


def _resolve_split_index(graph: TraceGraph, boundary: str) -> tuple[int, str, str]:
    ordered = graph.ordered_nodes()
    if boundary.endswith("%"):
        pct = float(boundary[:-1])
        return _percent_index(ordered, pct, boundary)
    if boundary.startswith("percent:"):
        pct = float(boundary.split(":", maxsplit=1)[1])
        return _percent_index(ordered, pct, boundary)
    match = re.match(r"^(after|before):(.+)$", boundary)
    if not match:
        raise SplitSpecError(
            "SplitSpec.boundary must be after:<target>, before:<target>, percent:N, or N%."
        )
    mode, target = match.groups()
    matches = _matching_nodes(ordered, target)
    if not matches:
        raise SplitSpecError(f"Split boundary target {target!r} did not match any trace node.")
    if len(matches) > 1:
        label_matches = [node for node in matches if node.torchlens_label == target]
        module_matches = [node for node in matches if node.module_path == target]
        if len(label_matches) == 1:
            node = label_matches[0]
        elif module_matches:
            node = module_matches[-1] if mode == "after" else module_matches[0]
        else:
            labels = ", ".join(node.torchlens_label for node in matches[:8])
            raise SplitSpecError(f"Split boundary target {target!r} is ambiguous: {labels}.")
    else:
        node = matches[0]
    index = ordered.index(node)
    if mode == "before":
        index -= 1
    if index < 0:
        raise SplitSpecError(f"Split boundary {boundary!r} leaves no prefix nodes.")
    return index, node.torchlens_label, f"{mode}:{target}"


def _percent_index(nodes: tuple[TraceNode, ...], percent: float, boundary: str) -> tuple[int, str, str]:
    if percent <= 0 or percent >= 100:
        raise SplitSpecError("Percent split must be between 0 and 100.")
    compute_nodes = [node for node in nodes if not node.is_input and not node.is_output]
    if not compute_nodes:
        raise SplitSpecError("Percent split requires at least one compute node.")
    compute_index = max(0, min(len(compute_nodes) - 1, int(len(compute_nodes) * percent / 100) - 1))
    node = compute_nodes[compute_index]
    return nodes.index(node), node.torchlens_label, boundary


def _matching_nodes(nodes: tuple[TraceNode, ...], target: str) -> list[TraceNode]:
    result: list[TraceNode] = []
    for node in nodes:
        candidates = {
            node.torchlens_label,
            node.layer.layer_label_no_pass,
            node.module_path,
            str(getattr(node.layer, "containing_module", "") or "").split(":", maxsplit=1)[0],
        }
        if target in candidates:
            result.append(node)
    return result


__all__ = ["plan_split"]
