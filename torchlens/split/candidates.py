"""Partition analysis and a bounded topological candidate strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .ir import GraphPartition, SplitGraphIR, SplitRequest
from .planner import SplitPlan
from .program import ReplayProgram, SplitCapabilityReport

if TYPE_CHECKING:
    from .graph import SplitTraceGraph
    from .graph_profile import PartitionMetrics


@dataclass(frozen=True)
class PartitionAnalysis:
    """Validated DAG partition and programs, with no backend executables."""

    source_graph: SplitTraceGraph = field(repr=False, compare=False)
    request: SplitRequest
    plan: SplitPlan
    graph_ir: SplitGraphIR
    prefix_program: ReplayProgram
    suffix_program: ReplayProgram
    capability_report: SplitCapabilityReport
    metrics: PartitionMetrics | None = None


def topological_partitions(graph: SplitTraceGraph) -> tuple[GraphPartition, ...]:
    """Return native-call-atomic prefixes, including the empty and complete sets.

    This linear candidate strategy does not restrict manually selected DAG partitions.
    """
    selected: set[str] = set()
    result = [GraphPartition(frozenset())]
    for call in graph.replay_calls:
        selected.update(call.output_node_ids)
        result.append(GraphPartition(frozenset(selected)))
    return tuple(result)


def _partition_requirements(graph: SplitTraceGraph) -> dict[str, set[str]]:
    """Collect atomic data/effect predecessors and local write co-location groups."""
    from .graph import ReplayValueRef
    from .planner import _node_mutates, _partition_call_arguments, _torch_mutation_targets

    compute = graph.compute_nodes
    compute_ids = frozenset(n.canonical_id for n in compute)
    effects = graph._partition_effect_index
    targets = _torch_mutation_targets()
    requirements: dict[str, set[str]] = {node.canonical_id: set() for node in graph.nodes}
    for node in graph.nodes:
        requirements[node.canonical_id].update(
            graph.node_id_by_alias.get(parent, parent) for parent in node.parents
        )
        requirements[node.canonical_id].update(effects.predecessors.get(node.canonical_id, ()))
    for call in graph.replay_calls:
        for node_id in call.output_node_ids:
            requirements[node_id].update(call.output_node_ids)
    for node in compute:
        args, kwargs = _partition_call_arguments(node)
        if not _node_mutates(node, args, kwargs) or node.target not in targets.writes:
            continue
        argument = args[0] if args else kwargs.get("input")
        if isinstance(argument, ReplayValueRef) and argument.value_id in compute_ids:
            requirements[node.canonical_id].add(argument.value_id)
            requirements[argument.value_id].add(node.canonical_id)
    return requirements


def _effect_closed_partitions(
    graph: SplitTraceGraph, *, limit: int = 512
) -> tuple[GraphPartition, ...]:
    """Search bounded native-call closures of data, effects and local write groups.

    Single branch closures and unions with representative topological prefixes
    sample a linear number of seeds. This is a discovery aid, never a second
    validator; callers still submit each result to ``SplitGraph.analyze``.
    """
    if limit < 1:
        return ()
    compute = graph.compute_nodes
    compute_ids = frozenset(n.canonical_id for n in compute)
    requirements = _partition_requirements(graph)

    def closure(seeds: list[str]) -> frozenset[str]:
        """Compute a finite least fixed point over the one captured graph."""
        selected: set[str] = set()
        pending = list(seeds)
        while pending:
            node_id = pending.pop()
            if node_id not in selected:
                selected.add(node_id)
                pending.extend(requirements.get(node_id, ()))
        return frozenset(selected & compute_ids)

    indexes = sorted(range(len(compute)), key=lambda i: (abs(i - len(compute) // 2), i))
    seen: set[frozenset[str]] = set()
    result = []
    # Reserve half the bounded seed budget for subgraph unions, which a
    # single-call ancestor search cannot discover.
    indexes = indexes[: max(1, limit // 2)]
    seeds: list[list[str]] = [[compute[i].canonical_id] for i in indexes]
    seeds.extend(
        [*(node.canonical_id for node in compute[: i // 2]), compute[i].canonical_id]
        for i in indexes
    )
    for seed in seeds:
        selected = closure(seed)
        if (
            selected in seen
            or not selected
            or len(selected) == len(compute)
            or selected == frozenset(n.canonical_id for n in compute[: len(selected)])
            or not any(
                parent in selected
                for node in (*compute, *(graph.node_by_id[n] for n in graph.output_node_ids))
                if node.canonical_id not in selected
                for parent in node.parents
            )
        ):
            continue
        seen.add(selected)
        result.append(GraphPartition(selected))
        if len(result) == limit:
            break
    return tuple(result)
