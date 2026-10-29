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
