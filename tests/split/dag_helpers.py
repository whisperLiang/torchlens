"""External exhaustive test algorithms over the public native ComputeDAG projection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from heapq import heappop, heappush
from itertools import count, islice

import networkx as nx

from torchlens.split import ComputeDAG, GraphPartition, SplitGraph
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.graph import SplitTraceGraph
from torchlens.split.graph_profile import TensorCostFeatures, partition_metrics


def call_ids(
    graph: SplitGraph, *, modules: frozenset[str] = frozenset(), ops: frozenset[str] = frozenset()
) -> frozenset[str]:
    """Query native calls by exact module membership or operation type, without ordering."""
    return frozenset(
        call.call_id
        for call in graph.compute_dag().nodes
        if (node := graph.trace_graph.node_by_id[call.output_node_ids[0]]).module_path in modules
        or node.op_type in ops
    )


def _closures(dag: ComputeDAG) -> tuple[list[int], list[int]]:
    """Return assignment implications, including effects and equal-stage groups."""
    positions = {call.call_id: i for i, call in enumerate(dag.nodes)}
    required = [1 << i for i in range(len(positions))]
    for source, target in {*(edge[:2] for edge in dag.data_edges), *dag.effect_edges}:
        required[positions[target]] |= 1 << positions[source]
    for group in dag.must_colocate:
        mask = sum(1 << positions[call_id] for call_id in group)
        for call_id in group:
            required[positions[call_id]] |= mask
    # Bitset transitive closure also handles groups that span dependency paths.
    # These are assignment implications; the public directed DAG stays untouched.
    for i in range(len(required)):
        for j in range(len(required)):
            if required[j] & (1 << i):
                required[j] |= required[i]
    excluded = [0] * len(required)
    for i, mask in enumerate(required):
        for j in range(len(required)):
            if mask & (1 << j):
                excluded[j] |= 1 << i
    return required, excluded


def count_dag_partitions(dag: ComputeDAG) -> int:
    """Count every dependency/effect/colocation-compatible assignment exactly."""
    required, excluded = _closures(dag)
    full = (1 << len(dag.nodes)) - 1
    counts = {0: 1}
    pending = [full]
    while pending:
        remaining = pending[-1]
        if remaining in counts:
            pending.pop()
            continue
        index = remaining.bit_length() - 1
        children = (remaining & ~required[index], remaining & ~excluded[index])
        if all(child in counts for child in children):
            counts[remaining] = sum(counts[child] for child in children)
            pending.pop()
        else:
            pending.extend(child for child in set(children) if child not in counts)
    return counts[full]


def close_call_ids(dag: ComputeDAG, call_ids: frozenset[str]) -> frozenset[str]:
    """Close an external call selection under the DAG's assignment implications."""
    required, _ = _closures(dag)
    positions = {call.call_id: i for i, call in enumerate(dag.nodes)}
    mask = 0
    for call_id in call_ids:
        mask |= required[positions[call_id]]
    return frozenset(call.call_id for i, call in enumerate(dag.nodes) if mask & (1 << i))


def iter_dag_partitions(dag: ComputeDAG) -> Iterator[GraphPartition]:
    """Enumerate every compatible call assignment once; native analyze remains required."""
    required, excluded = _closures(dag)
    full = (1 << len(dag.nodes)) - 1
    pending = [(0, 0)]
    while pending:
        prefix, suffix = pending.pop()
        remaining = full & ~(prefix | suffix)
        if not remaining:
            yield dag.to_partition(
                call.call_id for i, call in enumerate(dag.nodes) if prefix & (1 << i)
            )
            continue
        index = remaining.bit_length() - 1
        pending.append((prefix, suffix | excluded[index]))
        pending.append((prefix | required[index], suffix))


def _source_constraints(
    network: nx.DiGraph,
    native: SplitTraceGraph,
    owners: dict[str, tuple[str, str]],
    side: Callable[[str], tuple[str, str]],
    infinity: int,
) -> None:
    """Keep source provenance fixed and buffer observations beside their producers."""
    for node in native.nodes:
        value, variable = node.canonical_id, side(node.canonical_id)
        network.add_node(variable)
        if value in owners:
            continue
        if node.is_input or ((node.is_buffer or node.is_buffer_only_source) and not node.parents):
            network.add_edge(("fixed", "prefix"), variable, capacity=infinity)
        elif node.is_buffer and node.parents:
            for parent in node.parents:
                network.add_edge(variable, side(parent), capacity=infinity)
                network.add_edge(side(parent), variable, capacity=infinity)
        else:
            network.add_edge(variable, ("fixed", "suffix"), capacity=infinity)


def _frontier_costs(
    network: nx.DiGraph,
    tensors: tuple[TensorCostFeatures, ...],
    native: SplitTraceGraph,
    side: Callable[[str], tuple[str, str]],
    infinity: int,
) -> None:
    """Charge a tensor once if any suffix consumer or returned output needs it."""
    for tensor in tensors:
        is_output = native.node_by_id[tensor.value_id].is_output
        if not tensor.consumer_ids and not is_output:
            continue
        producer, auxiliary = side(tensor.value_id), ("value", tensor.value_id)
        network.add_edge(producer, auxiliary, capacity=tensor.bytes)
        if is_output:
            # tinygrad returns compute values directly, without output markers.
            network.add_edge(auxiliary, ("fixed", "suffix"), capacity=infinity)
        for consumer in tensor.consumer_ids:
            network.add_edge(auxiliary, side(consumer), capacity=infinity)
            network.add_edge(side(consumer), producer, capacity=infinity)


def _transfer_network(graph: SplitGraph, dag: ComputeDAG) -> tuple[nx.DiGraph, int]:
    """Encode native costs and DAG implications as an exact integer cut problem."""
    native, tensors = graph.trace_graph, graph.graph_profile().tensors
    if any(
        tensor.bytes is None
        for tensor in tensors
        if tensor.consumer_ids or native.node_by_id[tensor.value_id].is_output
    ):
        raise ValueError("Cannot rank boundary sizes with unknown native tensor costs")
    infinity = sum(tensor.bytes or 0 for tensor in tensors) + 1
    network = nx.DiGraph()
    network.add_nodes_from((("fixed", "prefix"), ("fixed", "suffix")))
    owners = {value: ("call", call.call_id) for call in dag.nodes for value in call.output_node_ids}

    def side(value: str) -> tuple[str, str]:
        """Keep native calls atomic without losing non-compute provenance."""
        return owners.get(value, ("provenance", value))

    _source_constraints(network, native, owners, side, infinity)
    _frontier_costs(network, tensors, native, side, infinity)
    for first, second in dag.effect_edges:
        network.add_edge(("call", second), ("call", first), capacity=infinity)
    for group in dag.must_colocate:
        members = sorted(group)
        for member in members[1:]:
            network.add_edge(("call", members[0]), ("call", member), capacity=infinity)
            network.add_edge(("call", member), ("call", members[0]), capacity=infinity)
    return network, infinity


def _minimum_region(
    network: nx.DiGraph, dag: ComputeDAG, infinity: int, prefix: int, suffix: int
) -> tuple[int, int] | None:
    """Solve one fixed-assignment region; infinity means the region is impossible."""
    constrained = network.copy()
    for i, call in enumerate(dag.nodes):
        if prefix & (1 << i):
            constrained.add_edge(("fixed", "prefix"), ("call", call.call_id), capacity=infinity)
        if suffix & (1 << i):
            constrained.add_edge(("call", call.call_id), ("fixed", "suffix"), capacity=infinity)
    cost, (chosen, _) = nx.minimum_cut(
        constrained,
        ("fixed", "prefix"),
        ("fixed", "suffix"),
        flow_func=nx.algorithms.flow.boykov_kolmogorov,
    )
    if cost >= infinity:
        return None
    winner = sum(1 << i for i, call in enumerate(dag.nodes) if ("call", call.call_id) in chosen)
    return cost, winner


def _region_children(
    winner: int, prefix: int, suffix: int, required: list[int], excluded: list[int]
) -> Iterator[tuple[int, int]]:
    """A first differing free call uniquely identifies each disjoint child region."""
    for i in range(len(required)):
        bit = 1 << i
        if (prefix | suffix) & bit:
            continue
        if winner & bit:
            child = prefix, suffix | excluded[i]
            prefix |= required[i]
        else:
            child = prefix | required[i], suffix
            suffix |= excluded[i]
        if not child[0] & child[1]:
            yield child


def _ranked_assignments(graph: SplitGraph, dag: ComputeDAG) -> Iterator[tuple[int, GraphPartition]]:
    """Lazily solve Lawler regions in cost order, emitting solved ties first."""
    network, infinity = _transfer_network(graph, dag)
    required, excluded = _closures(dag)
    serial = count()
    # Unsolved regions inherit the parent's rigorous minimum-cost lower bound.
    pending = [(0, 1, next(serial), -1, 0, 0)]
    while pending:
        cost, unsolved, _serial, winner, prefix, suffix = heappop(pending)
        if unsolved:
            minimum = _minimum_region(network, dag, infinity, prefix, suffix)
            if minimum is not None:
                cost, winner = minimum
                heappush(pending, (cost, 0, next(serial), winner, prefix, suffix))
            continue
        yield (
            cost,
            dag.to_partition(call.call_id for i, call in enumerate(dag.nodes) if winner & (1 << i)),
        )
        for child_prefix, child_suffix in _region_children(
            winner, prefix, suffix, required, excluded
        ):
            heappush(pending, (cost, 1, next(serial), -1, child_prefix, child_suffix))


def _native_costs(
    graph: SplitGraph, candidates: Iterable[tuple[int | None, GraphPartition]]
) -> Iterator[tuple[int | None, GraphPartition]]:
    """Retain native-legal assignments and corroborate every ranked network cost."""
    profile = graph.graph_profile()
    for expected, partition in candidates:
        try:
            analysis = graph.analyze(partition)
        except SplitUnsupportedError as exc:
            if not exc.context or not exc.context.reason.startswith("partition_"):
                raise
            continue
        actual = partition_metrics(profile, analysis.plan, training=False).forward_boundary_bytes
        if expected is not None:
            assert actual == expected, (actual, expected, partition)
        yield actual, partition


def smallest_transfer_partitions(graph: SplitGraph, *, limit: int = 100) -> list[GraphPartition]:
    """Select exact smallest transfers, or every native-legal assignment when fewer.

    This external test algorithm uses the existing capture-batch profile, including
    input, output and buffer frontier values. Every assignment is analyzed by the
    native planner; candidate enumeration is never a production API.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    dag = graph.compute_dag()
    exhaustive = list(islice(iter_dag_partitions(dag), limit + 1))
    if len(exhaustive) <= limit:
        complete = list(_native_costs(graph, ((None, partition) for partition in exhaustive)))
        if all(size is not None for size, _ in complete):
            complete.sort(key=lambda row: row[0])
        return [partition for _, partition in complete]
    return [
        partition
        for _, partition in islice(_native_costs(graph, _ranked_assignments(graph, dag)), limit)
    ]
