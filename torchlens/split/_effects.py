"""Graph-local storage effects for the existing two-segment partition planner.

Only structural provenance is cached. Live source alias relations are checked
again at analysis and execution; activation addresses never enter the proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..intervention.types import CapturedArgTemplate, LiteralTensor
from .graph import ReplayValueRef, SplitTraceGraph, SplitTraceNode


@dataclass(frozen=True)
class _Access:
    """Conservative storage reads and writes of one atomic native call."""

    reads: frozenset[str]
    writes: frozenset[str]
    kind: str


@dataclass(frozen=True)
class _EffectIndex:
    """Linear graph metadata, without tensor payloads or runtime replicas."""

    accesses: dict[str, _Access]
    origins: dict[str, frozenset[str]]
    predecessors: dict[str, frozenset[str]]
    source_nodes: tuple[tuple[str, str], ...]
    literal_nodes: tuple[str, ...]
    source_occurrences: dict[str, str]
    source_writes: frozenset[str]


def _literal_tensors(value: Any) -> tuple[LiteralTensor, ...]:
    """Visit native template tensors in the executor's occurrence order."""
    if isinstance(value, LiteralTensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(item for part in value for item in _literal_tensors(part))
    if isinstance(value, dict):
        return tuple(item for part in value.values() for item in _literal_tensors(part))
    return ()


def _call_literals(node: SplitTraceNode) -> tuple[LiteralTensor, ...]:
    """Read tensor literals from normalized arguments, including keyword operands."""
    template = node.args_template
    if isinstance(template, CapturedArgTemplate):
        return (
            _literal_tensors(template.args)
            + tuple(item for _, part in template.kwargs for item in _literal_tensors(part))
            + _literal_tensors(node.kwargs_template)
        )
    return _literal_tensors(template) + _literal_tensors(node.kwargs_template)


def _source_root(value: Any) -> str:
    """Name a live state identity without retaining the state itself."""
    return f"state:{id(value)}"


def _operand_roots(value: Any, origins: dict[str, frozenset[str]]) -> frozenset[str]:
    """Resolve possible storage origins, overapproximating runtime-copying views."""
    if isinstance(value, ReplayValueRef):
        return origins.get(value.value_id, frozenset({f"unknown:{value.value_id}"}))
    if isinstance(value, LiteralTensor):
        return frozenset({_source_root(value.value)})
    if isinstance(value, (tuple, list)):
        return frozenset(root for part in value for root in _operand_roots(part, origins))
    if isinstance(value, dict):
        return frozenset(root for part in value.values() for root in _operand_roots(part, origins))
    return frozenset()


def _write_operands(
    node: SplitTraceNode, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, ...]:
    """Identify native BatchNorm running-state writes and ordinary destination writes."""
    if "batchnorm" in node.op_type.replace("_", ""):
        positions = (
            (1, 2) if getattr(node.target, "__module__", "") == "torch.nn.functional" else (3, 4)
        )
        return tuple(
            kwargs.get(name, args[index] if len(args) > index else None)
            for name, index in zip(("running_mean", "running_var"), positions, strict=True)
        )
    return (
        kwargs.get("out")
        if kwargs.get("out") is not None
        else args[0]
        if args
        else kwargs.get("input"),
    )


def _access_for_node(
    node: SplitTraceNode,
    origins: dict[str, frozenset[str]],
    targets: Any,
    stochastic: dict[int, str],
) -> tuple[_Access, frozenset[str]]:
    """Classify storage and generator effects using exact native target identities."""
    from ._torch_rng_replay import _uses_randomness
    from .planner import _node_mutates, _partition_call_arguments

    args, kwargs = _partition_call_arguments(node)
    reads = _operand_roots((args, kwargs), origins)
    mutates = _node_mutates(node, args, kwargs)
    writes = (
        _operand_roots(_write_operands(node, args, kwargs), origins) if mutates else frozenset()
    )
    kind = (
        "buffer_write"
        if node.target in targets.batchnorm and mutates
        else "inplace_write"
        if mutates
        else "read"
    )
    output = (
        frozenset({f"fresh:{node.canonical_id}"})
        if node.target in targets.fresh | targets.batchnorm | targets.out_allocations
        else reads
    )
    if mutates and node.target not in targets.batchnorm:
        output = writes
    if not output:
        output = frozenset({f"unknown:{node.canonical_id}"})
    random_kind = stochastic.get(id(node.target))
    if random_kind is not None and _uses_randomness(random_kind, args, kwargs):
        # Ordinary generators are restored per call by native replay. An
        # explicitly supplied Generator has mutable state outside that scope.
        generator = kwargs.get("generator")
        if generator is not None or not getattr(node.op, "func_rng_states", None):
            # Argument capture may copy Generator objects independently per
            # call. Their copied Python identities cannot prove independence
            # of the original generator; preserve order until that provenance
            # has a native capture contract.
            resource = "rng:explicit_unproven" if generator is not None else "rng:ambient"
            reads |= {resource}
            writes |= {resource}
            kind = "generator_state"
    return _Access(reads, writes, kind), output


def _source_provenance(
    graph: SplitTraceGraph,
) -> tuple[dict[str, frozenset[str]], list[tuple[str, str]], list[str], dict[str, str]]:
    """Collect structural source identities and native argument occurrences."""
    from ._torch_compact import live_buffer_value

    origins: dict[str, frozenset[str]] = {}
    source_nodes, literal_nodes = [], []
    occurrences: dict[str, str] = {}
    for node in graph.nodes:
        if node.is_input:
            origins[node.canonical_id] = frozenset({"inputs"})
        elif node.is_buffer or (node.target is None and not node.parents):
            value = (
                live_buffer_value(node.buffer_refs[0])
                if node.buffer_refs
                else getattr(node.op, "out", None)
            )
            root = _source_root(value) if value is not None else f"unknown:{node.canonical_id}"
            origins[node.canonical_id] = frozenset({root})
            source_nodes.append((node.canonical_id, root))
            occurrences[f"{node.canonical_id}:source"] = root
        literals = _call_literals(node)
        if literals:
            literal_nodes.append(node.canonical_id)
        for index, literal in enumerate(literals):
            occurrences[f"{node.canonical_id}:literal:{index}"] = _source_root(literal.value)
    return origins, source_nodes, literal_nodes, occurrences


def _build_effect_index(graph: SplitTraceGraph) -> _EffectIndex:
    """Record RAW, WAR and WAW edges once, with no global mutation chain."""
    from ._torch_rng_replay import stochastic_targets
    from .planner import _torch_mutation_targets

    targets, stochastic = _torch_mutation_targets(), stochastic_targets()
    origins, source_nodes, literal_nodes, occurrences = _source_provenance(graph)
    accesses: dict[str, _Access] = {}
    predecessors: dict[str, frozenset[str]] = {}
    last_writer: dict[str, str] = {}
    readers: dict[str, set[str]] = {}
    for call in graph.replay_calls:
        node = graph.node_by_id[call.output_node_ids[0]]
        access, output = _access_for_node(node, origins, targets, stochastic)
        dependencies = {
            last_writer[root] for root in access.reads | access.writes if root in last_writer
        }
        for root in access.writes:
            dependencies.update(readers.get(root, ()))
        for member in call.output_node_ids:
            accesses[member] = access
            origins[member] = output
            predecessors[member] = frozenset(dependencies)
        for root in access.writes:
            last_writer[root] = node.canonical_id
            readers[root] = set()
        for root in access.reads - access.writes:
            readers.setdefault(root, set()).add(node.canonical_id)
    return _EffectIndex(
        accesses,
        origins,
        predecessors,
        tuple(source_nodes),
        tuple(literal_nodes),
        occurrences,
        frozenset(
            root
            for access in accesses.values()
            for root in access.writes
            if root.startswith("state:")
        ),
    )


def _source_values(graph: SplitTraceGraph) -> dict[str, list[Any]]:
    """Resolve current source handles without caching their addresses or replicas."""
    from ._torch_compact import live_buffer_value

    result: dict[str, list[Any]] = {}
    index = graph._partition_effect_index
    for node_id, root in index.source_nodes:
        node = graph.node_by_id[node_id]
        value = (
            live_buffer_value(node.buffer_refs[0])
            if node.buffer_refs
            else getattr(node.op, "out", None)
        )
        result.setdefault(root, []).append(value)
    for node_id in index.literal_nodes:
        for occurrence, literal in enumerate(_call_literals(graph.node_by_id[node_id])):
            root = index.source_occurrences[f"{node_id}:literal:{occurrence}"]
            result.setdefault(root, []).append(literal.value)
    return result


def _source_alias_conflict(
    values: dict[str, list[Any]], writes: frozenset[str]
) -> tuple[str, str, str] | None:
    """Check current written sources against other logical sources, failing closed."""
    from .._state import pause_logging
    from ..utils.alias_footprint import tensor_byte_footprint, touched_bytes_relation

    unique = {
        root: tuple({id(value): value for value in items}.values())
        for root, items in values.items()
    }
    with pause_logging():
        footprints = {
            id(value): tensor_byte_footprint(value) for items in unique.values() for value in items
        }
        for root in writes:
            for value in unique.get(root, ()):
                for other, items in unique.items():
                    if root == other:
                        continue
                    left = footprints[id(value)]
                    for item in items:
                        right = footprints[id(item)]
                        # Use the shared engine's absolute device-scoped geometry
                        # once per source, rather than once per pair of sources.
                        if (
                            left is not None
                            and right is not None
                            and (
                                left.device_key != right.device_key
                                or left.end_byte <= right.start_byte
                                or right.end_byte <= left.start_byte
                            )
                        ):
                            continue
                        relation = touched_bytes_relation(value, item)
                        if relation != "disjoint":
                            return root, other, relation
    return None


def _alias_conflict_category(conflict: tuple[str, str, str]) -> str:
    """Distinguish an observed overlap from an unknown native storage relation."""
    return (
        "cross_segment_state_conflict" if conflict[2] == "overlap" else "alias_provenance_unknown"
    )


def _guard_effect_sources(
    runtime: Any,
    inputs: Any,
    *,
    plan: Any = None,
    snapshots: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Recheck source alias assumptions against current inputs and effective state."""
    from .adapters.torch import _flatten_tensor_leaves, _torch
    from .errors import SplitErrorContext, SplitUnsupportedError

    graph = runtime.trace_graph
    plan = runtime.plan if plan is None else plan
    index = graph._partition_effect_index
    if not index.source_writes:
        return
    values = _source_values(graph)
    values["inputs"] = _flatten_tensor_leaves(inputs, _torch())
    snapshots = runtime._segment_state_snapshots() if snapshots is None else snapshots
    for snapshot in snapshots.values():
        for occurrence, value in snapshot.items():
            root = index.source_occurrences.get(occurrence)
            if root is not None:
                values.setdefault(root, []).append(value)
    conflict = _source_alias_conflict(values, index.source_writes)
    if conflict is not None:
        error = SplitUnsupportedError(
            f"Current state/input storage invalidates partition independence: {conflict!r}.",
            context=SplitErrorContext(
                backend="torch",
                split_point=runtime.request.boundary,
                reason="partition_effect_order_unproven",
            ),
        )
        error._partition_diagnostic = {
            "partition": plan.split_id,
            "category": _alias_conflict_category(conflict),
            "source_alias_conflict": conflict,
            "reason": "partition_effect_order_unproven",
            "native_call": None,
        }
        raise error


def _effect_inversions(
    graph: SplitTraceGraph, prefix: frozenset[str]
) -> tuple[tuple[str, str], ...]:
    """Find required suffix-before-prefix edges that a two-segment run cannot honor."""
    return tuple(
        (parent, child)
        for child, parents in graph._partition_effect_index.predecessors.items()
        if child in prefix
        for parent in sorted(parents, key=graph.order_by_id.__getitem__)
        if parent not in prefix
    )


def _boundary_state_aliases(
    graph: SplitTraceGraph,
    boundary: tuple[str, ...],
    suffix: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    """Find activation frontiers that may alias state subsequently written by the suffix.

    Training leaf isolation and device transport do not preserve such aliases.
    Dedicated mutable buffer slots use the existing publication contract; an
    additional activation view of that storage has no corresponding contract.
    """
    index = graph._partition_effect_index
    writes = {
        node_id: access.writes & index.source_writes
        for node_id, access in index.accesses.items()
        if node_id in suffix and access.writes & index.source_writes
    }
    return tuple(
        (node_id, writer)
        for node_id in boundary
        if not graph.node_by_id[node_id].is_buffer
        for writer, roots in writes.items()
        if roots & index.origins.get(node_id, frozenset())
    )


def _partition_diagnostic(
    graph: SplitTraceGraph, spec: Any, reason: str, node: SplitTraceNode | None
) -> dict[str, Any]:
    """Return payload-free internal refusal evidence without changing public errors."""
    prefix = spec.partition.prefix_node_ids
    index = graph._partition_effect_index
    access = index.accesses.get(node.canonical_id) if node is not None else None
    inversions = _effect_inversions(graph, prefix)
    buffer_roots = {
        root for node_id, root in index.source_nodes if graph.node_by_id[node_id].is_buffer
    }
    parameter_roots = {
        _source_root(literal.value)
        for node_id in index.literal_nodes
        for literal in _call_literals(graph.node_by_id[node_id])
        if literal.param_barcode is not None
    }
    category = "mutation_order_unproven"
    if reason in {
        "partition_reverse_dependency",
        "partition_dependency_unresolved",
        "partition_call_split",
        "partition_node_invalid",
    }:
        category = "candidate_dependency_invalid"
    elif reason == "partition_custom_effects_unproven":
        category = "unsupported_native_effect"
    elif reason == "partition_shared_trainable_cross_device":
        category = "cross_segment_state_conflict"
    elif inversions:
        category = (
            "cross_segment_state_conflict"
            if any(
                root.startswith("state:")
                for a, b in inversions
                for root in index.accesses[a].writes | index.accesses[b].writes
            )
            else "true_effect_conflict"
        )
    elif access is not None and any(
        root.startswith("unknown:") for root in access.reads | access.writes
    ):
        category = "alias_provenance_unknown"
    if node is not None and access is not None and access.kind == "inplace_write":
        from .planner import _partition_call_arguments, _torch_mutation_targets

        targets = _torch_mutation_targets()
        if node.target not in targets.writes | targets.out_writes:
            category = "unsupported_native_effect"
        elif not inversions:
            args, kwargs = _partition_call_arguments(node)
            argument = args[0] if args else kwargs.get("input")
            producer = (
                graph.node_by_id.get(argument.value_id)
                if isinstance(argument, ReplayValueRef)
                else None
            )
            if (
                producer is not None
                and not (producer.is_input or producer.is_buffer or producer.is_param_source)
                and producer.target
                not in targets.fresh
                | targets.batchnorm
                | targets.permutations
                | targets.writes
                | targets.out_allocations
            ):
                category = "alias_provenance_unknown"
    crossed = (
        []
        if node is None
        else [
            other.canonical_id
            for other in graph.compute_nodes
            if (other.canonical_id in prefix) != (node.canonical_id in prefix)
            and (graph.order_by_id[other.canonical_id] < graph.order_by_id[node.canonical_id])
            == (node.canonical_id in prefix)
        ]
    )
    possible_aliases = (
        []
        if access is None
        else [value_id for value_id, roots in index.origins.items() if access.writes & roots]
    )
    return {
        "partition": spec.boundary,
        "prefix_nodes": len(prefix),
        "suffix_nodes": len(graph.compute_nodes) - len(prefix),
        "native_call": None if node is None else node.canonical_id,
        "native_call_id": None if node is None else node.func_call_id,
        "original_order": None if node is None else graph.order_by_id[node.canonical_id],
        "raw_call_order": None if node is None else node.raw_index,
        "effect": None if access is None else access.kind,
        "mutation": bool(access and access.writes),
        "storage_origins": [] if access is None else sorted(access.reads | access.writes),
        "possible_aliases": possible_aliases,
        "accesses_parameter": bool(access and parameter_roots & access.reads),
        "accesses_buffer": bool(access and buffer_roots & (access.reads | access.writes)),
        "intervening_calls": crossed,
        "crosses_other_calls": bool(crossed),
        "effect_conflicts": inversions,
        "effect_constraint_basis": "conservative_native_storage_provenance",
        "category": category,
        "reason": reason,
    }
