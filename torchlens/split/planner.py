"""Split target resolution and live-frontier planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import floor
from types import BuiltinFunctionType
from typing import Any, Literal

from ..backends.registry import TORCH_BACKEND_NAME
from ..intervention.types import CapturedArgTemplate, LiteralValue
from .errors import SplitErrorContext, SplitRequestError, SplitUnsupportedError
from .frontier import (
    boundary_key_for_node,
    classify_boundary_role,
    make_boundary_schema,
)
from .graph import (
    ReplayValueRef,
    SplitTraceGraph,
    SplitTraceNode,
    iter_replay_value_refs as _walk_replay_value_refs,
)
from .ir import BoundarySchema, GraphPartition, SplitRequest

BoundaryKind = Literal["after", "before", "partition"]


@dataclass(frozen=True)
class SplitPlan:
    """Resolved split plan."""

    split_id: str
    boundary_kind: BoundaryKind
    target_node_id: str | None
    prefix_node_ids: frozenset[str]
    suffix_node_ids: frozenset[str]
    boundary_node_ids: tuple[str, ...]
    boundary_spec: dict[str, BoundarySchema]
    boundary_bindings: dict[str, str]
    ordered_mutation_node_ids: frozenset[str] = frozenset()


def _resolve_percent_target(graph: SplitTraceGraph, percent: float) -> SplitTraceNode:
    """Resolve a percent target among eligible compute nodes."""

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
    target: SplitTraceNode | None,
    boundary_kind: BoundaryKind,
    boundary_node_ids: tuple[str, ...],
) -> dict[str, BoundarySchema]:
    """Build public boundary specs for crossing node IDs."""

    node_by_id = graph.node_by_id
    direct_parent_ids = {
        parent_node.canonical_id
        for parent in (() if target is None else target.parents)
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
            target_node_id=None if target is None else target.canonical_id,
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


def _split_id(graph: SplitTraceGraph, spec: SplitRequest, target: SplitTraceNode | None) -> str:
    """Compute a stable split identifier."""

    payload = repr(
        (
            graph.backend,
            graph.graph_shape_hash,
            spec.boundary,
            spec.batch_symbol,
            spec.validation,
            spec.features.training,
            None if target is None else target.canonical_id,
        )
    ).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def _partition_refusal(
    graph: SplitTraceGraph,
    spec: SplitRequest,
    reason: str,
    message: str,
    node: SplitTraceNode | None = None,
) -> SplitUnsupportedError:
    """Build a stable partition refusal without inventing a point target."""

    return SplitUnsupportedError(
        message,
        context=SplitErrorContext(
            backend=graph.backend,
            split_point=spec.boundary,
            reason=reason,
            module_path=None if node is None else node.module_path,
            op_type=None if node is None else node.op_type,
            layer_label=None if node is None else node.label,
        ),
    )


def _partition_sets(
    graph: SplitTraceGraph, spec: SplitRequest
) -> tuple[frozenset[str], frozenset[str]]:
    """Validate a downward-closed compute selection with fixed input/output sides.

    Native calls with several output values are indivisible. Source buffers live
    on the prefix; mutated state cannot yet be proven safe across a partition and
    refuses explicitly. Existing native lowerers retain responsibility for target
    and device support, and for restoring per-call RNG/autocast context.
    """

    if not isinstance(spec.point, GraphPartition):
        raise TypeError("Partition planning requires a GraphPartition")
    selected = spec.point.prefix_node_ids
    compute_ids = {node.canonical_id for node in graph.compute_nodes}
    unknown = selected - compute_ids
    if unknown:
        raise _partition_refusal(
            graph,
            spec,
            "partition_node_invalid",
            f"Partition requires canonical compute-node IDs; invalid IDs: {tuple(sorted(unknown))!r}.",
        )
    for call in graph.replay_calls:
        members = set(call.output_node_ids)
        if selected & members and not members <= selected:
            raise _partition_refusal(
                graph,
                spec,
                "partition_call_split",
                f"Native call {call.call_id!r} must keep all outputs together: "
                f"{call.output_node_ids!r}.",
            )

    prefix = _partition_source_ids(graph, spec, selected)
    suffix = set(graph.node_by_id) - prefix
    _validate_partition_dependencies(graph, spec, prefix)
    _validate_partition_shared_training_state(graph, spec, frozenset(prefix))
    return frozenset(prefix), frozenset(suffix)


def _partition_source_ids(
    graph: SplitTraceGraph, spec: SplitRequest, selected: frozenset[str]
) -> set[str]:
    """Place fixed sources and read-only buffer observations with their producers."""

    prefix = set(selected)
    for node in graph.nodes:
        if node.is_input or node.is_buffer or node.is_buffer_only_source:
            if node.parents and not node.is_input:
                # Captures also emit buffer observations following a read-only
                # operation (for example eval BatchNorm). Keep those observations
                # with their producer instead of treating them as input sources.
                sides = {
                    observation_parent.canonical_id in prefix
                    for label in node.parents
                    if (observation_parent := graph.node_for_label(label)) is not None
                }
                if len(sides) != 1:
                    raise _partition_refusal(
                        graph,
                        spec,
                        "partition_mutable_state_unsupported",
                        "Partition cannot place a buffer observation spanning both segments.",
                        node,
                    )
                if sides == {True}:
                    prefix.add(node.canonical_id)
                continue
            prefix.add(node.canonical_id)
    return prefix


def _validate_partition_dependencies(
    graph: SplitTraceGraph, spec: SplitRequest, prefix: set[str]
) -> None:
    """Reject unresolved edges and dependencies from suffix into prefix."""

    for node in graph.nodes:
        for parent in node.parents:
            parent_node = graph.node_for_label(parent)
            if parent_node is None:
                raise _partition_refusal(
                    graph,
                    spec,
                    "partition_dependency_unresolved",
                    f"Cannot resolve dependency {parent!r} of {node.canonical_id!r}.",
                    node,
                )
            if node.canonical_id in prefix and parent_node.canonical_id not in prefix:
                raise _partition_refusal(
                    graph,
                    spec,
                    "partition_reverse_dependency",
                    f"Suffix node {parent_node.canonical_id!r} feeds prefix node "
                    f"{node.canonical_id!r}; include its compute ancestors in the prefix.",
                    node,
                )


def _validate_partition_shared_training_state(
    graph: SplitTraceGraph, spec: SplitRequest, prefix: frozenset[str]
) -> None:
    """Refuse tied trainable replicas without a cross-device gradient merge contract."""

    if not spec.trainable or not spec.placement.is_heterogeneous:
        return
    sides_by_parameter: dict[int, set[bool]] = {}
    for node in graph.compute_nodes:
        for parameter in node.param_refs:
            value = getattr(parameter, "handle", None)
            if value is None:
                value = getattr(parameter, "_param_ref", None)
            trainable = bool(
                getattr(parameter, "is_trainable", False) or getattr(value, "requires_grad", False)
            )
            if not trainable:
                continue
            identity = id(parameter) if value is None else id(value)
            sides = sides_by_parameter.setdefault(identity, set())
            sides.add(node.canonical_id in prefix)
            if len(sides) > 1:
                raise _partition_refusal(
                    graph,
                    spec,
                    "partition_shared_trainable_cross_device",
                    "A trainable parameter is shared across different segment devices; "
                    "the runtime cannot consolidate its replica gradients. Place the "
                    "shared consumers on one device or keep them in one segment.",
                    node,
                )


@dataclass(frozen=True)
class _MutationTargets:
    """Exact native call identities covered by the local storage proof."""

    writes: frozenset[Any]
    fresh: frozenset[Any]
    batchnorm: frozenset[Any]
    permutations: frozenset[Any]
    out_writes: frozenset[Any]
    out_allocations: frozenset[Any]


@dataclass
class _MutationProof:
    """One partition's assignment and already certified ordered writes."""

    prefix: frozenset[str]
    targets: _MutationTargets
    certified: set[str] = field(default_factory=set)


def _torch_mutation_targets() -> _MutationTargets:
    """Resolve exact native targets with known write/fresh-allocation semantics."""

    import torch

    from .. import _state

    originals, _ = _state.wrap_epoch_ledgers()
    # The capture wrapper ledger already owns native target identities. Reuse
    # that census instead of probing a new Torch-private attribute surface.
    native_silu_writes = tuple(
        target
        for target in originals.values()
        if isinstance(target, BuiltinFunctionType)
        and target.__module__ == "torch._C._nn"
        and target.__name__ == "silu_"
    )

    def targets(values: tuple[Any, ...]) -> frozenset[Any]:
        """Resolve persistent logging wrappers to their original callable objects."""

        return frozenset(originals.get(id(value), value) for value in values if value is not None)

    writes = targets(
        (
            torch.relu_,
            torch.Tensor.relu_,
            torch.Tensor.add_,
            torch.Tensor.__iadd__,
            torch.nn.functional.relu,
            torch.nn.functional.silu,
            *native_silu_writes,
        )
    )
    fresh = targets(
        (
            torch.conv1d,
            torch.conv2d,
            torch.conv3d,
            torch.nn.functional.linear,
            torch.add,
            torch.Tensor.add,
            torch.Tensor.__add__,
            torch.layer_norm,
            torch.nn.functional.layer_norm,
        )
    )
    batchnorm = targets((torch.batch_norm, torch.nn.functional.batch_norm))
    permutations = targets((torch.permute, torch.Tensor.permute))
    out_writes = targets((torch.arange,))
    out_allocations = targets((torch.Tensor.new_full,))
    return _MutationTargets(writes, fresh, batchnorm, permutations, out_writes, out_allocations)


def _certify_local_write(
    graph: SplitTraceGraph,
    node: SplitTraceNode,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    proof: _MutationProof,
) -> bool:
    """Prove a native write stays with its fresh allocation and preserves effect order.

    The executor separately refuses tensor subclasses and dispatch modes for
    plans carrying this proof. Read-only aliases may escape only after the write;
    permutations preserve the proven local allocation's storage origin. Other
    views and transported values are not certified.
    """

    targets = proof.targets
    prefix = proof.prefix
    has_out = kwargs.get("out") is not None
    allowed_targets = targets.out_writes if has_out else targets.writes
    argument = kwargs["out"] if has_out else args[0] if args else kwargs.get("input")
    if node.target not in allowed_targets:
        return False
    if not isinstance(argument, ReplayValueRef):
        return False
    producer = graph.node_by_id.get(argument.value_id)
    if producer is None or any((producer.is_input, producer.is_buffer, producer.is_param_source)):
        return False
    in_prefix = node.canonical_id in prefix
    if (producer.canonical_id in prefix) != in_prefix:
        return False
    if has_out and not _out_initialization_matches(graph, node, producer, targets.out_allocations):
        return False
    origin = _local_storage_origin(graph, producer, proof, in_prefix)
    return (
        origin is not None
        and _write_preserves_order(graph, node, prefix)
        and (
            has_out
            or origin.canonical_id in proof.certified
            or origin.target in targets.fresh
            or _is_readonly_batchnorm(origin, targets.batchnorm)
        )
    )


def _out_initialization_matches(
    graph: SplitTraceGraph,
    node: SplitTraceNode,
    producer: SplitTraceNode,
    allocations: frozenset[Any],
) -> bool:
    """Certify direct non-resizing arange initialization of fresh anchor storage."""

    if (
        producer.target not in allocations
        or producer.requires_grad is not False
        or node.requires_grad is not False
        or producer.output_shape is None
        or producer.output_shape != node.output_shape
        or producer.dtype != node.dtype
        or producer.symbolic_output_shape is None
        or producer.symbolic_output_shape != node.symbolic_output_shape
        or graph.shape_program is None
    ):
        return False
    source_shape = graph.shape_program.value_shapes.get(producer.canonical_id)
    result_shape = graph.shape_program.value_shapes.get(node.canonical_id)
    return (
        source_shape is not None
        and result_shape is not None
        and source_shape.dims == result_shape.dims
    )


def _local_storage_origin(
    graph: SplitTraceGraph, producer: SplitTraceNode, proof: _MutationProof, in_prefix: bool
) -> SplitTraceNode | None:
    """Follow only proven local permutations to their allocation, rejecting cycles."""

    seen: set[str] = set()
    while producer.target in proof.targets.permutations:
        if producer.canonical_id in seen:
            return None
        seen.add(producer.canonical_id)
        template = producer.args_template
        if not isinstance(template, CapturedArgTemplate) or not template.args:
            return None
        source = template.args[0]
        if not isinstance(source, ReplayValueRef):
            return None
        parent = graph.node_by_id.get(source.value_id)
        if parent is None or (parent.canonical_id in proof.prefix) != in_prefix:
            return None
        producer = parent
    return producer


def _write_preserves_order(
    graph: SplitTraceGraph, node: SplitTraceNode, prefix: frozenset[str]
) -> bool:
    """Keep every earlier/later call on the safe side of an ordered write."""

    order = graph.order_by_id
    in_prefix = node.canonical_id in prefix
    for other in graph.compute_nodes:
        if (
            in_prefix
            and order[other.canonical_id] < order[node.canonical_id]
            and other.canonical_id not in prefix
        ):
            return False
        if (
            not in_prefix
            and order[other.canonical_id] > order[node.canonical_id]
            and other.canonical_id in prefix
        ):
            return False
    return True


def _is_readonly_batchnorm(producer: SplitTraceNode, batchnorm: frozenset[Any]) -> bool:
    """Recognize the affirmative disabled training flag of a native BatchNorm."""

    if producer.target in batchnorm and isinstance(producer.args_template, CapturedArgTemplate):
        template = producer.args_template
        training = dict(template.kwargs).get(
            "training", template.args[5] if len(template.args) > 5 else None
        )
        return isinstance(training, LiteralValue) and training.value is False
    return False


def _validate_partition_effects(
    graph: SplitTraceGraph,
    spec: SplitRequest,
    *,
    prefix_node_ids: frozenset[str] | None = None,
) -> frozenset[str]:
    """Refuse writes whose ordering/alias obligations are absent from the data DAG.

    An explicit partition may certify a narrow set of ordered writes on fresh
    native allocations. Without an assignment, all writes remain refused (for
    example when a profiler cannot isolate arbitrary alias relationships).
    """

    known_identity: Any = None
    proof = None
    certified: set[str] = set()
    if graph.backend == TORCH_BACKEND_NAME:
        from ..utils.display import identity

        known_identity = identity
        if prefix_node_ids is not None:
            proof = _MutationProof(prefix_node_ids, _torch_mutation_targets(), certified)
    for node in graph.compute_nodes:
        args, kwargs = _partition_call_arguments(node)
        if _node_mutates(node, args, kwargs):
            if proof is not None and _certify_local_write(graph, node, args, kwargs, proof):
                certified.add(node.canonical_id)
            else:
                raise _partition_refusal(
                    graph,
                    spec,
                    "partition_effect_order_unproven",
                    f"Partition cannot certify mutation/alias ordering for {node.canonical_id!r}; "
                    "keep a supported write with its fresh producer and preserve effect order, "
                    "or use an out-of-place graph or point split.",
                    node,
                )
        if (
            graph.backend == TORCH_BACKEND_NAME
            and node.target is not None
            and node.target is not known_identity
        ):
            module = str(getattr(node.target, "__module__", "") or "")
            owner = getattr(node.target, "__objclass__", None)
            owner_module = str(getattr(owner, "__module__", "") or "")
            if not any(
                name == "torch" or name.startswith("torch.") for name in (module, owner_module)
            ):
                raise _partition_refusal(
                    graph,
                    spec,
                    "partition_custom_effects_unproven",
                    f"Partition cannot certify side effects of custom target at {node.canonical_id!r}.",
                    node,
                )
    return frozenset(certified)


def _partition_call_arguments(node: SplitTraceNode) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Read captured call arguments and unwrap only literal metadata values."""

    template = node.args_template
    args = (
        template.args
        if isinstance(template, CapturedArgTemplate)
        else template
        if isinstance(template, (tuple, list))
        else ()
    )
    kwargs = dict(template.kwargs) if isinstance(template, CapturedArgTemplate) else {}
    if isinstance(node.kwargs_template, dict):
        kwargs.update(node.kwargs_template)
    elif isinstance(node.kwargs_template, CapturedArgTemplate):
        kwargs.update(node.kwargs_template.kwargs)
    return (
        tuple(item.value if isinstance(item, LiteralValue) else item for item in args),
        {
            key: item.value if isinstance(item, LiteralValue) else item
            for key, item in kwargs.items()
        },
    )


def _node_mutates(node: SplitTraceNode, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Recognize write markers and conservatively classify BatchNorm training."""

    names = (node.op_type.lower(), str(getattr(node.target, "__name__", "")).lower())
    mutates = (
        bool(getattr(node.op, "is_inplace", False))
        or any(name.endswith("_") and not name.endswith("__") for name in names)
        or any(name in {"__setitem__", "setitem"} for name in names)
        or kwargs.get("inplace") is True
        or kwargs.get("out") is not None
    )
    if any("batch_norm" in name or "batchnorm" in name for name in names):
        training = kwargs.get("training", args[5] if len(args) > 5 else None)
        mutates = mutates or training is not False
    return mutates


def plan_split(graph: SplitTraceGraph, spec: SplitRequest) -> SplitPlan:
    """Resolve a :class:`SplitRequest` into executable split plan."""

    _validate_replay_dependencies(graph)
    target: SplitTraceNode | None
    boundary_kind: BoundaryKind
    ordered_mutations: frozenset[str] = frozenset()
    if isinstance(spec.point, GraphPartition):
        boundary_kind = "partition"
        target = None
        prefix_node_ids, suffix_node_ids = _partition_sets(graph, spec)
        ordered_mutations = _validate_partition_effects(
            graph, spec, prefix_node_ids=prefix_node_ids
        )
    else:
        if spec.point.kind == "percent":
            boundary_kind = "after"
            target = _resolve_percent_target(graph, float(spec.point.target))
        else:
            boundary_kind = spec.point.kind
            target = _resolve_named_target(graph, str(spec.point.target))
        _reject_invalid_target(target)
        prefix_node_ids, suffix_node_ids = _prefix_suffix_sets(
            graph, target, boundary_kind=boundary_kind
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
        target_node_id=None if target is None else target.canonical_id,
        prefix_node_ids=prefix_node_ids,
        suffix_node_ids=suffix_node_ids,
        boundary_node_ids=boundary_node_ids,
        boundary_spec=boundary_spec,
        boundary_bindings=boundary_bindings,
        ordered_mutation_node_ids=ordered_mutations,
    )


__all__ = ["BoundaryKind", "SplitPlan", "plan_split"]
