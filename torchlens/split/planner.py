"""Canonical DAG partition validation and live-frontier planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import BuiltinFunctionType
from typing import Any

from ..backends.registry import TORCH_BACKEND_NAME
from ..intervention.types import CapturedArgTemplate, LiteralTensor, LiteralValue
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


@dataclass(frozen=True)
class SplitPlan:
    """Resolved split plan."""

    split_id: str
    prefix_node_ids: frozenset[str]
    suffix_node_ids: frozenset[str]
    boundary_node_ids: tuple[str, ...]
    boundary_spec: dict[str, BoundarySchema]
    boundary_bindings: dict[str, str]
    ordered_mutation_node_ids: frozenset[str] = frozenset()
    training_boundary_writes: tuple[str, ...] = ()


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


def _boundary_metadata(
    graph: SplitTraceGraph,
    *,
    spec: SplitRequest,
    boundary_node_ids: tuple[str, ...],
) -> tuple[dict[str, BoundarySchema], dict[str, str]]:
    """Build boundary schemas and canonical bindings from the same unique keys."""

    node_by_id = graph.node_by_id
    spatial_shapes: set[tuple[int, ...]] = set()
    for node_id in boundary_node_ids:
        output_shape = node_by_id[node_id].output_shape
        if output_shape is not None and len(output_shape) >= 4:
            spatial_shapes.add(tuple(output_shape[-2:]))
    specs: dict[str, BoundarySchema] = {}
    bindings: dict[str, str] = {}
    for node_id in boundary_node_ids:
        node = node_by_id[node_id]
        key = boundary_key_for_node(node.canonical_id, node.output_container_path)
        if key in bindings:
            raise SplitRequestError("Split boundary contains duplicate canonical value bindings.")
        bindings[key] = node_id
        role = classify_boundary_role(
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
    return specs, bindings


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


def _split_id(graph: SplitTraceGraph, spec: SplitRequest) -> str:
    """Compute a stable split identifier."""

    payload = repr(
        (
            graph.backend,
            graph.graph_shape_hash,
            spec.boundary,
            spec.batch_symbol,
            spec.validation,
            spec.features.training,
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

    if not isinstance(spec.partition, GraphPartition):
        raise TypeError("Partition planning requires a GraphPartition")
    selected = spec.partition.prefix_node_ids
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
    ordered: frozenset[str]
    certified: set[str] = field(default_factory=set)
    training: bool = False
    heterogeneous: bool = False


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
            torch.embedding_renorm_,
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
            torch.clone,
            torch.Tensor.clone,
        )
    )
    batchnorm = targets((torch.batch_norm, torch.nn.functional.batch_norm))
    permutations = targets(
        (
            torch.permute,
            torch.Tensor.permute,
            torch.Tensor.view,
            torch.Tensor.view_as,
            torch.reshape,
            torch.Tensor.reshape,
            torch.transpose,
            torch.Tensor.transpose,
            torch.flatten,
            torch.Tensor.flatten,
            torch.chunk,
            torch.Tensor.chunk,
            torch.split,
            torch.Tensor.split,
            torch.Tensor.detach,
        )
    )
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
    if isinstance(argument, LiteralTensor):
        return (
            getattr(node.target, "__name__", "") == "embedding_renorm_"
            and getattr(argument.value, "requires_grad", True) is False
            and node.canonical_id in proof.ordered
        )
    if not isinstance(argument, ReplayValueRef):
        return False
    producer = graph.node_by_id.get(argument.value_id)
    if producer is None:
        return False
    if producer.is_input or producer.is_buffer or producer.is_buffer_only_source:
        # Native source writes are safe only in the unchanged execution order.
        # An input must stay on its owning side; non-differentiable buffer writes
        # may consume the source across a sequential frontier.
        return (
            not has_out
            and node.canonical_id in proof.ordered
            and (
                (
                    producer.is_input
                    and not proof.training
                    and (node.canonical_id in prefix or not proof.heterogeneous)
                )
                or (
                    producer.is_buffer
                    and producer.requires_grad is not True
                    and (node.canonical_id in prefix or not proof.heterogeneous)
                )
            )
        )
    if producer.is_param_source:
        return (
            getattr(node.target, "__name__", "") == "embedding_renorm_"
            and producer.requires_grad is False
            and node.canonical_id in proof.ordered
        )
    if (
        getattr(node.target, "__name__", "") == "embedding_renorm_"
        and node.canonical_id in proof.ordered
        and producer.target in targets.permutations
        and getattr(producer.target, "__name__", "") == "detach"
    ):
        detached_args, _ = _partition_call_arguments(producer)
        return bool(
            detached_args
            and isinstance(detached_args[0], LiteralTensor)
            and getattr(detached_args[0].value, "requires_grad", True) is False
        )
    in_prefix = node.canonical_id in prefix
    if (producer.canonical_id in prefix) != in_prefix:
        # Training boundaries isolate integral leaves before suffix execution.
        # Such leaves have no autograd history to sever with a sequential write.
        return (
            not has_out
            and node.canonical_id in proof.ordered
            and producer.dtype is not None
            and (
                (not proof.training and not proof.heterogeneous)
                or any(kind in producer.dtype for kind in ("int", "bool"))
            )
        )
    if has_out and not _out_initialization_matches(graph, node, producer, targets.out_allocations):
        return False
    origin = _local_storage_origin(graph, producer, proof, in_prefix)
    if (
        not proof.training
        and not proof.heterogeneous
        and not has_out
        and node.canonical_id in proof.ordered
    ):
        # A sequential, same-device inference frontier preserves tensor aliases:
        # detach does not copy storage, and no autograd leaf is mutated.
        return True
    return (
        origin is not None
        and (
            node.canonical_id in proof.ordered
            or (not has_out and _independent_crossings(graph, node, origin, proof))
        )
        and (
            has_out
            or origin.canonical_id in proof.certified
            or origin.target in targets.fresh
            or origin.target in targets.batchnorm
            or (origin.is_input and not proof.training)
        )
    )


def _independent_crossings(
    graph: SplitTraceGraph, node: SplitTraceNode, origin: SplitTraceNode, proof: _MutationProof
) -> bool:
    """Allow fresh storage writes to cross only data-independent native operations.

    Unknown alias roots, source writes, out= initialization and mutable state
    keep their original ordering barrier. A crossed local write must have its own
    proven fresh allocation, independent of this allocation in both directions.
    """
    if origin.target not in proof.targets.fresh | proof.targets.batchnorm:
        return False
    order = graph.order_by_id
    index = order[node.canonical_id]
    in_prefix = node.canonical_id in proof.prefix
    origin_ancestors = graph.ancestors(origin.canonical_id)
    for crossed in graph.compute_nodes:
        if (crossed.canonical_id in proof.prefix) == in_prefix:
            continue
        if (order[crossed.canonical_id] < index) != in_prefix:
            continue
        if origin.canonical_id in graph.ancestors(crossed.canonical_id):
            return False
        args, kwargs = _partition_call_arguments(crossed)
        if not _node_mutates(crossed, args, kwargs):
            continue
        argument = args[0] if args else kwargs.get("input")
        if crossed.target not in proof.targets.writes or kwargs.get("out") is not None:
            return False
        if not isinstance(argument, ReplayValueRef):
            return False
        producer = graph.node_by_id.get(argument.value_id)
        if producer is None or (producer.canonical_id in proof.prefix) == in_prefix:
            return False
        other = _local_storage_origin(graph, producer, proof, not in_prefix)
        if other is None or other.target not in proof.targets.fresh | proof.targets.batchnorm:
            return False
        if (
            other.canonical_id == origin.canonical_id
            or other.canonical_id in origin_ancestors
            or origin.canonical_id in graph.ancestors(other.canonical_id)
        ):
            return False
    return True


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


def _ordered_partition_nodes(graph: SplitTraceGraph, prefix: frozenset[str]) -> frozenset[str]:
    """Find calls whose order relative to every other call survives the partition.

    A prefix write must precede every suffix call; a suffix write must follow
    every prefix call. Only the leading prefix and trailing suffix runs qualify.
    Compute those runs once instead of scanning the graph for each write.
    """

    ordered: set[str] = set()
    for node in graph.compute_nodes:
        if node.canonical_id not in prefix:
            break
        ordered.add(node.canonical_id)
    for node in reversed(graph.compute_nodes):
        if node.canonical_id in prefix:
            break
        ordered.add(node.canonical_id)
    return frozenset(ordered)


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
            proof = _MutationProof(
                prefix_node_ids,
                _torch_mutation_targets(),
                _ordered_partition_nodes(graph, prefix_node_ids),
                certified,
                spec.trainable,
                spec.placement.is_heterogeneous,
            )
    for node in graph.compute_nodes:
        args, kwargs = _partition_call_arguments(node)
        if _node_mutates(node, args, kwargs):
            if proof is not None and (
                _certify_local_write(graph, node, args, kwargs, proof)
                or (node.target in proof.targets.batchnorm and node.canonical_id in proof.ordered)
            ):
                certified.add(node.canonical_id)
            else:
                raise _partition_refusal(
                    graph,
                    spec,
                    "partition_effect_order_unproven",
                    f"Partition cannot certify mutation/alias ordering for {node.canonical_id!r}; "
                    "keep a supported write with its fresh producer and preserve effect order, "
                    "or use an out-of-place graph.",
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
    if any(name in {"dropout", "dropout_"} for name in names):
        training = kwargs.get("training", kwargs.get("train", args[2] if len(args) > 2 else None))
        if training is False:
            # Affirmatively disabled native dropout is an identity even when
            # the module requested its inplace spelling.
            return False
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


def plan_partition(graph: SplitTraceGraph, spec: SplitRequest) -> SplitPlan:
    """Validate a canonical partition and derive its unique live frontier."""

    _validate_replay_dependencies(graph)
    prefix_node_ids, suffix_node_ids = _partition_sets(graph, spec)
    ordered_mutations = _validate_partition_effects(graph, spec, prefix_node_ids=prefix_node_ids)
    boundary_node_ids = _frontier_node_ids(
        graph, prefix_node_ids=prefix_node_ids, suffix_node_ids=suffix_node_ids
    )
    boundary_spec, boundary_bindings = _boundary_metadata(
        graph, spec=spec, boundary_node_ids=boundary_node_ids
    )
    training_boundary_writes = tuple(
        node.canonical_id
        for node in graph.compute_nodes
        if node.canonical_id in ordered_mutations
        and node.canonical_id in suffix_node_ids
        and getattr(node.target, "__name__", "") != "batch_norm"
        and isinstance((arguments := _partition_call_arguments(node)[0]), tuple)
        and arguments
        and isinstance(arguments[0], ReplayValueRef)
        and arguments[0].value_id in prefix_node_ids
        and (dtype := graph.node_by_id[arguments[0].value_id].dtype) is not None
        and ("float" in dtype or "complex" in dtype)
    )
    return SplitPlan(
        split_id=_split_id(graph, spec),
        prefix_node_ids=prefix_node_ids,
        suffix_node_ids=suffix_node_ids,
        boundary_node_ids=boundary_node_ids,
        boundary_spec=boundary_spec,
        boundary_bindings=boundary_bindings,
        ordered_mutation_node_ids=ordered_mutations,
        training_boundary_writes=training_boundary_writes,
    )


__all__ = ["SplitPlan", "plan_partition"]
