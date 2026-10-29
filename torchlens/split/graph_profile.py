"""Immutable static graph costs for external partition and placement algorithms.

Logical tensor bytes are not allocator peaks. FLOPs count a multiply-add as two
operations and never represent latency. Unknown facts remain ``None``.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from math import prod
from typing import Any, TypeVar

from ..capture import flops as capture_flops
from ..intervention.types import CapturedArgTemplate, LiteralTensor, LiteralValue
from .errors import SplitUnsupportedError
from .graph import ReplayValueRef, SplitTraceGraph, SplitTraceNode
from .planner import SplitPlan

_Cost = TypeVar("_Cost", int, float)


@dataclass(frozen=True)
class ParameterCostFeatures:
    """One deduplicated parameter with stable named or occurrence provenance."""

    state_id: str
    aliases: tuple[str, ...]
    consumer_ids: tuple[str, ...]
    shape: tuple[int, ...] | None
    dtype: str | None
    bytes: int | None
    trainable: bool


@dataclass(frozen=True)
class NodeCostFeatures:
    """Compute-value node whose ID can be supplied directly to GraphPartition.

    ``output_node_ids`` identifies the indivisible native call. Its FLOPs are
    charged once on the first output row, with zero on other outputs.
    """

    node_id: str
    call_id: str
    op_type: str
    input_node_ids: tuple[str, ...]
    output_node_ids: tuple[str, ...]
    forward_flops: int | None
    backward_flops: int | None
    captured_forward_flops: int | None
    captured_backward_flops: int | None
    parameter_ids: tuple[str, ...]
    parameter_bytes: int | None
    activation_bytes: int | None
    trainable: bool


@dataclass(frozen=True)
class TensorCostFeatures:
    """One canonical graph value, counted once regardless of consumer count."""

    value_id: str
    producer_id: str | None
    consumer_ids: tuple[str, ...]
    shape: tuple[int, ...] | None
    dtype: str | None
    bytes: int | None
    requires_grad: bool | None
    container_path: tuple[Any, ...]
    alias_group: str | None = None


@dataclass(frozen=True)
class GraphProfile:
    """Static, payload-free costs tied to a captured graph and shape contract."""

    graph_identity: str | None
    backend: str
    batch_size: int | None
    shape_fingerprint: str | None
    nodes: tuple[NodeCostFeatures, ...]
    tensors: tuple[TensorCostFeatures, ...]
    parameters: tuple[ParameterCostFeatures, ...]
    unknown_cost_diagnostics: tuple[str, ...]
    backward_flops_estimated: bool = True

    @property
    def forward_flops(self) -> int | None:
        """Return a total only when every native call has a known cost."""
        return _sum_known(node.forward_flops for node in self.nodes)

    @property
    def backward_flops(self) -> int | None:
        """Return the capture engine's heuristic estimate, never measured time."""
        return _sum_known(node.backward_flops for node in self.nodes)


@dataclass(frozen=True)
class PartitionMetrics:
    """Unique logical costs; known FLOPs are a partial sum when totals are unknown."""

    prefix_node_ids: frozenset[str]
    suffix_node_ids: frozenset[str]
    unique_boundary_tensors: int
    forward_boundary_bytes: int | None
    differentiable_boundary_bytes: int | None
    backward_gradient_bytes: int | None
    parameter_bytes: int | None
    prefix_parameter_bytes: int | None
    suffix_parameter_bytes: int | None
    forward_flops: int | None
    prefix_forward_flops: int | None
    suffix_forward_flops: int | None
    known_forward_flops: int
    backward_flops: int | None
    prefix_backward_flops: int | None
    suffix_backward_flops: int | None
    unknown_cost_diagnostics: tuple[str, ...]


def dtype_bytes(dtype: str | None) -> int | None:
    """Return widths only for explicitly recognized dense tensor dtypes."""

    name = (dtype or "").removeprefix("torch.").removeprefix("paddle.").removeprefix("mlx.core.")
    widths = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
        "complex64": 8,
        "complex128": 16,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
    }
    return widths.get(name)


def tensor_bytes(shape: tuple[int, ...] | None, dtype: str | None) -> int | None:
    """Compute dense logical size from shape/dtype without touching payloads."""

    width = dtype_bytes(dtype)
    if shape is None or width is None or any(dim < 0 for dim in shape):
        return None
    return prod(shape) * width


def _may_require_gradient(dtype: str | None) -> bool:
    """Keep gradient requirements unknown for floating, complex and unrecognized dtypes."""

    return dtype_bytes(dtype) is None or (
        dtype is not None and ("float" in dtype or "complex" in dtype)
    )


def _sum_known(values: Iterable[_Cost | None]) -> _Cost | None:
    """Return an exact total only when every constituent cost is known."""

    total: _Cost = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _shape(value: Any) -> tuple[int, ...] | None:
    """Read shape metadata without materializing a backend tensor."""

    raw = getattr(value, "shape", None)
    try:
        return None if raw is None else tuple(int(dim) for dim in raw)
    except (TypeError, ValueError):
        return None


def _parameters(graph: SplitTraceGraph, model: Any) -> tuple[ParameterCostFeatures, ...]:
    """Join live handles by identity internally, exporting only stable provenance."""

    names: dict[int, list[str]] = {}
    named = getattr(model, "named_parameters", None)
    if callable(named):
        try:
            pairs = named(remove_duplicate=False)
        except TypeError:
            pairs = named()
        for name, value in pairs:
            names.setdefault(id(value), []).append(str(name))
    entries: dict[int, tuple[Any, list[str], str]] = {}
    for node in graph.compute_nodes:
        for index, param in enumerate(node.param_refs):
            value = getattr(param, "handle", None)
            if value is None:
                value = getattr(param, "_param_ref", None)
            # Missing live handles still retain a distinct occurrence cost row.
            key = id(param) if value is None else id(value)
            if key not in entries:
                entries[key] = (value, [], f"occurrence:{node.canonical_id}:parameter:{index}")
            consumers = entries[key][1]
            if node.canonical_id not in consumers:
                consumers.append(node.canonical_id)
    result = []
    for key, (value, consumers, fallback) in entries.items():
        aliases = tuple(sorted(set(names.get(key, ()))))
        shape, dtype = _shape(value), str(getattr(value, "dtype", "")) or None
        result.append(
            ParameterCostFeatures(
                state_id=f"parameter:{aliases[0]}" if aliases else fallback,
                aliases=aliases,
                consumer_ids=tuple(consumers),
                shape=shape,
                dtype=dtype,
                bytes=tensor_bytes(shape, dtype),
                trainable=bool(getattr(value, "requires_grad", False)),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class _TensorMetadata:
    """Tensor-shaped argument with no allocation, value access or autograd state."""

    shape: tuple[int, ...]
    dtype: str | None = None


def _cost_tensor_metadata(
    component: ReplayValueRef | LiteralTensor,
    shapes: dict[str, tuple[int, ...] | None],
    graph: SplitTraceGraph,
) -> _TensorMetadata:
    """Resolve a tensor argument's shape and dtype without reading its payload."""
    if isinstance(component, ReplayValueRef):
        shape = shapes.get(component.value_id)
        if shape is None:
            raise ValueError("argument shape unavailable")
        dtype = graph.node_by_id[component.value_id].dtype
    else:
        shape = _shape(component.value)
        if shape is None:
            raise ValueError("literal shape unavailable")
        dtype = str(getattr(component.value, "dtype", "")) or None
    return _TensorMetadata(shape, dtype)


def _cost_argument(
    component: Any, shapes: dict[str, tuple[int, ...] | None], graph: SplitTraceGraph
) -> Any:
    """Decode replay templates once into pure shape/scalar/container metadata."""
    if (
        isinstance(component, LiteralValue)
        or component is None
        or isinstance(component, (int, float, bool, str))
    ):
        return component.value if isinstance(component, LiteralValue) else component
    if isinstance(component, (ReplayValueRef, LiteralTensor)):
        return _cost_tensor_metadata(component, shapes, graph)
    if isinstance(component, tuple):
        if component and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in component
        ):
            return {key: _cost_argument(value, shapes, graph) for key, value in component}
        return tuple(_cost_argument(value, shapes, graph) for value in component)
    if isinstance(component, list):
        return [_cost_argument(value, shapes, graph) for value in component]
    if isinstance(component, dict):
        return {key: _cost_argument(value, shapes, graph) for key, value in component.items()}
    raise ValueError("argument metadata unavailable")


def _call_cost(
    node: SplitTraceNode,
    shapes: dict[str, tuple[int, ...] | None],
    graph: SplitTraceGraph,
    batch: int | None,
) -> tuple[int | None, int | None]:
    """Call the capture FLOPs authority with shape-only replay arguments."""
    shape = shapes.get(node.canonical_id)
    if shape is None:
        return None, None
    name = node.cost_name or str(getattr(node.target, "__name__", node.op_type))
    template = node.args_template
    if not isinstance(template, CapturedArgTemplate):
        return None, None
    try:
        args = tuple(_cost_argument(item, shapes, graph) for item in template.args)
        kwargs = {key: _cost_argument(item, shapes, graph) for key, item in template.kwargs}
        if isinstance(node.kwargs_template, dict):
            kwargs.update(
                {
                    key: _cost_argument(item, shapes, graph)
                    for key, item in node.kwargs_template.items()
                }
            )
        program = graph.shape_program
        if program is not None and batch is not None and program.has_batch_axes:
            binding = program.binding_from_batch(batch)
            args = program.rewrite(node.canonical_id, args, binding)
            kwargs = program.rewrite(node.canonical_id, kwargs, binding)
        forward = capture_flops.compute_forward_flops(
            name, shape, node.cost_param_shapes, args, kwargs
        )
        return forward, capture_flops.compute_backward_flops(name, forward)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError):
        # Custom rules requiring payloads cannot be evaluated by a shape-only profile.
        return None, None


def _profile_shapes(
    graph: SplitTraceGraph, batch: int | None
) -> tuple[dict[str, tuple[int, ...] | None], list[str]]:
    """Evaluate static shapes, preserving unresolved batch evidence as unknown."""

    shapes: dict[str, tuple[int, ...] | None] = {}
    unknown: list[str] = []
    program = graph.shape_program
    extrapolate = batch != graph.traced_batch_size
    binding = None
    if extrapolate and program is not None and batch is not None:
        try:
            program.require_batch_resolvable(
                batch, backend=graph.backend, split_point="graph_profile"
            )
            if program.has_batch_axes:
                binding = program.binding_from_batch(batch)
        except (ValueError, SplitUnsupportedError):
            unknown.append("batch_shape:extrapolation_unproven")
    for node in graph.nodes:
        shape = node.output_shape
        if extrapolate:
            shape = None
            if program is not None and binding is not None:
                with suppress(ValueError):
                    shape = program.value_shape(node.canonical_id, binding)
        shapes[node.canonical_id] = shape
    return shapes, unknown


def build_graph_profile(
    graph: SplitTraceGraph, *, model: Any = None, batch_size: int | None = None
) -> GraphProfile:
    """Build costs without capture, replay, payload reads or device synchronization.

    Parameters
    ----------
    graph:
        Normalized graph reused by native segment execution.
    model:
        Optional model supplying named parameter provenance, including aliases.
    batch_size:
        Positive batch extent. Unproven shape extrapolations stay unknown.
    """

    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    batch = graph.traced_batch_size if batch_size is None else batch_size
    shapes, unknown = _profile_shapes(graph, batch)
    program = graph.shape_program
    parameters = _parameters(graph, model)
    parameters_by_consumer: dict[str, list[ParameterCostFeatures]] = {}
    for parameter in parameters:
        for consumer in parameter.consumer_ids:
            parameters_by_consumer.setdefault(consumer, []).append(parameter)
    consumers = _value_consumers(graph)
    tensors = []
    for node in graph.nodes:
        size = tensor_bytes(shapes[node.canonical_id], node.dtype)
        if size is None:
            unknown.append(f"tensor_bytes:{node.canonical_id}")
        tensors.append(
            TensorCostFeatures(
                value_id=node.canonical_id,
                producer_id=None
                if node.is_input or node.is_buffer_only_source
                else node.canonical_id,
                consumer_ids=tuple(consumers[node.canonical_id]),
                shape=shapes[node.canonical_id],
                dtype=node.dtype,
                bytes=size,
                requires_grad=node.requires_grad,
                container_path=node.output_container_path,
            )
        )
    tensor_by_id = {item.value_id: item for item in tensors}
    nodes = []
    for node in graph.compute_nodes:
        call = graph.replay_call_by_output_id[node.canonical_id]
        forward, backward = (
            _call_cost(node, shapes, graph, batch)
            if node.canonical_id == call.output_node_ids[0]
            else (0, 0)
        )
        used = parameters_by_consumer.get(node.canonical_id, [])
        if forward is None:
            unknown.append(f"flops:{node.canonical_id}")
        if (
            batch == graph.traced_batch_size
            and node.canonical_id == call.output_node_ids[0]
            and node.cost_name not in capture_flops._CUSTOM_OP_RULES
            and node.captured_forward_flops is not None
            and forward != node.captured_forward_flops
        ):
            unknown.append(f"flops_capture_mismatch:{node.canonical_id}")
        nodes.append(
            NodeCostFeatures(
                node_id=node.canonical_id,
                call_id=call.call_id,
                op_type=node.op_type,
                input_node_ids=tuple(
                    graph.node_id_by_alias.get(parent, parent) for parent in node.parents
                ),
                output_node_ids=call.output_node_ids,
                forward_flops=forward,
                backward_flops=backward,
                captured_forward_flops=node.captured_forward_flops,
                captured_backward_flops=node.captured_backward_flops,
                parameter_ids=tuple(param.state_id for param in used),
                parameter_bytes=_sum_known(param.bytes for param in used),
                activation_bytes=tensor_by_id[node.canonical_id].bytes,
                trainable=any(param.trainable for param in used) or node.requires_grad is True,
            )
        )
    unknown.extend(
        f"parameter_bytes:{param.state_id}" for param in parameters if param.bytes is None
    )
    return GraphProfile(
        graph_identity=graph.graph_shape_hash,
        backend=graph.backend,
        batch_size=batch,
        shape_fingerprint=None if program is None else program.fingerprint,
        nodes=tuple(nodes),
        tensors=tuple(tensors),
        parameters=parameters,
        unknown_cost_diagnostics=tuple(unknown),
    )


def _value_consumers(graph: SplitTraceGraph) -> dict[str, list[str]]:
    """Collect each value's distinct consumers in captured graph order."""

    consumers: dict[str, list[str]] = {node.canonical_id: [] for node in graph.nodes}
    for node in graph.nodes:
        for parent_label in node.parents:
            parent = graph.node_for_label(parent_label)
            if parent is not None and node.canonical_id not in consumers[parent.canonical_id]:
                consumers[parent.canonical_id].append(node.canonical_id)
    return consumers


def partition_metrics(
    profile: GraphProfile, plan: SplitPlan, *, training: bool
) -> PartitionMetrics:
    """Project immutable graph costs onto one validated split without double counting."""

    tensors = {tensor.value_id: tensor for tensor in profile.tensors}
    boundary = tuple(tensors[node_id] for node_id in plan.boundary_node_ids)
    differentiable = _sum_known(tensor.bytes for tensor in boundary if tensor.requires_grad)
    if any(tensor.requires_grad is None for tensor in boundary):
        differentiable = None
    prefix = tuple(node for node in profile.nodes if node.node_id in plan.prefix_node_ids)
    suffix = tuple(node for node in profile.nodes if node.node_id in plan.suffix_node_ids)
    params = profile.parameters
    diagnostics = list(profile.unknown_cost_diagnostics)
    # A static requires_grad flag does not prove the selected loss uses a boundary
    # leaf. Communication bytes for backward need an actual training contract.
    backward_unknown = training and any(_may_require_gradient(t.dtype) for t in boundary)
    if backward_unknown:
        diagnostics.append("backward_gradient_bytes:loss_path_not_observed")
    return PartitionMetrics(
        prefix_node_ids=plan.prefix_node_ids,
        suffix_node_ids=plan.suffix_node_ids,
        unique_boundary_tensors=len(boundary),
        forward_boundary_bytes=_sum_known(t.bytes for t in boundary),
        differentiable_boundary_bytes=differentiable,
        backward_gradient_bytes=None if backward_unknown else 0,
        parameter_bytes=_sum_known(p.bytes for p in params),
        prefix_parameter_bytes=_sum_known(
            p.bytes for p in params if set(p.consumer_ids) & plan.prefix_node_ids
        ),
        suffix_parameter_bytes=_sum_known(
            p.bytes for p in params if set(p.consumer_ids) & plan.suffix_node_ids
        ),
        forward_flops=_sum_known(n.forward_flops for n in profile.nodes),
        prefix_forward_flops=_sum_known(n.forward_flops for n in prefix),
        suffix_forward_flops=_sum_known(n.forward_flops for n in suffix),
        known_forward_flops=sum(
            n.forward_flops for n in profile.nodes if n.forward_flops is not None
        ),
        backward_flops=_sum_known(n.backward_flops for n in profile.nodes),
        prefix_backward_flops=_sum_known(n.backward_flops for n in prefix),
        suffix_backward_flops=_sum_known(n.backward_flops for n in suffix),
        unknown_cost_diagnostics=tuple(diagnostics),
    )
