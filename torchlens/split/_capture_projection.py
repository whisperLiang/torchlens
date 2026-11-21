"""Project upstream capture records into the existing Split Runtime contracts.

This is the sole record-schema boundary. It reads the upstream physical slots
and grouped facts once; executors retain only the data they actually consume.
It neither emulates a Trace nor captures or executes model code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .._ir.container import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    TupleIndex,
)
from .._utils.collections import _path_component_key

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .graph import ReplaySourcePolicy, SplitTraceNode


@dataclass(frozen=True, slots=True)
class _CapturedOp:
    """Execution facts retained independently of the upstream Op facade."""

    _label_raw: str | None
    address: str | None
    func_rng_states: dict[str, Any] | None
    func_autocast_state: dict[str, Any] | None
    multi_output_index: Any
    is_inplace: bool
    out: Any
    device_ref: Any
    parent_arg_positions: Any
    saved_args: Any
    saved_kwargs: Any


def capture_batch_size(trace: Trace) -> int | None:
    """Infer the capture batch from the complete upstream boundary row sequence."""

    outputs = set(trace.output_layers)
    output_dims = {int(op.shape[0]) for op in trace._layer_list if op.label in outputs and op.shape}
    inputs = [
        int(op.shape[0]) for op in trace._layer_list if op._slot("is_input", False) and op.shape
    ]
    return next((dim for dim in inputs if dim in output_dims), inputs[0] if inputs else None)


def _capture_execution_state(op: Op) -> _CapturedOp:
    """Read the retained execution subset through one upstream call view."""

    call = op.call
    slot = op._slot
    return _CapturedOp(
        _label_raw=op._label_raw,
        address=op.buffer_address,
        func_rng_states=call.rng_states,
        func_autocast_state=call.autocast_state,
        multi_output_index=slot("multi_output_index"),
        is_inplace=bool(slot("is_inplace", False)),
        out=slot("out"),
        device_ref=slot("device_ref"),
        parent_arg_positions=call.parent_arg_positions,
        saved_args=call.saved_args,
        saved_kwargs=call.saved_kwargs,
    )


def project_op(op: Op, *, backend: str, traced_batch_size: int | None) -> SplitTraceNode:
    """Normalize one captured Op, preserving native-call and state identities."""

    from .graph import SplitTraceNode
    from .shape import symbolic_shape_from_tensor_ref

    slot, call, cost = op._slot, op.call, op.cost
    shape = None if op.shape is None else tuple(int(dim) for dim in op.shape)
    parents, params = tuple(op.parents), tuple(op.params)
    is_buffer = bool(op.graph.is_buffer)
    is_param_source = bool(slot("input_was_parameter", False))
    stack, address = op.module_context.call_stack, op.buffer_address
    module_path = address or (str(stack[-1]).rsplit(":", 1)[0] if stack else None)
    state = _capture_execution_state(op)
    forward_flops, backward_flops = cost.flops_forward, cost.flops_backward
    policy: ReplaySourcePolicy = "constant"
    if params:
        policy = "live_param"
    elif is_param_source:
        policy = "live_param_derived"
    elif shape and traced_batch_size is not None:
        policy = "batch_dynamic_constant"
    return SplitTraceNode(
        label=op.label,
        raw_label=op._label_raw,
        canonical_id=f"v:{op.raw_index}",
        backend=backend,
        raw_index=op.raw_index,
        op_type=op.type,
        target=slot("func"),
        func_call_id=call.call_id,
        args_template=call.args_template,
        kwargs_template=call.kwargs_template,
        parents=parents,
        children=tuple(op.children),
        output_ref=slot("out_ref"),
        module_path=module_path,
        output_shape=shape,
        symbolic_output_shape=symbolic_shape_from_tensor_ref(shape),
        dtype=None if op.dtype is None else str(op.dtype),
        requires_grad=getattr(state.out, "requires_grad", None),
        output_container_path=tuple(slot("container_path", ()) or ()),
        output_container_spec=slot("container_spec"),
        is_input=bool(slot("is_input", False)),
        is_output=bool(slot("is_output", False)),
        is_buffer=is_buffer,
        is_buffer_only_source=is_buffer and not parents and op.buffer_write_kind is None,
        is_param_source=is_param_source,
        param_refs=params,
        replay_source_policy=policy,
        op=state,
        cost_name=op.func_name,
        cost_param_shapes=tuple(tuple(s) for s in (slot("param_shapes", ()) or ())),
        captured_forward_flops=None if forward_flops is None else int(forward_flops),
        captured_backward_flops=None if backward_flops is None else int(backward_flops),
    )


def _captures_by_raw_label(captures: Iterable[Any]) -> dict[str, Any]:
    """Index native captures that carry an upstream raw operation label."""

    return {
        str(label): capture
        for capture in captures
        if (label := getattr(capture, "label_raw", None)) is not None
    }


def _attach_capture_targets(
    nodes: list[SplitTraceNode], captures: dict[str, Any]
) -> list[SplitTraceNode]:
    """Attach each sidecar to the projected nodes with its raw operation label."""

    if not captures:
        return nodes
    updated = []
    for node in nodes:
        label = str(getattr(node.op, "_label_raw", node.raw_label or ""))
        capture = captures.get(label)
        updated.append(replace(node, target=capture) if capture is not None else node)
    return updated


def _leaf_paths(spec: ContainerSpec, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Enumerate the upstream container's leaf slots in reconstruction order."""

    if spec.kind in {"literal", "opaque"}:
        return []
    children = dict(spec.child_specs)
    components: Sequence[Any]
    if spec.kind in {"tuple", "list"}:
        components = [TupleIndex(index) for index in range(spec.length or 0)]
    elif spec.kind in {"dict", "hf_model_output"}:
        factory = DictKey if spec.kind == "dict" else HFKey
        components = [factory(key) for key in spec.keys]
    elif spec.kind in {"namedtuple", "dataclass", "distribution"}:
        field = DataclassField if spec.kind == "dataclass" else NamedField
        components = [field(name) for name in spec.fields]
    elif spec.kind == "registered":
        components = [component for component, _child in spec.child_specs]
    else:
        raise ValueError(f"Unsupported output container kind {spec.kind!r}.")
    result: list[tuple[Any, ...]] = []
    for component in components:
        path = (*prefix, _path_component_key(component))
        child = children.get(component)
        result.extend([path] if child is None else _leaf_paths(child, path))
    return result


def reorder_container_leaves(
    spec: ContainerSpec, leaves: Sequence[tuple[tuple[Any, ...], Any]]
) -> list[Any]:
    """Order split outputs by upstream container slots and refuse mismatched paths."""

    values = {
        tuple(_path_component_key(component) for component in path): value for path, value in leaves
    }
    expected = _leaf_paths(spec)
    if len(values) != len(leaves) or len(values) != len(expected) or set(values) != set(expected):
        raise ValueError(
            f"ContainerSpec leaf paths do not match captured output paths: "
            f"expected={expected!r}, actual={tuple(values)!r}."
        )
    return [values[path] for path in expected]


def _paddle_capture_parameter_refs(capture: Any, params_by_id: dict[int, Any]) -> tuple[Any, ...]:
    """Return exactly the native parameters consumed by this captured Paddle call.

    Parameters
    ----------
    capture
        Session-only Paddle operation capture with live unlabeled tensor operands.
    params_by_id
        Native parameter records indexed by their live handle's identity.

    Returns
    -------
    tuple[Any, ...]
        Unique consumed parameter records in argument order.
    """

    consumed = {
        id(value): params_by_id[id(value)]
        for value in capture.param_inputs.values()
        if id(value) in params_by_id
    }
    return tuple(consumed.values())


def paddle_call_metadata(
    capture: Any,
) -> tuple[tuple[Any, ...], dict[str, Any], tuple[tuple[Any, ...], ...]]:
    """Bind the Paddle call's attested state operands and native output paths.

    Parameters
    ----------
    capture
        Upstream runtime call record with path-addressed ``param_inputs``.

    Returns
    -------
    tuple
        Detached argument containers and the emitted native tensor-output paths.
        Unlabeled parameter slots retain their attested live handles. Unattested
        slots stay unresolved. Model-output occurrence paths are independent:
        upstream output marking can rebind the producing Op's container path.
    """

    sources = {**capture.buffer_inputs, **capture.param_inputs}

    def bind(value: Any, path: tuple[Any, ...]) -> Any:
        """Project one template occurrence using its recorded native operand path."""

        if isinstance(value, dict) and value.get("kind") == "tensor":
            result = dict(value)
            if result.get("label") is None and path in sources:
                result["value"] = sources[path]
            return result
        if isinstance(value, tuple):
            return tuple(bind(child, (*path, index)) for index, child in enumerate(value))
        if isinstance(value, list):
            return [bind(child, (*path, index)) for index, child in enumerate(value)]
        if isinstance(value, dict):
            return {key: bind(child, (*path, key)) for key, child in value.items()}
        return value

    aliases = {tuple(annotation["output_path"]) for annotation in capture.alias_annotations}
    return (
        bind(capture.args_template, ("args",)),
        bind(capture.kwargs_template, ("kwargs",)),
        tuple(tuple(path) for path in capture.output_leaf_paths if tuple(path) not in aliases),
    )
