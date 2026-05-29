"""Generated eager prefix and suffix modules for TorchLens split runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..intervention.types import (
    DataclassField,
    DictKey,
    HFKey,
    LiteralTensor,
    LiteralValue,
    NamedField,
    ParentRef,
    TupleIndex,
    Unsupported,
)
from ..utils.rng import execute_with_restored_rng_autocast

from .boundary import ReplayBoundary
from .errors import SplitErrorContext, SplitUnsupportedError, unsupported
from .frontier import SplitPlan
from .trace_graph import TraceGraph, TraceNode


@dataclass
class GeneratedPrefix(nn.Module):
    """Eager prefix segment lowered from a TraceGraph."""

    graph: TraceGraph
    plan: SplitPlan
    detach_outputs: bool = True

    def __post_init__(self) -> None:
        nn.Module.__init__(self)

    def forward(self, *inputs: Any) -> ReplayBoundary:
        """Execute prefix nodes and return a ReplayBoundary."""

        env = _input_env(self.graph, inputs)
        for label in self.plan.prefix_nodes:
            node = self.graph.get(label)
            if node.is_input:
                continue
            _execute_node(node, self.graph, env, split_point=self.plan.split_id)
        tensors: dict[str, torch.Tensor] = {}
        for label in self.plan.boundary_nodes:
            tensor = env[label]
            if not isinstance(tensor, torch.Tensor):
                raise SplitUnsupportedError(f"Boundary node {label!r} did not produce a tensor.")
            if self.detach_outputs:
                tensor = tensor.detach()
            tensors[label] = tensor
        metadata = {
            "split_id": self.plan.split_id,
            "split_label": self.plan.split_label,
            "graph_shape_hash": self.graph.graph_shape_hash,
            "batch_size": _first_batch_size(tensors.values()) or _first_batch_size(inputs),
            "boundary_order": self.plan.boundary_nodes,
        }
        return ReplayBoundary(tensors=tensors, spec=self.plan.boundary_specs, metadata=metadata)


@dataclass
class GeneratedSuffix(nn.Module):
    """Eager suffix segment lowered from a TraceGraph."""

    graph: TraceGraph
    plan: SplitPlan

    def __post_init__(self) -> None:
        nn.Module.__init__(self)

    def forward(self, boundary: ReplayBoundary) -> Any:
        """Execute suffix nodes from a boundary and return model output."""

        env: dict[str, Any] = dict(boundary.tensors)
        for label in self.plan.suffix_nodes:
            _execute_node(self.graph.get(label), self.graph, env, split_point=self.plan.split_id)
        for label in self.graph.output_nodes:
            _execute_node(self.graph.get(label), self.graph, env, split_point=self.plan.split_id)
        return _reconstruct_model_output(self.graph, env)


def _execute_node(
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    *,
    split_point: str,
) -> None:
    if node.is_input:
        return
    if node.torchlens_label in env and node.target is None and not node.parents:
        return
    activation = getattr(node.layer, "activation", None)
    if (
        not getattr(node.layer, "has_input_ancestor", True)
        and isinstance(activation, torch.Tensor)
    ):
        env[node.torchlens_label] = activation
        return
    if node.is_output or node.target is None:
        if not node.parents:
            raise unsupported(_context(node, split_point, "node has no target or parent"))
        env[node.torchlens_label] = env[node.parents[0]]
        return
    args, kwargs = _resolve_call_args(node, graph, env, split_point=split_point)
    try:
        output = execute_with_restored_rng_autocast(
            node.target,
            args,
            kwargs,
            rng_states=getattr(node.layer, "func_rng_states", None),
            autocast_state=getattr(node.layer, "func_autocast_state", None),
        )
    except Exception as exc:  # pragma: no cover - message is validated through callers.
        raise unsupported(_context(node, split_point, f"operation failed: {exc}")) from exc
    if output is None and getattr(node.layer, "func_is_inplace", False) and args:
        output = args[0]
    env[node.torchlens_label] = _slice_output_by_path(output, tuple(node.layer.output_path or ()))


def _resolve_call_args(
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    *,
    split_point: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    param_state = {"index": 0}
    args = tuple(
        _resolve_component(
            component,
            node,
            graph,
            env,
            param_state,
            split_point=split_point,
            path=("args", index),
        )
        for index, component in enumerate(node.args)
    )
    kwargs = {
        key: _resolve_component(
            component,
            node,
            graph,
            env,
            param_state,
            split_point=split_point,
            path=("kwargs", key),
        )
        for key, component in node.kwargs.items()
    }
    return args, kwargs


def _resolve_component(
    component: Any,
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    param_state: dict[str, int],
    *,
    split_point: str,
    path: tuple[Any, ...],
) -> Any:
    if isinstance(component, ParentRef):
        label = _final_label_for_ref(graph, component.parent_label)
        if label not in env:
            raise unsupported(_context(node, split_point, f"parent {label!r} is unavailable"))
        return env[label]
    if isinstance(component, LiteralTensor):
        param = _next_matching_param(node, component.value, param_state)
        if param is not None:
            return param
        return component.value
    if isinstance(component, LiteralValue):
        runtime_batch = _dynamic_batch_literal(component.value, node, graph, env, path)
        if runtime_batch is not None:
            return runtime_batch
        return component.value
    if isinstance(component, Unsupported):
        if component.value_type == "ellipsis":
            return Ellipsis
        raise unsupported(_context(node, split_point, component.reason))
    if isinstance(component, tuple):
        if _looks_like_template_dict(component):
            return {
                key: _resolve_component(
                    value,
                    node,
                    graph,
                    env,
                    param_state,
                    split_point=split_point,
                    path=(*path, key),
                )
                for key, value in component
            }
        return tuple(
            _resolve_component(
                value,
                node,
                graph,
                env,
                param_state,
                split_point=split_point,
                path=(*path, index),
            )
            for index, value in enumerate(component)
        )
    return component


_BATCH_SHAPE_OPS = frozenset(
    {
        "view",
        "reshape",
        "flatten",
        "expand",
        "repeat",
        "zeros",
        "ones",
        "empty",
        "arange",
        "meshgrid",
    }
)


def _dynamic_batch_literal(
    value: Any,
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    path: tuple[Any, ...],
) -> int | None:
    """Replace trace-batch literals in leading shape positions with runtime B."""

    traced_batch = graph.traced_batch_size
    if traced_batch is None or value != traced_batch:
        return None
    if node.op_type not in _BATCH_SHAPE_OPS:
        return None
    if not _is_leading_shape_position(node.op_type, path):
        return None
    runtime_batch = _runtime_batch_from_env(env)
    if runtime_batch is None:
        raise unsupported(_context(node, graph.batch_symbol, "could not infer runtime batch"))
    return runtime_batch


def _is_leading_shape_position(op_type: str, path: tuple[Any, ...]) -> bool:
    """Return whether a literal path is the leading batch dimension of a shape arg."""

    if len(path) < 2 or path[0] not in {"args", "kwargs"}:
        return False
    if op_type in {"view", "reshape", "expand", "repeat"}:
        return path in {("args", 1), ("args", 1, 0), ("kwargs", "shape", 0), ("kwargs", "size", 0)}
    if op_type in {"zeros", "ones", "empty"}:
        return path in {("args", 0), ("args", 0, 0), ("kwargs", "size", 0)}
    return False


def _runtime_batch_from_env(env: dict[str, Any]) -> int | None:
    for value in env.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _next_matching_param(
    node: TraceNode,
    value: Any,
    param_state: dict[str, int],
) -> torch.nn.Parameter | None:
    logs = list(getattr(node.layer, "parent_param_logs", []) or [])
    index = param_state["index"]
    if index >= len(logs) or not isinstance(value, torch.Tensor):
        return None
    param = getattr(logs[index], "_param_ref", None)
    if param is None:
        return None
    if tuple(param.shape) != tuple(value.shape) or param.dtype != value.dtype:
        return None
    param_state["index"] = index + 1
    return param


def _input_env(graph: TraceGraph, inputs: tuple[Any, ...]) -> dict[str, Any]:
    leaves = list(_walk_input_leaves(inputs))
    if len(leaves) < len(graph.input_nodes):
        raise SplitUnsupportedError(
            f"Expected at least {len(graph.input_nodes)} tensor input(s), got {len(leaves)}."
        )
    env: dict[str, Any] = {}
    used: set[int] = set()
    for index, label in enumerate(graph.input_nodes):
        node = graph.get(label)
        match_index = _match_input_leaf(node, leaves, used)
        if match_index is None:
            match_index = index
        env[label] = leaves[match_index]
        used.add(match_index)
    for label, node in graph.nodes.items():
        activation = getattr(node.layer, "activation", None)
        if (
            label not in env
            and node.target is None
            and not node.parents
            and isinstance(activation, torch.Tensor)
        ):
            env[label] = activation
    return env


def _match_input_leaf(
    node: TraceNode,
    leaves: list[Any],
    used: set[int],
) -> int | None:
    """Find the runtime input tensor matching a traced input node."""

    for index, leaf in enumerate(leaves):
        if index in used or not isinstance(leaf, torch.Tensor):
            continue
        if node.dtype is not None and leaf.dtype != node.dtype:
            continue
        if node.output_shape is not None and tuple(leaf.shape) != tuple(node.output_shape):
            continue
        return index
    return None


def _walk_input_leaves(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        yield value
        return
    nested_tensors = getattr(value, "tensors", None)
    nested_mask = getattr(value, "mask", None)
    if isinstance(nested_tensors, torch.Tensor):
        yield nested_tensors
        if isinstance(nested_mask, torch.Tensor):
            yield nested_mask
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_input_leaves(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_input_leaves(item)
        return


def _slice_output_by_path(output: Any, path: tuple[Any, ...]) -> Any:
    current = output
    for component in path:
        current = _index_output_component(current, component)
    return current


def _index_output_component(output: Any, component: Any) -> Any:
    if isinstance(component, TupleIndex):
        return output[component.index]
    if isinstance(component, DictKey):
        return output[component.key]
    if isinstance(component, NamedField):
        return getattr(output, component.name)
    if isinstance(component, DataclassField):
        return getattr(output, component.name)
    if isinstance(component, HFKey):
        return output[component.key]
    if isinstance(component, int):
        return output[component]
    if isinstance(component, str):
        if isinstance(output, Mapping) or hasattr(output, "keys"):
            return output[component]
        return getattr(output, component)
    raise SplitUnsupportedError(f"Unsupported output path component {component!r}.")


def _reconstruct_model_output(graph: TraceGraph, env: dict[str, Any]) -> Any:
    output_nodes = [graph.get(label) for label in graph.output_nodes]
    if not output_nodes:
        raise SplitUnsupportedError("TraceGraph has no output nodes.")
    if len(output_nodes) == 1 and not getattr(output_nodes[0].layer, "container_spec", None):
        return env[output_nodes[0].torchlens_label]
    root_spec = getattr(output_nodes[0].layer, "container_spec", None)
    if root_spec is None:
        return tuple(env[node.torchlens_label] for node in output_nodes)
    values = {tuple(node.layer.output_path or ()): env[node.torchlens_label] for node in output_nodes}
    return _materialize_container(root_spec, (), values)


def _materialize_container(spec: Any, path: tuple[Any, ...], values: dict[tuple[Any, ...], Any]) -> Any:
    if path in values:
        return values[path]
    child_specs = dict(getattr(spec, "child_specs", ()) or ())
    if spec.kind == "tuple":
        return tuple(_materialize_child(index, child_specs, path, values) for index in range(spec.length))
    if spec.kind == "list":
        return [_materialize_child(index, child_specs, path, values) for index in range(spec.length)]
    if spec.kind in {"dict", "hf_model_output"}:
        return {
            key: _materialize_child(key, child_specs, path, values)
            for key in getattr(spec, "keys", ()) or ()
        }
    if spec.kind == "namedtuple":
        return tuple(
            _materialize_child(field, child_specs, path, values)
            for field in getattr(spec, "fields", ()) or ()
        )
    if spec.kind == "dataclass":
        return {
            field: _materialize_child(field, child_specs, path, values)
            for field in getattr(spec, "fields", ()) or ()
        }
    raise SplitUnsupportedError(f"Unsupported output container kind {spec.kind!r}.")


def _materialize_child(
    key: Any,
    child_specs: dict[Any, Any],
    path: tuple[Any, ...],
    values: dict[tuple[Any, ...], Any],
) -> Any:
    component = _component_for_key(key, child_specs)
    child_path = (*path, component)
    child_spec = child_specs.get(component)
    if child_spec is None:
        return values[child_path]
    return _materialize_container(child_spec, child_path, values)


def _component_for_key(key: Any, child_specs: dict[Any, Any]) -> Any:
    for component in child_specs:
        if isinstance(component, TupleIndex) and component.index == key:
            return component
        if isinstance(component, DictKey) and component.key == key:
            return component
        if isinstance(component, HFKey) and component.key == key:
            return component
        if isinstance(component, NamedField) and component.name == key:
            return component
        if isinstance(component, DataclassField) and component.name == key:
            return component
    if isinstance(key, int):
        return TupleIndex(key)
    return DictKey(key)


def _looks_like_template_dict(component: tuple[Any, ...]) -> bool:
    return bool(component) and all(isinstance(item, tuple) and len(item) == 2 for item in component)


def _final_label_for_ref(graph: TraceGraph, label: str) -> str:
    if label in graph.nodes:
        return label
    return str(getattr(graph.model_log, "_raw_to_final_layer_labels", {}).get(label, label))


def _first_batch_size(values: Any) -> int | None:
    for value in values:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (tuple, list)):
            nested = _first_batch_size(value)
            if nested is not None:
                return nested
    return None


def _context(node: TraceNode, split_point: str, reason: str) -> SplitErrorContext:
    return SplitErrorContext(
        split_point=split_point,
        module_path=node.module_path,
        op_type=node.op_type,
        layer_label=node.torchlens_label,
        traced_shape=node.output_shape,
        dtype=node.dtype,
        reason=reason,
    )


__all__ = ["GeneratedPrefix", "GeneratedSuffix"]
