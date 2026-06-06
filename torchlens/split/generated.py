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
from .shape import MAX_BATCH_MULTIPLIER
from .trace_graph import BATCH_DYNAMIC_CONSTANT_OPS, TraceGraph, TraceNode

_RNG_SENSITIVE_OP_NAMES = frozenset(
    {
        "bernoulli",
        "bernoulli_",
        "dropout",
        "dropout_",
        "dropout1d",
        "dropout2d",
        "dropout3d",
        "alpha_dropout",
        "alpha_dropout_",
        "alphadropout",
        "feature_alpha_dropout",
        "feature_alpha_dropout_",
        "featurealphadropout",
        "feature_dropout",
        "feature_dropout_",
        "gumbel_softmax",
        "multinomial",
        "native_dropout",
        "normal",
        "normal_",
        "poisson",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "rrelu",
        "rrelu_",
    }
)
_LIVE_PARAM_SOURCES_ENV_KEY = "__torchlens_live_param_sources__"
_LIVE_PARAM_SOURCES_METADATA_KEY = "use_live_param_sources"


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
            _LIVE_PARAM_SOURCES_METADATA_KEY: self.plan.use_live_param_sources,
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
        if boundary.metadata.get(
            _LIVE_PARAM_SOURCES_METADATA_KEY,
            self.plan.use_live_param_sources,
        ):
            env[_LIVE_PARAM_SOURCES_ENV_KEY] = True
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
    if env.get(_LIVE_PARAM_SOURCES_ENV_KEY, False) and node.replay_source_policy == "live_param":
        live_source = _live_source_value(node, graph)
        if live_source is not None:
            env[node.torchlens_label] = live_source
            return
    activation = getattr(node.layer, "activation", None)
    if _can_reuse_trace_activation(node, activation, env):
        env[node.torchlens_label] = activation
        return
    if node.is_output or node.target is None:
        if not node.parents:
            raise unsupported(_context(node, split_point, "node has no target or parent"))
        env[node.torchlens_label] = env[node.parents[0]]
        return
    args, kwargs = _resolve_call_args(node, graph, env, split_point=split_point)
    try:
        output = _call_node_target(node, args, kwargs)
    except Exception as exc:  # pragma: no cover - message is validated through callers.
        raise unsupported(_context(node, split_point, f"operation failed: {exc}")) from exc
    if output is None and getattr(node.layer, "func_is_inplace", False) and args:
        output = args[0]
    env[node.torchlens_label] = _slice_output_by_path(output, tuple(node.layer.output_path or ()))


def _can_reuse_trace_activation(
    node: TraceNode,
    activation: Any,
    env: dict[str, Any],
) -> bool:
    """Return whether suffix replay may reuse a trace-time no-input activation."""

    if node.replay_source_policy == "batch_dynamic_constant":
        return False
    if env.get(_LIVE_PARAM_SOURCES_ENV_KEY, False) and node.replay_source_policy in {
        "live_param",
        "live_param_derived",
    }:
        return False
    return (
        not getattr(node.layer, "has_input_ancestor", True)
        and isinstance(activation, torch.Tensor)
        and node.op_type not in _BATCH_SHAPE_OPS
        and not node.parents
        and not _has_parent_refs((node.args, node.kwargs))
    )


def _live_source_value(node: TraceNode, graph: TraceGraph) -> torch.Tensor | None:
    """Return a live module tensor for source nodes that have no callable target."""

    if bool(getattr(node.layer, "is_buffer_layer", False)):
        return _live_buffer_value(node, graph)
    return None


def _live_buffer_value(node: TraceNode, graph: TraceGraph) -> torch.Tensor | None:
    """Resolve a current model buffer/plain tensor by TorchLens buffer address."""

    address = getattr(node.layer, "buffer_address", None)
    if not isinstance(address, str) or not address:
        return None
    model = _source_model(graph)
    if model is None:
        return None
    for buffer_name, buffer in model.named_buffers():
        if buffer_name == address:
            return buffer
    value = _resolve_model_tensor_address(model, address)
    if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
        return value
    return None


def _source_model(graph: TraceGraph) -> nn.Module | None:
    """Return the live source model captured by the graph, if still available."""

    source_ref = getattr(graph.model_log, "_source_model_ref", None)
    return source_ref() if source_ref is not None else None


def _resolve_model_tensor_address(model: nn.Module, address: str) -> Any:
    """Best-effort dotted path resolver for non-registered tensor attributes."""

    current: Any = model
    for part in address.split("."):
        if isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
            continue
        if isinstance(current, nn.ModuleDict) and part in current:
            current = current[part]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        return None
    return current


def _call_node_target(node: TraceNode, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Execute a traced target with replay context only when it can affect results."""

    autocast_state = getattr(node.layer, "func_autocast_state", None)
    if _node_needs_replay_context(node, autocast_state):
        return execute_with_restored_rng_autocast(
            node.target,
            args,
            kwargs,
            rng_states=getattr(node.layer, "func_rng_states", None),
            autocast_state=autocast_state,
        )
    return node.target(*args, **kwargs)


def _node_needs_replay_context(
    node: TraceNode,
    autocast_state: dict[str, Any] | None,
) -> bool:
    if _autocast_enabled(autocast_state):
        return True
    op_name = str(node.op_type)
    if op_name in _RNG_SENSITIVE_OP_NAMES:
        return True
    target_name = getattr(node.target, "__name__", "")
    return bool(target_name in _RNG_SENSITIVE_OP_NAMES)


def _autocast_enabled(autocast_state: dict[str, Any] | None) -> bool:
    if not autocast_state:
        return False
    return any(bool(state.get("enabled", False)) for state in autocast_state.values())


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


def _has_parent_refs(value: Any) -> bool:
    if isinstance(value, ParentRef):
        return True
    if isinstance(value, Mapping):
        return any(_has_parent_refs(item) for item in value.values())
    if isinstance(value, tuple):
        return any(_has_parent_refs(item) for item in value)
    if isinstance(value, list):
        return any(_has_parent_refs(item) for item in value)
    return False


_BATCH_SHAPE_OPS = BATCH_DYNAMIC_CONSTANT_OPS
_MAX_VIEW_BATCH_MULTIPLIER = 512


def _dynamic_batch_literal(
    value: Any,
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    path: tuple[Any, ...],
) -> int | None:
    """Replace trace-batch literals in leading shape positions with runtime B."""

    traced_batch = graph.traced_batch_size
    if traced_batch is None or not isinstance(value, int) or value <= 0:
        return None
    if node.op_type not in _BATCH_SHAPE_OPS:
        return None
    if not _is_leading_shape_position(node.op_type, path):
        return None
    runtime_batch = _runtime_batch_from_env(env, graph)
    if runtime_batch is None:
        raise unsupported(_context(node, graph.batch_symbol, "could not infer runtime batch"))
    batch_multiplier = _batch_literal_multiplier(value, node, graph, env, path, runtime_batch)
    if batch_multiplier is None:
        return None
    if node.op_type == "repeat" and _repeat_input_already_has_batch_dim(
        node,
        graph,
        env,
    ):
        return None
    return runtime_batch * batch_multiplier


def _batch_literal_multiplier(
    value: int,
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    path: tuple[Any, ...],
    runtime_batch: int,
) -> int | None:
    traced_batch = graph.traced_batch_size
    if traced_batch is None:
        return None
    if value == traced_batch:
        if (
            node.op_type in {"view", "reshape"}
            and not _path_has_symbolic_batch_dim(node, path, graph.batch_symbol)
            and _is_last_shape_dim(node, path)
        ):
            return None
        if not _candidate_shape_is_compatible(runtime_batch, node, graph, env, path, runtime_batch):
            return None
        return 1
    if node.op_type not in {"view", "reshape", "expand"}:
        return None
    if value % traced_batch != 0:
        return None
    multiplier = value // traced_batch
    max_multiplier = _MAX_VIEW_BATCH_MULTIPLIER if node.op_type in {"view", "reshape"} else MAX_BATCH_MULTIPLIER
    if multiplier <= 1 or multiplier > max_multiplier:
        return None
    if node.op_type in {"view", "reshape"} and _shape_dim_index(path) != 0 and value < 8:
        return None
    if (
        node.op_type in {"view", "reshape"}
        and not _path_has_symbolic_batch_dim(node, path, graph.batch_symbol)
        and _is_last_shape_dim(node, path)
    ):
        return None
    if (
        node.op_type in {"view", "reshape"}
        and not _path_has_symbolic_batch_dim(node, path, graph.batch_symbol)
        and _has_better_compatible_dynamic_dim(node, graph, env, path, runtime_batch, multiplier)
    ):
        return None
    if node.output_shape and path_matches_output_dim(node, path, value):
        if not _candidate_shape_is_compatible(
            runtime_batch * multiplier,
            node,
            graph,
            env,
            path,
            runtime_batch,
        ):
            return None
        return multiplier
    return None


def _candidate_shape_is_compatible(
    candidate_dim: int,
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    path: tuple[Any, ...],
    runtime_batch: int,
) -> bool:
    if node.op_type not in {"view", "reshape"}:
        return True
    if not node.parents:
        return False
    parent_value = env.get(_final_label_for_ref(graph, node.parents[0]))
    if not isinstance(parent_value, torch.Tensor):
        return False
    shape = _candidate_view_shape(node, path, candidate_dim, graph.batch_symbol, runtime_batch)
    if shape is None:
        return False
    unknown_dims = sum(dim == -1 for dim in shape)
    if unknown_dims == 1 and not _path_has_symbolic_batch_dim(node, path, graph.batch_symbol):
        return False
    if unknown_dims > 1:
        return True
    known_product = 1
    for dim in shape:
        if dim == -1:
            continue
        if dim < 0:
            return True
        known_product *= dim
    if unknown_dims == 1:
        return known_product != 0 and parent_value.numel() % known_product == 0
    return parent_value.numel() == known_product


def _candidate_view_shape(
    node: TraceNode,
    target_path: tuple[Any, ...],
    candidate_dim: int,
    batch_symbol: str,
    runtime_batch: int,
) -> list[int] | None:
    shape_components = _view_shape_components(node)
    if not shape_components:
        return None
    shape: list[int] = []
    for component_path, component in shape_components:
        if component_path == target_path:
            shape.append(candidate_dim)
        elif (
            node.symbolic_output_shape is not None
            and (dim_index := _shape_dim_index(component_path)) is not None
            and 0 <= dim_index < len(node.symbolic_output_shape)
            and (batch_multiplier := _symbolic_batch_multiplier(
                node.symbolic_output_shape[dim_index],
                batch_symbol,
            ))
            is not None
        ):
            shape.append(runtime_batch * batch_multiplier)
        elif isinstance(component, LiteralValue) and isinstance(component.value, int):
            shape.append(component.value)
        elif isinstance(component, int):
            shape.append(component)
        else:
            return None
    return shape


def _view_shape_components(node: TraceNode) -> list[tuple[tuple[Any, ...], Any]]:
    if "shape" in node.kwargs:
        value = node.kwargs["shape"]
        if isinstance(value, tuple):
            return [(("kwargs", "shape", index), item) for index, item in enumerate(value)]
        return [(("kwargs", "shape"), value)]
    if "size" in node.kwargs:
        value = node.kwargs["size"]
        if isinstance(value, tuple):
            return [(("kwargs", "size", index), item) for index, item in enumerate(value)]
        return [(("kwargs", "size"), value)]
    if len(node.args) >= 2 and isinstance(node.args[1], tuple):
        return [(("args", 1, index), item) for index, item in enumerate(node.args[1])]
    if len(node.args) >= 2:
        return [(("args", index), item) for index, item in enumerate(node.args[1:], start=1)]
    return []


def _has_better_compatible_dynamic_dim(
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
    target_path: tuple[Any, ...],
    runtime_batch: int,
    target_multiplier: int,
) -> bool:
    if graph.traced_batch_size is None:
        return False
    for component_path, component in _view_shape_components(node):
        if component_path == target_path:
            continue
        if _path_has_symbolic_batch_dim(node, component_path, graph.batch_symbol):
            continue
        value = component.value if isinstance(component, LiteralValue) else component
        if not isinstance(value, int) or value <= 0 or value < 8:
            if (
                value == graph.traced_batch_size
                and not _is_last_shape_dim(node, component_path)
                and _candidate_shape_is_compatible(
                    runtime_batch,
                    node,
                    graph,
                    env,
                    component_path,
                    runtime_batch,
                )
            ):
                return True
            continue
        if _is_last_shape_dim(node, component_path):
            continue
        if value % graph.traced_batch_size != 0:
            continue
        multiplier = value // graph.traced_batch_size
        if (
            multiplier <= 1
            or multiplier > _MAX_VIEW_BATCH_MULTIPLIER
            or multiplier >= target_multiplier
        ):
            continue
        if not path_matches_output_dim(node, component_path, value):
            continue
        if _candidate_shape_is_compatible(
            runtime_batch * multiplier,
            node,
            graph,
            env,
            component_path,
            runtime_batch,
        ):
            return True
    return False


def _is_last_shape_dim(node: TraceNode, path: tuple[Any, ...]) -> bool:
    if not node.output_shape:
        return False
    dim_index = _shape_dim_index(path)
    return dim_index == len(node.output_shape) - 1


def path_matches_output_dim(node: TraceNode, path: tuple[Any, ...], value: int) -> bool:
    if not node.output_shape:
        return False
    dim_index = _shape_dim_index(path)
    if dim_index is None or dim_index < 0 or dim_index >= len(node.output_shape):
        return False
    return node.output_shape[dim_index] == value


def _path_has_symbolic_batch_dim(
    node: TraceNode,
    path: tuple[Any, ...],
    batch_symbol: str,
) -> bool:
    if not node.symbolic_output_shape:
        return False
    dim_index = _shape_dim_index(path)
    if (
        dim_index is None
        or dim_index < 0
        or dim_index >= len(node.symbolic_output_shape)
    ):
        return False
    return _symbolic_dim_contains_batch(node.symbolic_output_shape[dim_index], batch_symbol)


def _shape_dim_index(path: tuple[Any, ...]) -> int | None:
    if path[0] == "args" and len(path) == 2:
        return int(path[1]) - 1
    if path[0] == "args" and len(path) == 3 and path[1] == 1:
        return int(path[2])
    if path[0] == "kwargs" and len(path) == 3 and path[1] in {"shape", "size"}:
        return int(path[2])
    return None


def _is_leading_shape_position(op_type: str, path: tuple[Any, ...]) -> bool:
    """Return whether a literal path is the leading batch dimension of a shape arg."""

    if len(path) < 2 or path[0] not in {"args", "kwargs"}:
        return False
    if op_type in {"view", "reshape"}:
        return (path[0] == "args" and path[1] != 0) or (
            path[0] == "kwargs" and path[1] in {"shape", "size"}
        )
    if op_type in {"expand", "repeat"}:
        return path in {("args", 1), ("args", 1, 0), ("kwargs", "shape", 0), ("kwargs", "size", 0)}
    if op_type in {"zeros", "ones", "empty"}:
        return path in {("args", 0), ("args", 0, 0), ("kwargs", "size", 0)}
    if op_type in {"new_zeros", "new_ones", "new_empty"}:
        return path in {("args", 1), ("args", 1, 0), ("kwargs", "size", 0)}
    if op_type == "arange":
        return path in {("args", 0), ("kwargs", "end")}
    return False


def _runtime_batch_from_env(env: dict[str, Any], graph: TraceGraph) -> int | None:
    candidates: list[int] = []
    for label in graph.input_nodes:
        value = env.get(label)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            candidates.append(int(value.shape[0]))
    for node in graph.ordered_nodes():
        value = env.get(node.torchlens_label)
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and node.symbolic_output_shape is not None
            and node.symbolic_output_shape
        ):
            runtime_batch = _runtime_batch_from_symbolic_dim(
                int(value.shape[0]),
                node.symbolic_output_shape[0],
                graph.batch_symbol,
            )
            if runtime_batch is not None:
                candidates.append(runtime_batch)
    if candidates:
        return max(candidates)
    for value in env.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _runtime_batch_from_symbolic_dim(
    actual_dim: int,
    symbolic_dim: Any,
    batch_symbol: str,
) -> int | None:
    if symbolic_dim == batch_symbol:
        return actual_dim
    if not isinstance(symbolic_dim, str):
        return None
    prefix = f"{batch_symbol}*"
    if not symbolic_dim.startswith(prefix):
        return None
    try:
        multiplier = int(symbolic_dim[len(prefix) :])
    except ValueError:
        return None
    if multiplier <= 1 or actual_dim % multiplier != 0:
        return None
    return actual_dim // multiplier


def _repeat_input_already_has_batch_dim(
    node: TraceNode,
    graph: TraceGraph,
    env: dict[str, Any],
) -> bool:
    if not node.parents or graph.traced_batch_size is None:
        return False
    parent_label = _final_label_for_ref(graph, node.parents[0])
    parent_value = env.get(parent_label)
    parent = graph.get(parent_label)
    if isinstance(parent_value, torch.Tensor) and parent_value.ndim > 0:
        if bool(getattr(graph.get(parent_label).layer, "has_input_ancestor", True)):
            return True
        return _symbolic_dim_contains_batch(
            parent.symbolic_output_shape[0] if parent.symbolic_output_shape else None,
            graph.batch_symbol,
        )
    if bool(getattr(parent.layer, "has_input_ancestor", True)):
        return True
    return _symbolic_dim_contains_batch(
        parent.symbolic_output_shape[0] if parent.symbolic_output_shape else None,
        graph.batch_symbol,
    )


def _symbolic_dim_contains_batch(symbolic_dim: Any, batch_symbol: str) -> bool:
    return _symbolic_batch_multiplier(symbolic_dim, batch_symbol) is not None


def _symbolic_batch_multiplier(symbolic_dim: Any, batch_symbol: str) -> int | None:
    if symbolic_dim == batch_symbol:
        return 1
    if not isinstance(symbolic_dim, str):
        return None
    prefix = f"{batch_symbol}*"
    if not symbolic_dim.startswith(prefix):
        return None
    try:
        multiplier = int(symbolic_dim[len(prefix) :])
    except ValueError:
        return None
    return multiplier if multiplier > 1 else None


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
