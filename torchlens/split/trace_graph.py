"""TraceGraph adapter from TorchLens ModelLog."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

from .canonical import build_canonical_ids
from .shape import ShapeEnv, SymbolicShape

if TYPE_CHECKING:
    from torchlens.data_classes.layer_pass_log import LayerPassLog
    from torchlens.data_classes.model_log import ModelLog

ReplaySourcePolicy = Literal[
    "constant",
    "live_param",
    "live_param_derived",
    "batch_dynamic_constant",
]

BATCH_DYNAMIC_CONSTANT_OPS = frozenset(
    {
        "view",
        "reshape",
        "flatten",
        "expand",
        "repeat",
        "zeros",
        "ones",
        "empty",
        "new_zeros",
        "new_ones",
        "new_empty",
        "arange",
        "meshgrid",
    }
)


@dataclass(frozen=True)
class TraceNode:
    """Split compiler IR node derived from one TorchLens layer pass."""

    canonical_id: str
    torchlens_label: str
    module_path: str
    op_type: str
    target: Any
    args: Any
    kwargs: Any
    parents: tuple[str, ...]
    children: tuple[str, ...]
    output_shape: tuple[int, ...] | None
    symbolic_output_shape: SymbolicShape | None
    dtype: torch.dtype | None
    requires_grad: bool
    is_input: bool
    is_output: bool
    is_param_op: bool
    replay_source_policy: ReplaySourcePolicy
    layer: "LayerPassLog"


@dataclass(frozen=True)
class TraceGraph:
    """Dependency graph used by TorchLens split planning and lowering."""

    nodes: OrderedDict[str, TraceNode]
    input_nodes: tuple[str, ...]
    output_nodes: tuple[str, ...]
    graph_shape_hash: str | None
    traced_batch_size: int | None
    batch_symbol: str
    shape_env: ShapeEnv
    model_log: "ModelLog"

    def ordered_nodes(self) -> tuple[TraceNode, ...]:
        """Return nodes in observed execution order."""

        return tuple(self.nodes.values())

    def get(self, label: str) -> TraceNode:
        """Return a node by TorchLens label."""

        return self.nodes[label]


def trace_graph_from_model_log(
    model_log: "ModelLog",
    *,
    traced_batch_size: int | None,
    batch_symbol: str,
    dynamic_batch: tuple[int, int] | None,
) -> TraceGraph:
    """Build a split TraceGraph from a TorchLens ModelLog."""

    shape_env = ShapeEnv(
        batch_symbol=batch_symbol,
        traced_batch_size=traced_batch_size,
        dynamic_batch=dynamic_batch,
    )
    layers = list(model_log.layer_list)
    canonical_ids = build_canonical_ids(layers, shape_env)
    nodes: OrderedDict[str, TraceNode] = OrderedDict()
    for layer in layers:
        label = str(layer.layer_label)
        canonical = canonical_ids[label]
        activation = getattr(layer, "activation", None)
        requires_grad = bool(activation.requires_grad) if isinstance(activation, torch.Tensor) else False
        template = getattr(layer, "captured_arg_template", None)
        nodes[label] = TraceNode(
            canonical_id=canonical.canonical_id,
            torchlens_label=label,
            module_path=canonical.module_path,
            op_type=canonical.op_type,
            target=getattr(layer, "func_applied", None),
            args=template.args if template is not None else (),
            kwargs=dict(template.kwargs) if template is not None else {},
            parents=tuple(getattr(layer, "parent_layers", ()) or ()),
            children=tuple(getattr(layer, "child_layers", ()) or ()),
            output_shape=getattr(layer, "tensor_shape", None),
            symbolic_output_shape=canonical.symbolic_shape,
            dtype=getattr(layer, "tensor_dtype", None),
            requires_grad=requires_grad,
            is_input=bool(getattr(layer, "is_input_layer", False)),
            is_output=bool(getattr(layer, "is_output_layer", False)),
            is_param_op=bool(getattr(layer, "parent_param_barcodes", None)),
            replay_source_policy=_initial_replay_source_policy(layer),
            layer=layer,
        )
    nodes = _apply_live_param_replay_policies(nodes, model_log)
    return TraceGraph(
        nodes=nodes,
        input_nodes=tuple(getattr(model_log, "input_layers", ()) or ()),
        output_nodes=tuple(getattr(model_log, "output_layers", ()) or ()),
        graph_shape_hash=getattr(model_log, "graph_shape_hash", None),
        traced_batch_size=traced_batch_size,
        batch_symbol=batch_symbol,
        shape_env=shape_env,
        model_log=model_log,
    )


def _initial_replay_source_policy(layer: "LayerPassLog") -> ReplaySourcePolicy:
    """Classify a trace node before live-param closure propagation."""

    if _is_live_param_source_layer(layer):
        return "live_param"
    if _is_batch_dynamic_constant_layer(layer):
        return "batch_dynamic_constant"
    return "constant"


def _is_live_param_source_layer(layer: "LayerPassLog") -> bool:
    """Return whether ``layer`` is a direct live parameter/buffer source."""

    if bool(getattr(layer, "is_buffer_layer", False)) and getattr(layer, "buffer_address", None):
        return True
    parent_param_logs = getattr(layer, "parent_param_logs", None)
    parent_param_barcodes = getattr(layer, "parent_param_barcodes", None)
    if not parent_param_logs and not parent_param_barcodes:
        return False
    if bool(getattr(layer, "has_input_ancestor", True)):
        return False
    return not bool(getattr(layer, "parent_layers", None))


def _is_batch_dynamic_constant_layer(layer: "LayerPassLog") -> bool:
    """Return whether ``layer`` is a no-input constant requiring dynamic shape replay."""

    if bool(getattr(layer, "has_input_ancestor", True)):
        return False
    return str(getattr(layer, "layer_type", "")) in BATCH_DYNAMIC_CONSTANT_OPS


def _apply_live_param_replay_policies(
    nodes: OrderedDict[str, TraceNode],
    model_log: "ModelLog",
) -> OrderedDict[str, TraceNode]:
    """Propagate live-param dependency only through no-input downstream nodes."""

    raw_to_final = getattr(model_log, "_raw_to_final_layer_labels", {}) or {}
    policies: dict[str, ReplaySourcePolicy] = {
        label: node.replay_source_policy for label, node in nodes.items()
    }
    queue = [label for label, policy in policies.items() if policy == "live_param"]
    cursor = 0
    while cursor < len(queue):
        parent_label = queue[cursor]
        cursor += 1
        parent = nodes[parent_label]
        for child_label_raw in parent.children:
            child_label = str(raw_to_final.get(child_label_raw, child_label_raw))
            if child_label not in nodes:
                continue
            child = nodes[child_label]
            if child.is_input or child.is_output:
                continue
            if bool(getattr(child.layer, "has_input_ancestor", True)):
                continue
            if policies[child_label] == "live_param":
                continue
            if policies[child_label] != "live_param_derived":
                policies[child_label] = "live_param_derived"
                queue.append(child_label)

    updated: OrderedDict[str, TraceNode] = OrderedDict()
    for label, node in nodes.items():
        updated[label] = replace(node, replay_source_policy=policies[label])
    return updated


def is_compute_split_node(node: TraceNode) -> bool:
    """Return whether ``node`` should be considered an automatic compute split point."""

    return (
        not node.is_input
        and not node.is_output
        and not bool(getattr(node.layer, "is_buffer_layer", False))
    )


__all__ = [
    "BATCH_DYNAMIC_CONSTANT_OPS",
    "ReplaySourcePolicy",
    "TraceGraph",
    "TraceNode",
    "is_compute_split_node",
    "trace_graph_from_model_log",
]
