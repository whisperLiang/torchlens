"""TraceGraph adapter from TorchLens ModelLog."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .canonical import build_canonical_ids
from .shape import ShapeEnv, SymbolicShape

if TYPE_CHECKING:
    from torchlens.data_classes.layer_pass_log import LayerPassLog
    from torchlens.data_classes.model_log import ModelLog


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
            layer=layer,
        )
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


__all__ = ["TraceGraph", "TraceNode", "trace_graph_from_model_log"]
