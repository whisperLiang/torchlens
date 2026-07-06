"""Split graph construction from current-main TorchLens traces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from .shape import SymbolicShape, infer_traced_batch_size, symbolic_shape_from_tensor_ref


ReplaySourcePolicy = Literal[
    "constant",
    "live_param",
    "live_param_derived",
    "batch_dynamic_constant",
]


@dataclass(frozen=True)
class SplitTraceNode:
    """Backend-neutral split-replay node projected from a TorchLens ``Op``."""

    label: str
    raw_label: str | None
    canonical_id: str
    backend: str
    raw_index: int | None
    op_type: str
    target: Any | None
    func_call_id: int | None
    args_template: Any | None
    kwargs_template: Any | None
    parents: tuple[str, ...]
    children: tuple[str, ...]
    output_ref: Any | None
    module_path: str | None
    output_shape: tuple[int, ...] | None
    symbolic_output_shape: SymbolicShape | None
    dtype: str | None
    requires_grad: bool | None
    output_container_path: tuple[Any, ...]
    output_container_spec: Any | None
    is_input: bool
    is_output: bool
    is_buffer: bool
    is_buffer_only_source: bool
    is_param_source: bool
    param_refs: tuple[Any, ...]
    replay_source_policy: ReplaySourcePolicy
    op: Any


@dataclass(frozen=True)
class SplitTraceGraph:
    """Ordered split graph projected from a TorchLens trace."""

    backend: str
    nodes: tuple[SplitTraceNode, ...]
    input_node_ids: tuple[str, ...]
    output_node_ids: tuple[str, ...]
    graph_shape_hash: str | None
    traced_batch_size: int | None

    @property
    def node_by_id(self) -> dict[str, SplitTraceNode]:
        """Return nodes keyed by canonical ID."""

        return {node.canonical_id: node for node in self.nodes}

    @property
    def node_by_label(self) -> dict[str, SplitTraceNode]:
        """Return nodes keyed by unique final display label."""

        labels: dict[str, SplitTraceNode] = {}
        duplicates: set[str] = set()
        for node in self.nodes:
            existing = labels.get(node.label)
            if existing is None:
                labels[node.label] = node
            elif existing.canonical_id != node.canonical_id:
                duplicates.add(node.label)
        for label in duplicates:
            labels.pop(label, None)
        return labels

    @property
    def node_id_by_alias(self) -> dict[str, str]:
        """Return unique raw/display/canonical label aliases to canonical IDs."""

        aliases: dict[str, str] = {}
        duplicates: set[str] = set()
        for node in self.nodes:
            for alias in (node.canonical_id, node.label, node.raw_label):
                if alias is None:
                    continue
                existing = aliases.get(alias)
                if existing is None:
                    aliases[alias] = node.canonical_id
                elif existing != node.canonical_id:
                    duplicates.add(alias)
        for alias in duplicates:
            aliases.pop(alias, None)
        return aliases

    def node_for_label(self, label: str) -> SplitTraceNode | None:
        """Resolve a unique raw/display/canonical label alias."""

        node_id = self.node_id_by_alias.get(label)
        if node_id is None:
            return None
        return self.node_by_id[node_id]

    @property
    def order_by_id(self) -> dict[str, int]:
        """Return topological order indexes keyed by canonical ID."""

        return {node.canonical_id: index for index, node in enumerate(self.nodes)}

    @property
    def compute_nodes(self) -> tuple[SplitTraceNode, ...]:
        """Return eligible compute-ish nodes before target-specific rejection."""

        return tuple(
            node
            for node in self.nodes
            if not (node.is_input or node.is_output or node.is_buffer or node.is_buffer_only_source)
        )


def _module_path_for_op(op: Any) -> str | None:
    """Return a stable module path for an op when available."""

    address = getattr(op, "address", None)
    if isinstance(address, str) and address:
        return address
    module = getattr(op, "module", None)
    if isinstance(module, tuple) and module and module[0]:
        return str(module[0])
    modules = getattr(op, "modules", None)
    if modules:
        try:
            return str(modules[-1][0])
        except (IndexError, TypeError):
            return None
    return None


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    """Normalize shape metadata to a tuple of ints."""

    if value is None:
        return None
    try:
        return tuple(int(dim) for dim in value)
    except (TypeError, ValueError):
        return None


def _requires_grad_for_op(op: Any) -> bool | None:
    """Return saved-output grad metadata when a tensor payload is available."""

    out = getattr(op, "out", None)
    requires_grad = getattr(out, "requires_grad", None)
    if requires_grad is None:
        return None
    return bool(requires_grad)


def _children_from_parents(nodes: tuple[SplitTraceNode, ...]) -> dict[str, tuple[str, ...]]:
    """Build child IDs when finalized child metadata is incomplete."""

    children: dict[str, list[str]] = {node.canonical_id: [] for node in nodes}
    graph = SplitTraceGraph(
        backend="",
        nodes=nodes,
        input_node_ids=(),
        output_node_ids=(),
        graph_shape_hash=None,
        traced_batch_size=None,
    )
    for node in nodes:
        for parent in node.parents:
            parent_node = graph.node_for_label(parent)
            if parent_node is not None:
                children[parent_node.canonical_id].append(node.canonical_id)
    return {label: tuple(values) for label, values in children.items()}


def _canonical_label_for_op(op: Any, display_label: str) -> str:
    """Return the stable per-op ID for split replay."""

    op_label = getattr(op, "label", None)
    if isinstance(op_label, str) and op_label:
        return op_label
    raw_label = getattr(op, "_label_raw", None) or getattr(op, "raw_label", None)
    if isinstance(raw_label, str) and raw_label:
        return raw_label
    raw_index = getattr(op, "raw_index", None)
    if raw_index is not None:
        return f"{display_label}#{raw_index}"
    return display_label


def _normalize_edge_aliases(nodes: list[SplitTraceNode]) -> tuple[SplitTraceNode, ...]:
    """Rewrite resolvable parent/child aliases to canonical node IDs."""

    graph = SplitTraceGraph(
        backend="",
        nodes=tuple(nodes),
        input_node_ids=(),
        output_node_ids=(),
        graph_shape_hash=None,
        traced_batch_size=None,
    )

    def normalize(labels: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for label in labels:
            node = graph.node_for_label(label)
            normalized.append(node.canonical_id if node is not None else label)
        return tuple(normalized)

    return tuple(
        replace(node, parents=normalize(node.parents), children=normalize(node.children))
        for node in nodes
    )


def _attach_paddle_capture_templates(
    trace: Any,
    nodes: list[SplitTraceNode],
) -> list[SplitTraceNode]:
    """Attach Paddle replay templates that live on backend capture records."""

    captures = {
        str(getattr(capture, "label_raw")): capture
        for capture in getattr(trace, "_paddle_op_captures", ()) or ()
        if getattr(capture, "label_raw", None) is not None
    }
    if not captures:
        return nodes
    updated: list[SplitTraceNode] = []
    for node in nodes:
        capture = captures.get(str(getattr(node.op, "_label_raw", node.raw_label or "")))
        if capture is None:
            updated.append(node)
            continue
        updated.append(
            replace(
                node,
                target=getattr(capture, "func", node.target),
                args_template=tuple(getattr(capture, "args_template", ()) or ()),
                kwargs_template=dict(getattr(capture, "kwargs_template", {}) or {}),
            )
        )
    return updated


def _attach_jax_captures(trace: Any, nodes: list[SplitTraceNode]) -> list[SplitTraceNode]:
    """Attach JAX runtime captures to their split graph nodes."""

    raw_by_index = getattr(trace, "_jax_capture_index_to_raw_op_label", {}) or {}
    captures = {
        str(raw_by_index.get(getattr(capture, "index"))): capture
        for capture in getattr(trace, "jax_ordered_captures", ()) or ()
        if raw_by_index.get(getattr(capture, "index", None)) is not None
    }
    if not captures:
        return nodes
    return [
        replace(
            node,
            target=captures.get(str(getattr(node.op, "_label_raw", node.raw_label or ""))),
        )
        if str(getattr(node.op, "_label_raw", node.raw_label or "")) in captures
        else node
        for node in nodes
    ]


def _attach_tinygrad_captures(trace: Any, nodes: list[SplitTraceNode]) -> list[SplitTraceNode]:
    """Attach tinygrad UOp captures to their split graph nodes."""

    captures = {
        str(getattr(capture, "label_raw")): capture
        for capture in getattr(trace, "tinygrad_uop_captures", ()) or ()
        if getattr(capture, "label_raw", None) is not None
    }
    if not captures:
        return nodes
    return [
        replace(
            node,
            target=captures.get(str(getattr(node.op, "_label_raw", node.raw_label or ""))),
        )
        if str(getattr(node.op, "_label_raw", node.raw_label or "")) in captures
        else node
        for node in nodes
    ]


def _attach_tf_captures(trace: Any, nodes: list[SplitTraceNode]) -> list[SplitTraceNode]:
    """Attach TensorFlow op-callback captures to their split graph nodes."""

    captures = {
        str(getattr(capture, "label_raw")): capture
        for capture in getattr(trace, "_tf_op_captures", ()) or ()
        if getattr(capture, "label_raw", None) is not None
    }
    if not captures:
        return nodes
    return [
        replace(
            node,
            target=captures.get(str(getattr(node.op, "_label_raw", node.raw_label or ""))),
        )
        if str(getattr(node.op, "_label_raw", node.raw_label or "")) in captures
        else node
        for node in nodes
    ]


def _node_from_op(
    op: Any,
    *,
    backend: str,
    batch_symbol: str,
    dynamic_batch: tuple[int, int] | None,
    traced_batch_size: int | None,
) -> SplitTraceNode:
    """Project one finalized TorchLens ``Op`` to a split graph node."""

    label = str(getattr(op, "layer_label", getattr(op, "label", "")))
    canonical_id = _canonical_label_for_op(op, label)
    raw_label = getattr(op, "_label_raw", None) or getattr(op, "raw_label", None)
    shape = _shape_tuple(getattr(op, "shape", None))
    is_buffer = bool(getattr(op, "is_buffer", False))
    parents = tuple(str(parent) for parent in (getattr(op, "parents", ()) or ()))
    children = tuple(str(child) for child in (getattr(op, "children", ()) or ()))
    is_buffer_only_source = (
        is_buffer and not parents and getattr(op, "buffer_write_kind", None) is None
    )
    param_refs = tuple(getattr(op, "_param_logs", ()) or ())
    is_param_source = bool(getattr(op, "input_was_parameter", False))
    replay_source_policy: ReplaySourcePolicy = "constant"
    if param_refs:
        replay_source_policy = "live_param"
    elif is_param_source:
        replay_source_policy = "live_param_derived"
    elif dynamic_batch is not None and shape and traced_batch_size is not None:
        replay_source_policy = "batch_dynamic_constant"
    return SplitTraceNode(
        label=label,
        raw_label=None if raw_label is None else str(raw_label),
        canonical_id=canonical_id,
        backend=backend,
        raw_index=getattr(op, "raw_index", None),
        op_type=str(getattr(op, "type", getattr(op, "layer_type", "unknown"))),
        target=getattr(op, "func", None),
        func_call_id=getattr(op, "func_call_id", None),
        args_template=getattr(op, "args_template", None),
        kwargs_template=getattr(op, "kwargs_template", None),
        parents=parents,
        children=children,
        output_ref=getattr(op, "out_ref", None),
        module_path=_module_path_for_op(op),
        output_shape=shape,
        symbolic_output_shape=symbolic_shape_from_tensor_ref(
            shape,
            batch_symbol=batch_symbol,
            dynamic_batch=dynamic_batch,
            traced_batch_size=traced_batch_size,
        ),
        dtype=None if getattr(op, "dtype", None) is None else str(getattr(op, "dtype")),
        requires_grad=_requires_grad_for_op(op),
        output_container_path=tuple(getattr(op, "container_path", ()) or ()),
        output_container_spec=getattr(op, "container_spec", None),
        is_input=bool(getattr(op, "is_input", False)),
        is_output=bool(getattr(op, "is_output", False)),
        is_buffer=is_buffer,
        is_buffer_only_source=is_buffer_only_source,
        is_param_source=is_param_source,
        param_refs=param_refs,
        replay_source_policy=replay_source_policy,
        op=op,
    )


def split_graph_from_trace(
    trace: Any,
    *,
    batch_symbol: str,
    dynamic_batch: tuple[int, int] | None,
) -> SplitTraceGraph:
    """Build a split graph from a current-main TorchLens ``Trace``.

    Parameters
    ----------
    trace:
        Current-main TorchLens trace.
    batch_symbol:
        Symbol to use for dynamic leading batch dimensions.
    dynamic_batch:
        Optional inclusive runtime batch range.

    Returns
    -------
    SplitTraceGraph
        Ordered split graph.
    """

    backend = str(getattr(trace, "backend", "torch"))
    traced_batch_size = infer_traced_batch_size(trace)
    nodes = [
        _node_from_op(
            op,
            backend=backend,
            batch_symbol=batch_symbol,
            dynamic_batch=dynamic_batch,
            traced_batch_size=traced_batch_size,
        )
        for op in getattr(trace, "layer_list", ()) or ()
    ]
    if backend == "paddle":
        nodes = _attach_paddle_capture_templates(trace, nodes)
    elif backend == "jax":
        nodes = _attach_jax_captures(trace, nodes)
    elif backend == "tinygrad":
        nodes = _attach_tinygrad_captures(trace, nodes)
    elif backend in {"tf", "tensorflow"}:
        nodes = _attach_tf_captures(trace, nodes)
    normalized_nodes = _normalize_edge_aliases(nodes)
    rebuilt_children = _children_from_parents(normalized_nodes)
    fixed_nodes = tuple(
        node
        if node.children
        else replace(node, children=rebuilt_children.get(node.canonical_id, ()))
        for node in normalized_nodes
    )
    return SplitTraceGraph(
        backend=backend,
        nodes=fixed_nodes,
        input_node_ids=tuple(node.canonical_id for node in fixed_nodes if node.is_input),
        output_node_ids=tuple(node.canonical_id for node in fixed_nodes if node.is_output),
        graph_shape_hash=getattr(trace, "graph_shape_hash", None),
        traced_batch_size=traced_batch_size,
    )


__all__ = [
    "ReplaySourcePolicy",
    "SplitTraceGraph",
    "SplitTraceNode",
    "split_graph_from_trace",
]
