"""Detach Torch split execution data from value-retaining capture products."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from weakref import ReferenceType, ref

from ..backends.registry import TORCH_BACKEND_NAME
from ..data_classes._runtime_handles import source_model_from_trace
from ..intervention.types import LiteralTensor
from ..ir.refs import TensorRef
from .errors import SplitErrorContext, SplitUnsupportedError
from .graph import SplitTraceGraph, SplitTraceNode


def _buffer_owner(buffer: Any) -> tuple[Any | None, str]:
    """Resolve a captured buffer's module without retaining its diagnostic Trace."""

    model = source_model_from_trace(getattr(buffer, "source_trace", None))
    address = getattr(buffer, "address", None)
    if model is None or not isinstance(address, str):
        return None, ""
    module_path, _, name = address.rpartition(".")
    try:
        return (model.get_submodule(module_path) if module_path else model), name
    except AttributeError:
        return None, name


def _owner_buffer_value(owner: Any, name: str) -> Any:
    """Read registered or plain Tensor state without executing a user descriptor."""

    if name in owner._buffers:
        return owner._buffers[name]
    return owner.__dict__.get(name)


def live_buffer_value(buffer: Any) -> Any:
    """Resolve both registered buffers and capture-tagged plain Tensor attributes."""

    value = getattr(buffer, "handle", buffer)
    if value is not None:
        return value
    owner, name = _buffer_owner(buffer)
    return None if owner is None else _owner_buffer_value(owner, name)


@dataclass(slots=True)
class _BufferHandle:
    """Follow buffer replacement without retaining Trace metadata or the owner."""

    _value: Any
    source_id: int
    _owner_ref: ReferenceType[Any] | None = None
    _name: str = ""
    address: str = ""

    @property
    def handle(self) -> Any:
        """Resolve the current registered or plain Tensor state value."""

        owner = None if self._owner_ref is None else self._owner_ref()
        if owner is not None:
            self._value = _owner_buffer_value(owner, self._name)
        return self._value


@dataclass(frozen=True, slots=True)
class _ModuleBuffers:
    """Retain only the module buffer handles consulted by Torch replay."""

    buffers: dict[str, _BufferHandle]


@dataclass(frozen=True, slots=True)
class _ParamHandle:
    """Keep parameter identity and replay-relevant buffer ownership only."""

    handle: Any
    is_trainable: bool
    module: _ModuleBuffers | None


@dataclass(frozen=True, slots=True)
class _ReplayOpState:
    """The closed subset of Op state consumed after graph normalization."""

    func_rng_states: dict[str, Any] | None
    func_autocast_state: dict[str, Any] | None
    multi_output_index: Any
    is_inplace: bool = False
    out: Any = None


def _needs_source_payload(node: SplitTraceNode) -> bool:
    """Return whether replay needs a value that has no executing producer."""

    if node.is_input:
        return False
    if node.is_output:
        return not node.parents
    return node.is_buffer or (node.target is None and not node.parents)


def _validate_training_sources(graph: SplitTraceGraph) -> None:
    """Refuse retained sources that would reuse a capture-time autograd graph.

    Parameters
    ----------
    graph:
        Normalized graph with source values and literal arguments still alive.

    Notes
    -----
    Ordinary activations are reconstructed by replay and do not need this
    check. A non-leaf tensor with no replay producer would instead carry a
    one-shot autograd history across training steps. Detaching it would silently
    lose gradients, so compact training refuses that unprovable lifetime.
    """

    def check(value: Any, node: SplitTraceNode) -> None:
        """Reject a retained tensor with a capture-owned backward history."""

        if getattr(value, "grad_fn", None) is not None:
            raise SplitUnsupportedError(
                "Compact training cannot retain a non-leaf source tensor's autograd "
                "history. Recompute that source inside the captured model.",
                context=SplitErrorContext(
                    backend=graph.backend,
                    split_point="",
                    layer_label=node.label,
                    op_type=node.op_type,
                    reason="compact_training_autograd_source",
                ),
            )

    def visit(component: Any, node: SplitTraceNode) -> None:
        """Inspect the normalized template tree without changing any values."""

        if isinstance(component, LiteralTensor):
            check(component.value, node)
        elif isinstance(component, tuple):
            for item in component:
                visit(item, node)

    for node in graph.nodes:
        if _needs_source_payload(node):
            check(getattr(node.op, "out", None), node)
        template = node.args_template
        if template is not None:
            visit(getattr(template, "args", ()), node)
            visit(getattr(template, "kwargs", ()), node)


def compact_torch_graph(graph: SplitTraceGraph, *, training: bool = False) -> SplitTraceGraph:
    """Retain only normalized Torch replay data, independent of the source Trace.

    Parameters
    ----------
    graph:
        Torch graph after edge normalization and argument-template repair. The
        original capture must still be alive so live state handles can resolve.
    training:
        Check that retained sources do not carry one-shot captured autograd
        histories. Training execution builds a fresh graph from live state on
        every call; the source Trace's backward hooks are not required.

    Returns
    -------
    SplitTraceGraph
        A new graph without full Op, Param, Buffer or activation TensorRef
        payloads. Ordinary activation snapshots are omitted; target-less source
        constants, captured buffer values and parent-less output sources remain
        because they have no replay operation that can reconstruct their values.

    Notes
    -----
    The original graph and Trace are not modified or cleaned up. Callable targets,
    normalized argument templates, RNG snapshots and live tensor handles retain
    their identities. In particular, this operation never detaches a parameter,
    copies a weight, or severs a runtime input's autograd connection. Shape
    compilation and the B=2 verification probe may consume the resulting graph.
    """

    if graph.backend != TORCH_BACKEND_NAME:
        raise ValueError("compact_torch_graph requires a normalized Torch graph.")
    if training:
        _validate_training_sources(graph)

    modules: dict[int, _ModuleBuffers] = {}
    parameters: dict[int, _ParamHandle] = {}
    buffers: dict[int, _BufferHandle] = {}

    def buffer_handle(buffer: Any) -> _BufferHandle:
        """Keep a weak owner lookup and stable binding identity for each buffer."""

        key = id(buffer)
        if key not in buffers:
            value = live_buffer_value(buffer)
            owner, name = _buffer_owner(buffer)
            buffers[key] = _BufferHandle(
                value,
                id(value),
                None if owner is None else ref(owner),
                name,
                getattr(buffer, "address", ""),
            )
        return buffers[key]

    def param_handle(param: Any) -> _ParamHandle:
        """Resolve parameter/module state before source metadata is released."""

        key = id(param)
        if key in parameters:
            return parameters[key]
        try:
            module = getattr(param, "module", None)
        except (AttributeError, KeyError):
            # A parameter's owner need not have an executed module record.
            module = None
        module_state = None
        if module is not None:
            module_key = id(module)
            if module_key not in modules:
                modules[module_key] = _ModuleBuffers(
                    {
                        name: buffer_handle(buffer)
                        for name, buffer in (getattr(module, "buffers", None) or {}).items()
                    }
                )
            module_state = modules[module_key]
        result = _ParamHandle(
            handle=getattr(param, "handle", None),
            is_trainable=bool(getattr(param, "is_trainable", False)),
            module=module_state,
        )
        parameters[key] = result
        return result

    nodes = []
    for node in graph.nodes:
        rng = getattr(node.op, "func_rng_states", None)
        autocast = getattr(node.op, "func_autocast_state", None)
        op = _ReplayOpState(
            func_rng_states=rng,
            func_autocast_state=autocast,
            multi_output_index=getattr(node.op, "multi_output_index", None),
            is_inplace=bool(getattr(node.op, "is_inplace", False)),
            out=getattr(node.op, "out", None) if _needs_source_payload(node) else None,
        )
        # TensorRef is metadata plus an optional strong activation reference.
        # Replay never consumes output_ref, but its portable metadata is useful
        # to graph inspectors. Unknown opaque refs cannot be kept safely.
        output_ref = (
            replace(node.output_ref, payload=None)
            if isinstance(node.output_ref, TensorRef)
            else None
        )
        nodes.append(
            replace(
                node,
                op=op,
                output_ref=output_ref,
                param_refs=tuple(param_handle(param) for param in node.param_refs),
                buffer_refs=tuple(buffer_handle(buffer) for buffer in node.buffer_refs),
            )
        )
    return replace(graph, nodes=tuple(nodes))
