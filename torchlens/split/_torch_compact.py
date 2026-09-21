"""Detach Torch split execution data from value-retaining capture products."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from weakref import ReferenceType, ref

from ..backends.registry import TORCH_BACKEND_NAME
from ..data_classes._runtime_handles import source_model_from_trace
from ..ir.refs import TensorRef
from .graph import SplitTraceGraph, SplitTraceNode


@dataclass(slots=True)
class _BufferHandle:
    """Follow buffer replacement without retaining Trace metadata or the owner."""

    _value: Any
    source_id: int
    _owner_ref: ReferenceType[Any] | None = None
    _name: str = ""

    @property
    def handle(self) -> Any:
        """Resolve the current registered value after ``Module.to`` replaces it."""

        owner = None if self._owner_ref is None else self._owner_ref()
        if owner is not None:
            self._value = owner._buffers.get(self._name)
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
    out: Any = None


def _needs_source_payload(node: SplitTraceNode) -> bool:
    """Return whether replay needs a value that has no executing producer."""

    if node.is_input:
        return False
    if node.is_output:
        return not node.parents
    return node.is_buffer or (node.target is None and not node.parents)


def compact_torch_graph(graph: SplitTraceGraph) -> SplitTraceGraph:
    """Retain only normalized Torch replay data, independent of the source Trace.

    Parameters
    ----------
    graph:
        Torch graph after edge normalization and argument-template repair. The
        original capture must still be alive so live state handles can resolve.

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

    modules: dict[int, _ModuleBuffers] = {}
    parameters: dict[int, _ParamHandle] = {}
    buffers: dict[int, _BufferHandle] = {}

    def buffer_handle(buffer: Any) -> _BufferHandle:
        """Keep a weak owner lookup and stable binding identity for each buffer."""

        key = id(buffer)
        if key not in buffers:
            value = getattr(buffer, "handle", buffer)
            model = source_model_from_trace(getattr(buffer, "source_trace", None))
            owner = None
            name = ""
            address = getattr(buffer, "address", None)
            if model is not None and isinstance(address, str):
                module_path, _, name = address.rpartition(".")
                owner = model.get_submodule(module_path) if module_path else model
            buffers[key] = _BufferHandle(
                value, id(value), None if owner is None else ref(owner), name
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
