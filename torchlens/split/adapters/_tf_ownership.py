"""Resolve executed TensorFlow resource reads to named native variables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ...backends.tf._tensor_probe import _tensor_ref_key
from ...backends.tf.op_callback_capture import TFOpCapture
from ..ownership import StateKind, _owners_for_nodes, _refuse_ownership, _StateSource
from . import tf as _adapter

if TYPE_CHECKING:
    from ..graph import SplitTraceGraph, SplitTraceNode
    from ..runtime import SplitRuntime


@dataclass(frozen=True)
class _TfStateIndex:
    """Immutable capture provenance, with native reads as the authority for consumers."""

    consumers: Mapping[object, frozenset[str]]
    names: Mapping[object, frozenset[str]]
    parameters: frozenset[object]


def _read_keys(node: SplitTraceNode) -> Iterator[object]:
    """Yield native resource operands of an actual ReadVariableOp."""

    target = node.target
    if isinstance(target, TFOpCapture) and target.op_type == "ReadVariableOp":
        for operand in target.inputs:
            if operand.source_kind == "resource" and operand.ref_key is not None:
                yield operand.ref_key


def _index_reads(graph: SplitTraceGraph) -> _TfStateIndex:
    """Join native resource operands to capture labels and classification."""

    consumers: dict[object, set[str]] = defaultdict(set)
    names: dict[object, set[str]] = defaultdict(set)
    parameters = set()
    for node in graph.nodes:
        for ref in (*node.param_refs, *node.buffer_refs):
            key = _tensor_ref_key(_adapter._tf_variable_resource(_adapter._param_ref_handle(ref)))
            if key is not None:
                names[key].add(ref.address.replace(".", "/"))
                if getattr(ref, "is_trainable", False):
                    parameters.add(key)
        for key in _read_keys(node):
            consumers[key].add(node.canonical_id)
    return _TfStateIndex(
        MappingProxyType({key: frozenset(ids) for key, ids in consumers.items()}),
        MappingProxyType({key: frozenset(paths) for key, paths in names.items()}),
        frozenset(parameters),
    )


def _tf_state_index(graph: SplitTraceGraph) -> _TfStateIndex:
    """Reuse the capture's resource index across every split in its lineage."""

    index: _TfStateIndex = graph._segment_blueprint(
        ("tf", "state_index"), lambda: _index_reads(graph)
    )
    return index


def _attribute_paths(model: Any, variable_ids: set[int]) -> dict[int, set[str]]:
    """Find public tf.Module attribute aliases, including nested module paths."""

    tf = _adapter._tf()
    paths: dict[int, set[str]] = defaultdict(set)
    pending = [(model, "", frozenset({id(model)}))]
    while pending:
        module, prefix, ancestors = pending.pop()
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            path, identity = prefix + name, id(value)
            if identity in variable_ids:
                paths[identity].add(path)
            elif isinstance(value, tf.Module) and identity not in ancestors:
                pending.append((value, path + "/", ancestors | {identity}))
    return paths


def _variable_names(
    variable: Any, root: str, attributes: set[str], captured: frozenset[str]
) -> tuple[str, tuple[str, ...]]:
    """Prefer Keras paths or real tf.Module attributes over ambiguous native names."""

    path = getattr(variable, "path", None)
    name = str(variable.name).removesuffix(":0").removeprefix(root)
    if path:
        canonical = str(path).removeprefix(root)
    elif attributes and name not in attributes:
        canonical = min(attributes)
    else:
        canonical = name
    aliases = tuple(sorted({canonical, *captured, *attributes} - {""}))
    return canonical, aliases


def _match_variables(
    runtime: SplitRuntime, index: _TfStateIndex, variables: tuple[Any, ...]
) -> dict[object, list[Any]]:
    """Check that the resource index describes the model's current live state."""

    by_resource: dict[object, list[Any]] = defaultdict(list)
    for variable in variables:
        key = _tensor_ref_key(_adapter._tf_variable_resource(variable))
        if key in index.consumers:
            by_resource[key].append(variable)
    if index.consumers.keys() - by_resource.keys():
        _refuse_ownership(
            runtime,
            "tf_state_resource_unbound",
            "An executed TensorFlow resource has no current model variable; prepare a new split graph.",
        )
    for variable in getattr(runtime.model, "trainable_variables", ()):
        key = _tensor_ref_key(_adapter._tf_variable_resource(variable))
        if key in index.names and key not in index.consumers:
            _refuse_ownership(
                runtime,
                "tf_state_resource_read_unproven",
                f"Variable {variable.name!r} has capture provenance without an executable resource read.",
            )
    return by_resource


def _tf_state_sources(runtime: SplitRuntime) -> Iterator[_StateSource]:
    """Join exact replay readers, live variables, aliases, and current trainability."""

    index = _tf_state_index(runtime.trace_graph)
    variables = tuple(getattr(runtime.model, "variables", ()))
    attributes = _attribute_paths(runtime.model, {id(value) for value in variables})
    by_resource = _match_variables(runtime, index, variables)
    active = _training_bindings(runtime)
    root = f"{getattr(runtime.model, 'name', '')}/"
    for key, aliases in by_resource.items():
        names = [
            _variable_names(
                value, root, attributes.get(id(value), set()), index.names.get(key, frozenset())
            )
            for value in aliases
        ]
        canonical = min(name for name, _ in names)
        nodes = index.consumers[key]
        owners = _owners_for_nodes(runtime.plan, nodes)
        bindings: dict[str, Any] = {owner: active[owner].get(key, aliases[0]) for owner in owners}
        trainable = any(key in active[owner] for owner in owners)
        kind: StateKind = "parameter" if trainable or key in index.parameters else "buffer"
        if owners:
            yield _StateSource(
                f"{kind}:{canonical}",
                kind,
                tuple(sorted({name for _, paths in names for name in paths})),
                nodes,
                bindings,
                trainable,
            )


def _training_bindings(runtime: SplitRuntime) -> dict[str, dict[object, Any]]:
    """Use the same active variable objects as the native training entry points."""

    prefix = runtime.segments.training_prefix or runtime.segments.prefix
    return {
        owner: {
            _tensor_ref_key(_adapter._tf_variable_resource(value)): value
            for value in segment._trainable_variables(runtime.model)
        }
        for owner, segment in (("prefix", prefix), ("suffix", runtime.segments.suffix))
    }
