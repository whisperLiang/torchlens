"""Project Torch segment replicas into backend-neutral state sources."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .. import _torch_compact
from ..ownership import StateKind, _StateSource
from ..state import SegmentState

if TYPE_CHECKING:
    from ..graph import SplitTraceGraph
    from ..runtime import SplitRuntime


_NamedSource = tuple[StateKind, tuple[str, ...], Any]


def _registered_sources(model: Any) -> dict[int, _NamedSource]:
    """Index registered parameters and buffers with all qualified Torch aliases."""

    sources: dict[int, _NamedSource] = {}
    methods: tuple[tuple[StateKind, str], ...] = (
        ("parameter", "named_parameters"),
        ("buffer", "named_buffers"),
    )
    for kind, method in methods:
        getter = getattr(model, method, None)
        if not callable(getter):
            continue
        for name, value in getter(remove_duplicate=False):
            prior = sources.get(id(value))
            aliases = (name,) if prior is None else tuple(sorted({*prior[1], name}))
            sources[id(value)] = (kind, aliases, value)

    return sources


def _named_sources(runtime: SplitRuntime) -> dict[int, _NamedSource]:
    """Join full or compact captured buffers to the same public Torch names."""

    sources = _registered_sources(runtime.model)
    # Full Buffer records and compact handles use different internal source
    # identities. Both resolve to the same public qualified model name.
    captured = []
    for ref in runtime.trace_graph._segment_blueprint(
        ("torch", "ownership_buffers"), lambda: _captured_buffers(runtime.trace_graph)
    ):
        handle = _torch_compact.live_buffer_value(ref)
        if handle is None:
            continue
        address = getattr(ref, "address", "")
        prior = sources.get(id(handle))
        if address:
            aliases = (address,) if prior is None else tuple(sorted({*prior[1], address}))
            sources[id(handle)] = ("buffer" if prior is None else prior[0], aliases, handle)
        captured.append((ref, handle))
    for ref, handle in captured:
        named = sources.get(id(handle))
        if named is not None:
            sources[getattr(ref, "source_id", id(ref))] = named
    return sources


def _captured_buffers(graph: SplitTraceGraph) -> tuple[Any, ...]:
    """Collect direct and parameter-module buffer references from the normalized graph."""

    refs: list[Any] = []
    for node in graph.nodes:
        refs.extend(node.buffer_refs)
        for parameter in node.param_refs:
            try:
                module = getattr(parameter, "module", None)
            except (AttributeError, KeyError):
                module = None
            refs.extend((getattr(module, "buffers", None) or {}).values())
    return tuple(refs)


def _node_id(occurrence: str) -> str:
    """Remove the state-occurrence suffix without stripping an op's pass index."""

    if ":literal:" in occurrence:
        return occurrence.rsplit(":literal:", 1)[0]
    return occurrence.removesuffix(":source")


@dataclass
class _EffectiveState:
    """Live occurrences of one source in the currently settled segments."""

    occurrences: set[str] = field(default_factory=set)
    bindings: dict[str, Any] = field(default_factory=dict)
    trainable: bool = False


def _effective_state(
    runtime: SplitRuntime,
    snapshots: dict[str, dict[str, Any]],
    named: dict[int, _NamedSource],
) -> dict[int, _EffectiveState]:
    """Group the segment bindings by the actual live source object."""

    raw: dict[int, _EffectiveState] = {}
    prefix = "training_prefix" if runtime.request.trainable else "prefix"
    for public, internal in (("prefix", prefix), ("suffix", "suffix")):
        segment = getattr(runtime.segments, internal)
        state = getattr(segment, "_state", None)
        if not isinstance(state, SegmentState):
            continue
        entries = {id(entry.value): entry for entry in state.entries()}
        for occurrence, value in snapshots.get(internal, {}).items():
            entry = entries.get(id(value))
            if entry is None:
                continue
            source = named.get(entry.source_id)
            source_key = entry.source_id if source is None else id(source[2])
            item = raw.setdefault(source_key, _EffectiveState())
            item.occurrences.add(occurrence)
            item.bindings[public] = value
            item.trainable |= entry.trainable
    return raw


def _torch_state_sources(
    runtime: SplitRuntime, snapshots: dict[str, dict[str, Any]]
) -> Iterator[_StateSource]:
    """Project settled effective bindings into immutable metadata and live mappings."""

    named = _named_sources(runtime)
    raw = _effective_state(runtime, snapshots, named)
    for source_id, item in raw.items():
        source = named.get(source_id)
        kind: StateKind = "literal" if source is None else source[0]
        aliases = () if source is None else source[1]
        identity = min(item.occurrences) if source is None else aliases[0]
        yield _StateSource(
            f"{kind}:{identity}",
            kind,
            aliases,
            frozenset(_node_id(key) for key in item.occurrences),
            item.bindings,
            item.trainable,
        )
