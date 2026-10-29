"""Stable state ownership and caller-owned optimizer migration descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from .errors import SplitErrorContext, SplitUnsupportedError
from .ir import GraphPartition
from .state import SegmentState

StateKind = Literal["parameter", "buffer", "literal"]


@dataclass(frozen=True)
class ParameterOwnership:
    """One unique replay state source, including buffers and anonymous literals.

    Named IDs use qualified model names and retain all tied aliases. Anonymous
    IDs use canonical replay occurrences. IDs are scoped to ``graph_id`` and
    never expose process-local object addresses.
    """

    state_id: str
    graph_id: str | None
    kind: StateKind
    aliases: tuple[str, ...]
    node_ids: tuple[str, ...]
    owners: tuple[str, ...]
    devices: tuple[tuple[str, str], ...]
    shape: tuple[int, ...]
    dtype: str
    trainable: bool
    optimizer_managed: bool


@dataclass(frozen=True)
class StateMigration:
    """Describe movement and external optimizer responsibilities for one source."""

    state_id: str
    kind: StateKind
    from_owners: tuple[str, ...]
    to_owners: tuple[str, ...]
    source_devices: tuple[tuple[str, str], ...]
    destination_devices: tuple[tuple[str, str], ...]
    identity_preserved: bool | None
    optimizer_action: Literal["none", "keep", "reassociate", "copy_and_reassociate", "unsupported"]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateMigrationPlan:
    """A description only: TorchLens neither moves optimizer state nor steps it."""

    graph_id: str | None
    source_partition: str
    target_partition: str
    entries: tuple[StateMigration, ...]
    unsupported_reasons: tuple[str, ...] = ()
    optimizer_owner: str = "caller"

    @property
    def supported(self) -> bool:
        """Return whether this description found no unresolvable state ownership."""

        return not self.unsupported_reasons


def _require_torch(runtime: Any) -> None:
    """Refuse backends without the effective-state binding protocol."""

    if runtime.adapter.name != "torch":
        raise SplitUnsupportedError(
            "Stable state ownership currently requires the Torch split adapter.",
            context=SplitErrorContext(
                backend=runtime.adapter.name,
                split_point=runtime.request.boundary,
                reason="state_ownership_unsupported",
            ),
        )


def _named_sources(runtime: Any) -> dict[int, tuple[StateKind, tuple[str, ...], Any]]:
    """Map transient source identities to stable model names and tied aliases."""

    from ._torch_compact import live_buffer_value

    sources: dict[int, tuple[StateKind, tuple[str, ...], Any]] = {}
    for kind, method in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        getter = getattr(runtime.model, method, None)
        if not callable(getter):
            continue
        for name, value in getter(remove_duplicate=False):
            prior = sources.get(id(value))
            aliases = (name,) if prior is None else tuple(sorted({*prior[1], name}))
            sources[id(value)] = (kind, aliases, value)  # type: ignore[assignment]

    # Full Buffer records and compact handles use different internal source
    # identities. Both resolve to the same public qualified model name.
    captured = []
    for ref in _captured_buffers(runtime):
        handle = live_buffer_value(ref)
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


def _captured_buffers(runtime: Any) -> list[Any]:
    """Collect direct and parameter-module buffer references from the normalized graph."""

    refs = []
    for node in runtime.trace_graph.nodes:
        refs.extend(node.buffer_refs)
        for parameter in node.param_refs:
            try:
                module = getattr(parameter, "module", None)
            except (AttributeError, KeyError):
                module = None
            refs.extend((getattr(module, "buffers", None) or {}).values())
    return refs


def _node_id(occurrence: str) -> str:
    """Remove the state-occurrence suffix without stripping an op's pass index."""

    if ":literal:" in occurrence:
        return occurrence.rsplit(":literal:", 1)[0]
    return occurrence.removesuffix(":source")


def _inventory(
    runtime: Any, snapshots: dict[str, dict[str, Any]]
) -> tuple[tuple[ParameterOwnership, ...], dict[str, dict[str, Any]]]:
    """Project settled effective bindings into immutable metadata and live mappings."""

    named = _named_sources(runtime)
    raw: dict[int, dict[str, Any]] = {}
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
            item = raw.setdefault(
                entry.source_id,
                {"occurrences": set(), "bindings": {}, "trainable": False},
            )
            item["occurrences"].add(occurrence)
            item["bindings"][public] = value
            item["trainable"] |= entry.trainable

    grouped: dict[str, dict[str, Any]] = {}
    for source_id, item in raw.items():
        source = named.get(source_id)
        kind: StateKind = "literal" if source is None else source[0]
        aliases = () if source is None else source[1]
        identity = min(item["occurrences"]) if source is None else aliases[0]
        state_id = f"{kind}:{identity}"
        row = grouped.setdefault(
            state_id,
            {"kind": kind, "aliases": aliases, "nodes": set(), "bindings": {}, "trainable": False},
        )
        row["nodes"].update(_node_id(key) for key in item["occurrences"])
        row["bindings"].update(item["bindings"])
        row["trainable"] |= item["trainable"]

    rows = []
    bindings = {}
    for state_id, item in sorted(grouped.items()):
        values = item["bindings"]
        value = next(iter(values.values()))
        owners = tuple(sorted(values))
        rows.append(
            ParameterOwnership(
                state_id=state_id,
                graph_id=runtime.trace_graph.graph_shape_hash,
                kind=item["kind"],
                aliases=item["aliases"],
                node_ids=tuple(sorted(item["nodes"])),
                owners=owners,
                devices=tuple((owner, str(values[owner].device)) for owner in owners),
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                trainable=item["trainable"],
                optimizer_managed=item["trainable"],
            )
        )
        bindings[state_id] = values
    return tuple(rows), bindings


def parameter_ownership(runtime: Any) -> tuple[ParameterOwnership, ...]:
    """Describe current executed state without retaining tensors or changing lazy bindings.

    Parameters
    ----------
    runtime:
        Prepared Torch runtime whose current effective state is inspected.

    Returns
    -------
    tuple[ParameterOwnership, ...]
        Deduplicated parameters, used buffers and anonymous literal sources.
        Model state unused by the captured execution is absent.
    """

    _require_torch(runtime)
    with runtime._temporary_segment_state_snapshots() as snapshots:
        rows, _bindings = _inventory(runtime, snapshots)
        return rows


def parameter_bindings(runtime: Any) -> Mapping[str, Mapping[str, Any]]:
    """Bind current parameters and expose stable IDs for caller-owned optimizers.

    Parameters
    ----------
    runtime:
        Torch runtime to bind. Unlike metadata inspection, this settles lazy
        state so returned tensors are the exact objects subsequent replay uses.

    Returns
    -------
    Mapping[str, Mapping[str, Any]]
        Read-only stable ID to owner to effective tensor mapping. Includes
        referenced parameters and owned replicas. Tied owners may refer to the
        same tensor; the caller must deduplicate objects when building an optimizer.
        No gradient is cleared and no optimizer operation is performed.
    """

    _require_torch(runtime)
    rows, bindings = _inventory(runtime, runtime._segment_state_snapshots())
    return MappingProxyType(
        {
            row.state_id: MappingProxyType(bindings[row.state_id])
            for row in rows
            if row.kind == "parameter" or row.trainable
        }
    )


def plan_state_migration(runtime: Any, partition: GraphPartition) -> StateMigrationPlan:
    """Describe state movement and optimizer reassociation without constructing segments.

    Parameters
    ----------
    runtime:
        Source Torch runtime with current effective state.
    partition:
        A legal canonical partition or a compatible point-based cut.

    Returns
    -------
    StateMigrationPlan
        Required movement and optimizer actions. The caller owns optimizer
        history migration (including momentum and Adam statistics); TorchLens
        only inherits effective model values during materialization. Plans are
        observations of current state, not authorization to merge later changes.
    """

    _require_torch(runtime)
    analysis = runtime.analyze(partition)
    with runtime._temporary_segment_state_snapshots() as snapshots:
        rows, bindings = _inventory(runtime, snapshots)
        migrations = []
        unsupported: set[str] = set()
        for row in rows:
            owners = tuple(
                owner
                for owner, nodes in (
                    ("prefix", analysis.plan.prefix_node_ids),
                    ("suffix", analysis.plan.suffix_node_ids),
                )
                if nodes.intersection(row.node_ids)
            )
            values = bindings[row.state_id]
            source_devices = {str(value.device) for value in values.values()}
            destinations = tuple(
                (
                    owner,
                    str(
                        runtime.adapter.normalize_device(
                            analysis.request.placement.for_segment(owner).device
                        )
                    )
                    if analysis.request.placement.for_segment(owner).is_explicit
                    else str(next(iter(values.values())).device),
                )
                for owner in owners
            )
            moved = owners != row.owners or destinations != row.devices
            copied = any(device not in source_devices for _owner, device in destinations)
            ambiguous = (
                row.trainable and moved and len({id(value) for value in values.values()}) > 1
            )
            cross_device_shared = (
                runtime.request.trainable
                and row.trainable
                and len({device for _owner, device in destinations}) > 1
            )
            diagnostics = (
                ("shared_optimizer_state_ambiguous",)
                if ambiguous
                else ("shared_trainable_cross_device",)
                if cross_device_shared
                else ()
            )
            unsupported.update(diagnostics)
            action: Literal["none", "keep", "reassociate", "copy_and_reassociate", "unsupported"]
            if not row.trainable:
                action = "none"
            elif diagnostics:
                action = "unsupported"
            elif copied:
                action = "copy_and_reassociate"
            elif moved:
                action = "reassociate"
            else:
                action = "keep"
            migrations.append(
                StateMigration(
                    state_id=row.state_id,
                    kind=row.kind,
                    from_owners=row.owners,
                    to_owners=owners,
                    source_devices=row.devices,
                    destination_devices=destinations,
                    identity_preserved=None if diagnostics else not copied,
                    optimizer_action=action,
                    diagnostics=diagnostics,
                )
            )
        return StateMigrationPlan(
            graph_id=runtime.trace_graph.graph_shape_hash,
            source_partition=runtime.request.boundary,
            target_partition=analysis.request.boundary,
            entries=tuple(migrations),
            unsupported_reasons=tuple(sorted(unsupported)),
        )


__all__ = [
    "ParameterOwnership",
    "StateMigration",
    "StateMigrationPlan",
    "parameter_bindings",
    "parameter_ownership",
    "plan_state_migration",
]
