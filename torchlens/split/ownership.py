"""Stable state ownership and caller-owned optimizer migration descriptions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from .errors import SplitErrorContext, SplitUnsupportedError
from .ir import GraphPartition
from .placement import PlacementPlan, SegmentName
from .planner import SplitPlan

if TYPE_CHECKING:
    from .runtime import SplitRuntime

StateKind = Literal["parameter", "buffer", "literal"]


@dataclass(frozen=True)
class ParameterOwnership:
    """One unique replay state source, including buffers and anonymous literals.

    Named IDs use qualified model names and retain all tied aliases. Anonymous
    IDs use canonical replay occurrences. IDs are scoped to ``graph_id`` and
    never expose process-local object addresses.

    Examples
    --------
    >>> graph = tl.split.prepare_graph(torch.nn.Linear(3, 2).eval(), torch.ones(1, 3))
    >>> partition = graph.compute_dag().to_partition(())
    >>> analysis = graph.analyze(partition)
    >>> runtime = graph.materialize(analysis)
    >>> owners = runtime.parameter_ownership()
    >>> bool(owners), all(row.kind == 'parameter' for row in owners)
    (True, True)
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
    """Describe movement and external optimizer responsibilities for one source.

    Examples
    --------
    >>> migration = tl.split.StateMigration('parameter:weight', 'parameter',
    ...     ('prefix',), ('suffix',), (('prefix', 'cpu'),), (('suffix', 'cpu'),), True, 'keep')
    >>> migration.identity_preserved, migration.optimizer_action
    (True, 'keep')
    """

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
    """A description only: TorchLens neither moves optimizer state nor steps it.

    Examples
    --------
    >>> graph = tl.split.prepare_graph(torch.nn.Linear(3, 2).eval(), torch.ones(1, 3))
    >>> partition = graph.compute_dag().to_partition(())
    >>> analysis = graph.analyze(partition)
    >>> runtime = graph.materialize(analysis)
    >>> migration = runtime.plan_state_migration(partition)
    >>> migration.supported, migration.optimizer_owner
    (True, 'caller')
    """

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


@dataclass(frozen=True)
class _StateSource:
    """One backend-proven set of live bindings for a named state source."""

    state_id: str
    kind: StateKind
    aliases: tuple[str, ...]
    node_ids: frozenset[str]
    bindings: Mapping[str, Any]
    trainable: bool


def _refuse_ownership(runtime: SplitRuntime, reason: str, message: str) -> NoReturn:
    """Report a missing or ambiguous live-state binding."""

    raise SplitUnsupportedError(
        message,
        context=SplitErrorContext(
            backend=runtime.adapter.name, split_point=runtime.request.boundary, reason=reason
        ),
        code=SplitUnsupportedError.code,
    )


def _inventory(
    runtime: SplitRuntime, snapshots: dict[str, dict[str, Any]]
) -> tuple[tuple[ParameterOwnership, ...], dict[str, Mapping[str, Any]]]:
    """Project adapter sources into immutable rows and effective owner bindings."""

    sources = getattr(runtime.adapter, "state_sources", None)
    if sources is None:
        _refuse_ownership(
            runtime,
            "state_ownership_unsupported",
            "Stable state ownership is unavailable for this split adapter.",
        )
    rows = []
    bindings = {}
    for source in sorted(sources(runtime, snapshots), key=lambda source: source.state_id):
        if source.state_id in bindings:
            _refuse_ownership(
                runtime,
                "state_identity_conflict",
                f"Distinct runtime resources have state ID {source.state_id!r}.",
            )
        values = source.bindings
        value = next(iter(values.values()))
        owners = tuple(sorted(values))
        rows.append(
            ParameterOwnership(
                state_id=source.state_id,
                graph_id=runtime.trace_graph.graph_shape_hash,
                kind=source.kind,
                aliases=source.aliases,
                node_ids=tuple(sorted(source.node_ids)),
                owners=owners,
                devices=tuple(
                    (owner, str(runtime.adapter.device_of(values[owner]))) for owner in owners
                ),
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                trainable=source.trainable,
                optimizer_managed=source.trainable,
            )
        )
        bindings[source.state_id] = values
    return tuple(rows), bindings


@contextmanager
def _inspect_state(
    runtime: SplitRuntime,
) -> Iterator[tuple[tuple[ParameterOwnership, ...], dict[str, Mapping[str, Any]]]]:
    """Read effective bindings while restoring any temporary lazy state."""

    with runtime._temporary_segment_state_snapshots() as snapshots:
        yield _inventory(runtime, snapshots)


def parameter_ownership(runtime: SplitRuntime) -> tuple[ParameterOwnership, ...]:
    """Describe current executed state without retaining tensors or changing lazy bindings.

    Parameters
    ----------
    runtime:
        Prepared split runtime whose current effective state is inspected.

    Returns
    -------
    tuple[ParameterOwnership, ...]
        Deduplicated parameters, used buffers and anonymous literal sources.
        Model state unused by the captured execution is absent.
    """

    with _inspect_state(runtime) as (rows, _bindings):
        return rows


def state_bindings(runtime: SplitRuntime) -> Mapping[str, Mapping[str, Any]]:
    """Bind stable state IDs to effective parameter, buffer and literal tensors.

    Parameters
    ----------
    runtime:
        Split runtime whose lazy segment state is settled before inspection.

    Returns
    -------
    Mapping[str, Mapping[str, Any]]
        Read-only stable state ID to owner to effective tensor mappings. Unlike
        ``parameter_bindings``, this includes non-trainable buffers and literals.
        No gradient, optimizer or source-model value is changed.
    """
    _rows, bindings = _inventory(runtime, runtime._segment_state_snapshots())
    return MappingProxyType(
        {state_id: MappingProxyType(values) for state_id, values in bindings.items()}
    )


def parameter_bindings(runtime: SplitRuntime) -> Mapping[str, Mapping[str, Any]]:
    """Bind current parameters and expose stable IDs for caller-owned optimizers.

    Parameters
    ----------
    runtime:
        Split runtime to bind. Unlike metadata inspection, this settles lazy
        state so returned tensors are the exact objects subsequent replay uses.

    Returns
    -------
    Mapping[str, Mapping[str, Any]]
        Read-only stable ID to owner to effective tensor mapping. Includes
        referenced parameters and owned replicas. Tied owners may refer to the
        same tensor; the caller must deduplicate objects when building an optimizer.
        No gradient is cleared and no optimizer operation is performed.
    """

    rows, bindings = _inventory(runtime, runtime._segment_state_snapshots())
    return MappingProxyType(
        {
            row.state_id: MappingProxyType(bindings[row.state_id])
            for row in rows
            if row.kind == "parameter" or row.trainable
        }
    )


def plan_state_migration(runtime: SplitRuntime, partition: GraphPartition) -> StateMigrationPlan:
    """Describe state movement and optimizer reassociation without constructing segments.

    Parameters
    ----------
    runtime:
        Source split runtime with current effective state.
    partition:
        A legal canonical compute-node partition.

    Returns
    -------
    StateMigrationPlan
        Required movement and optimizer actions. The caller owns optimizer
        history migration (including momentum and Adam statistics); TorchLens
        only inherits effective model values during materialization. Plans are
        observations of current state, not authorization to merge later changes.
    """

    analysis = runtime.analyze(partition)
    with _inspect_state(runtime) as (rows, bindings):
        migrations = tuple(
            _state_transition(
                runtime, analysis.request.placement, analysis.plan, row, bindings[row.state_id]
            )
            for row in rows
        )
        return StateMigrationPlan(
            graph_id=runtime.trace_graph.graph_shape_hash,
            source_partition=runtime.request.boundary,
            target_partition=analysis.request.boundary,
            entries=migrations,
            unsupported_reasons=tuple(
                sorted({reason for row in migrations for reason in row.diagnostics})
            ),
        )


def _destination_devices(
    runtime: SplitRuntime, placement: PlacementPlan, owners: tuple[SegmentName, ...], source: str
) -> tuple[tuple[str, str], ...]:
    """Resolve each target's declared placement with the original device as fallback."""

    devices = []
    for owner in owners:
        target = placement.for_segment(owner)
        device = (
            str(getattr(runtime.adapter, "normalize_device")(target.device))
            if target.is_explicit
            else source
        )
        devices.append((owner, device))
    return tuple(devices)


def _optimizer_diagnostics(
    row: ParameterOwnership,
    values: Mapping[str, Any],
    devices: set[str],
    *,
    moved: bool,
    training: bool,
) -> tuple[str, ...]:
    """Refuse a reassignment when independently owned trainable state is ambiguous."""

    if not row.trainable:
        return ()
    if moved and len(values) > 1 and values["prefix"] is not values["suffix"]:
        return ("shared_optimizer_state_ambiguous",)
    if training and len(devices) > 1:
        return ("shared_trainable_cross_device",)
    return ()


def _state_transition(
    runtime: SplitRuntime,
    placement: PlacementPlan,
    plan: SplitPlan,
    row: ParameterOwnership,
    values: Mapping[str, Any],
) -> StateMigration:
    """Describe one parameter, buffer, or literal's effective binding after a recut."""

    owners = _owners_for_nodes(plan, row.node_ids)
    destinations = _destination_devices(runtime, placement, owners, row.devices[0][1])
    source_devices = {device for _, device in row.devices}
    target_devices = {device for _, device in destinations}
    moved = owners != row.owners or destinations != row.devices
    copied = bool(target_devices - source_devices)
    diagnostics = _optimizer_diagnostics(
        row,
        values,
        target_devices,
        moved=moved,
        training=runtime.request.trainable,
    )
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
    return StateMigration(
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


def _owners_for_nodes(
    plan: SplitPlan, node_ids: tuple[str, ...] | frozenset[str]
) -> tuple[SegmentName, ...]:
    """Return the sides executing a source's recorded occurrences."""

    segments: tuple[tuple[SegmentName, frozenset[str]], ...] = (
        ("prefix", plan.prefix_node_ids),
        ("suffix", plan.suffix_node_ids),
    )
    return tuple(owner for owner, nodes in segments if nodes.intersection(node_ids))


__all__ = (
    "ParameterOwnership",
    "StateMigration",
    "StateMigrationPlan",
    "parameter_bindings",
    "parameter_ownership",
    "plan_state_migration",
)
