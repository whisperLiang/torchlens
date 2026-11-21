"""Plan, export and transactionally restore optimizer state through SplitRuntime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import astuple, dataclass, replace
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..._state import pause_logging
from ..errors import SplitUnsupportedError
from ..ir import GraphPartition
from ..ownership import ParameterOwnership, _inventory, _owners_for_nodes
from .base import (
    _GLOBAL,
    _SCHEMA_VERSION,
    OptimizerFactory,
    OptimizerState,
    OptimizerStateAdapter,
    OptimizerStateMigration,
    OptimizerStateMigrationPlan,
    OptimizerStateMigrationResult,
    OptimizerStateSnapshot,
    OptimizerStateTensor,
    _Action,
    _json_bytes,
    _ParameterSpec,
    _refuse,
)
from .registry import _adapter_by_name, _has_builtin_factory, get_optimizer_adapter

if TYPE_CHECKING:
    from ..runtime import SplitRuntime


@dataclass(frozen=True)
class _PreparedMigration:
    """One coherent observation of source state and target optimizer assignments."""

    plan: OptimizerStateMigrationPlan
    states: dict[str, OptimizerState]
    adapters: dict[str, str]
    specs: dict[str, _ParameterSpec]


def _check_runtime(runtime: SplitRuntime) -> str:
    """Require an identified graph with live training enabled."""

    graph_id = runtime.graph_identity
    if not runtime.request.trainable or graph_id is None:
        _refuse(
            "optimizer_training_required",
            "Optimizer migration requires an identified training graph.",
            runtime.adapter.name,
            runtime.request.boundary,
        )
    return graph_id


def _routes(
    rows: tuple[ParameterOwnership, ...], target: Mapping[str, tuple[str, ...]]
) -> tuple[OptimizerStateMigration, ...]:
    """Derive unique optimizer ownership directly from state consumers."""

    entries = []
    for row in rows:
        destination = target[row.state_id]
        if row.kind != "parameter" or len(row.owners) != 1 or len(destination) != 1:
            entries.append(
                OptimizerStateMigration(
                    row.state_id, "+".join(row.owners), "+".join(destination), "unsupported", ()
                )
            )
        else:
            before, after = row.owners[0], destination[0]
            entries.append(
                OptimizerStateMigration(
                    row.state_id, before, after, "keep" if before == after else "transfer", ()
                )
            )
    return tuple(entries)


def _by_owner(
    rows: tuple[ParameterOwnership, ...], bindings: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return native parameter bindings for nonempty optimizer owners."""

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        for owner, value in bindings[row.state_id].items():
            out.setdefault(owner, {})[row.state_id] = value
    return out


def _tensor_index(
    tensors: tuple[OptimizerStateTensor, ...], state_id: str, *, validate: bool
) -> dict[str, str]:
    """Index checksums while rejecting duplicate or misrouted payloads."""

    found = {}
    for tensor in tensors:
        if tensor.state_id != state_id or tensor.slot_name in found:
            _refuse("optimizer_slot_inventory", f"Invalid slot identity for {state_id!r}.")
        if validate:
            tensor._validate()
        found[tensor.slot_name] = tensor.checksum
    return found


def _state_inventory(
    state: OptimizerState, parameters: Mapping[str, Any], *, validate: bool = False
) -> None:
    """Enforce complete parameter and slot inventories before rebuilding state."""

    if state.expected_slots.keys() != parameters.keys() or state.slots.keys() != parameters.keys():
        _refuse(
            "optimizer_parameter_inventory",
            "Optimizer state parameter IDs differ from the target bindings.",
        )
    _tensor_index(state.global_state, _GLOBAL, validate=validate)
    for state_id, names in state.expected_slots.items():
        slots = _tensor_index(state.slots[state_id], state_id, validate=validate)
        if len(names) != len(set(names)) or slots and slots.keys() != set(names):
            _refuse(
                "optimizer_slot_inventory", f"Missing or unknown optimizer slots for {state_id!r}."
            )


def _state_manifest(state: OptimizerState) -> tuple[Any, ...]:
    """Bind configuration, parameters, and every state checksum to the snapshot."""

    return (
        state.optimizer_type,
        state.config,
        {slot.slot_name: slot.checksum for slot in state.global_state},
        state.expected_slots,
        {
            key: {slot.slot_name: slot.checksum for slot in slots}
            for key, slots in state.slots.items()
        },
    )


def _digest(value: Any) -> str:
    """Hash a canonical metadata tree."""

    return sha256(_json_bytes(value)).hexdigest()


def _source_states(
    runtime: SplitRuntime, optimizers: Mapping[str, Any], bindings: dict[str, dict[str, Any]]
) -> tuple[dict[str, OptimizerStateAdapter], dict[str, OptimizerState]]:
    """Read each supplying optimizer's history once by exact native parameter identity."""

    if bindings.keys() != optimizers.keys() or len({id(opt) for opt in optimizers.values()}) != len(
        optimizers
    ):
        _refuse(
            "optimizer_parameter_ownership",
            "Provide one distinct optimizer for each nonempty parameter owner (prefix or suffix).",
        )
    adapters = {}
    sources = {}
    for owner, optimizer in optimizers.items():
        adapter = get_optimizer_adapter(runtime.adapter.name, optimizer)
        state = adapter.export_state(optimizer, bindings[owner])
        _state_inventory(state, bindings[owner])
        adapters[owner], sources[owner] = adapter, state
    return adapters, sources


def _compatible_global(sources: Mapping[str, OptimizerState]) -> tuple[OptimizerStateTensor, ...]:
    """Require exact global state agreement across contributors to a target."""

    first = next(iter(sources.values()))
    reference = _tensor_index(first.global_state, _GLOBAL, validate=False)
    for state in sources.values():
        actual = _tensor_index(state.global_state, _GLOBAL, validate=False)
        different = sorted(
            key for key in reference.keys() | actual.keys() if reference.get(key) != actual.get(key)
        )
        if different:
            _refuse(
                "optimizer_global_state_incompatible",
                f"Source optimizer global state differs for {different!r}.",
            )
    return first.global_state


def _target_states(
    entries: tuple[OptimizerStateMigration, ...],
    adapters: Mapping[str, OptimizerStateAdapter],
    sources: Mapping[str, OptimizerState],
) -> tuple[dict[str, OptimizerState], dict[str, str]]:
    """Join compatible sources and route distinct parameter histories to each owner."""

    by_target: dict[str, dict[str, str]] = {}
    for entry in entries:
        by_target.setdefault(entry.target_optimizer, {})[entry.state_id] = entry.source_optimizer
    states = {}
    adapter_names = {}
    for owner, routes in by_target.items():
        contributing = {key: sources[key] for key in sorted(set(routes.values()))}
        first = next(iter(contributing))
        adapter = adapters[first]
        signature = {
            (adapters[key].name, state.optimizer_type) for key, state in contributing.items()
        }
        if len(signature) != 1:
            _refuse(
                "optimizer_type_incompatible", f"Incompatible source optimizer types for {owner!r}."
            )
        globals_ = _compatible_global(contributing)
        config = adapter.merge_config(contributing, routes)
        states[owner] = OptimizerState(
            sources[first].optimizer_type,
            config,
            globals_,
            {key: sources[src].expected_slots[key] for key, src in routes.items()},
            {key: sources[src].slots[key] for key, src in routes.items()},
        )
        adapter_names[owner] = adapter.name
    return states, adapter_names


def _filled_entry(entry: OptimizerStateMigration, state: OptimizerState) -> OptimizerStateMigration:
    """Mark uninitialized native histories in an otherwise valid plan."""

    names = state.expected_slots[entry.state_id]
    action = "initialize" if names and not state.slots[entry.state_id] else entry.action
    return replace(entry, action=action, slot_names=names)


def _prepare(
    runtime: SplitRuntime, partition: GraphPartition, optimizers: Mapping[str, Any]
) -> _PreparedMigration:
    """Observe the cut, validate external optimizers, and build a read-only plan."""

    if not isinstance(partition, GraphPartition):
        runtime.analyze(partition)  # Reuse the public argument check.
    plan = OptimizerStateMigrationPlan(
        runtime.adapter.name,
        runtime.graph_identity or "",
        runtime.request.boundary,
        partition.as_boundary(),
        (),
        _partition=partition,
    )
    entries: tuple[OptimizerStateMigration, ...] = ()
    try:
        _check_runtime(runtime)
        with pause_logging(), runtime._temporary_segment_state_snapshots() as snapshots:
            all_rows, bindings = _inventory(runtime, snapshots)
            analysis = runtime._analyze_partition(partition, state_snapshots=snapshots)
            rows = tuple(row for row in all_rows if row.trainable)
            destinations = {
                row.state_id: _owners_for_nodes(analysis.plan, row.node_ids) for row in rows
            }
            entries = _routes(rows, destinations)
            ambiguous = [row.state_id for row in entries if row.action == "unsupported"]
            if ambiguous:
                _refuse(
                    "optimizer_parameter_ownership",
                    f"Optimizer migration needs unique named parameter ownership; ambiguous IDs {ambiguous!r}.",
                )
            adapters, sources = _source_states(runtime, optimizers, _by_owner(rows, bindings))
            states, names = _target_states(entries, adapters, sources)
            specs: dict[str, _ParameterSpec] = {
                row.state_id: (destinations[row.state_id][0], row.shape, row.dtype) for row in rows
            }
            entries = tuple(_filled_entry(row, states[row.target_optimizer]) for row in entries)
    except SplitUnsupportedError as exc:
        reason = exc.code if exc.context is None else exc.context.reason
        refused = replace(
            plan,
            unsupported_reasons=(f"{reason}: {exc}",),
            entries=tuple(replace(row, action="unsupported") for row in entries),
        )
        return _PreparedMigration(refused, {}, {}, {})
    stamp = _digest(
        (
            {key: (adapters[key].name, _state_manifest(value)) for key, value in sources.items()},
            specs,
        )
    )
    return _PreparedMigration(
        replace(plan, entries=entries, _source_checksum=stamp), states, names, specs
    )


def _plan_optimizer_state_migration(
    runtime: SplitRuntime, partition: GraphPartition, optimizers: Mapping[str, Any]
) -> OptimizerStateMigrationPlan:
    """Plan an optimizer transition without changing either training owner."""

    return _prepare(runtime, partition, optimizers).plan


def _snapshot_checksum(snapshot: OptimizerStateSnapshot) -> str:
    """Bind the snapshot header and slot manifest independently of device placement."""

    return _digest(
        (
            snapshot.schema_version,
            snapshot.backend,
            snapshot.graph_id,
            snapshot.source_partition,
            snapshot.target_partition,
            tuple(astuple(entry) for entry in snapshot.entries),
            snapshot.adapter_names,
            snapshot.parameter_specs,
            {key: _state_manifest(state) for key, state in snapshot.optimizers.items()},
        )
    )


def _export_optimizer_state(
    runtime: SplitRuntime, optimizers: Mapping[str, Any], plan: OptimizerStateMigrationPlan
) -> OptimizerStateSnapshot:
    """Reattest the source plan and export only current participating state."""

    if not plan.supported or plan._partition is None:
        _refuse(
            "optimizer_plan_unsupported",
            f"Cannot export an unsupported plan: {plan.unsupported_reasons!r}.",
        )
    if (plan.backend, plan.graph_id, plan.source_partition) != (
        runtime.adapter.name,
        _check_runtime(runtime),
        runtime.request.boundary,
    ):
        _refuse(
            "optimizer_plan_source",
            "Optimizer plan belongs to a different source graph or partition.",
        )
    prepared = _prepare(runtime, plan._partition, optimizers)
    if prepared.plan != plan:
        _refuse(
            "optimizer_source_changed",
            f"Optimizer source state changed since planning: {prepared.plan.unsupported_reasons!r}.",
        )
    snapshot = OptimizerStateSnapshot(
        _SCHEMA_VERSION,
        plan.backend,
        plan.graph_id,
        plan.source_partition,
        plan.target_partition,
        plan.entries,
        MappingProxyType(prepared.states),
        MappingProxyType(prepared.adapters),
        MappingProxyType(prepared.specs),
        "",
    )
    return replace(snapshot, checksum=_snapshot_checksum(snapshot))


def _validate_header(runtime: SplitRuntime, snapshot: OptimizerStateSnapshot) -> None:
    """Reject an unsupported schema or a snapshot for another backend, graph, or cut."""

    expected = {
        "schema_version": _SCHEMA_VERSION,
        "backend": runtime.adapter.name,
        "graph_id": _check_runtime(runtime),
        "target_partition": runtime.request.boundary,
    }
    for field, value in expected.items():
        if getattr(snapshot, field) != value:
            _refuse(
                f"optimizer_snapshot_{field}",
                f"Optimizer snapshot {field} does not match the target runtime.",
            )


def _validate_entries(snapshot: OptimizerStateSnapshot) -> None:
    """Verify that the manifest accounts for all restored and initialized slots."""

    if sorted(row.state_id for row in snapshot.entries) != sorted(snapshot.parameter_specs):
        _refuse(
            "optimizer_parameter_inventory",
            "Optimizer migration entries are incomplete or duplicated.",
        )
    for row in snapshot.entries:
        owner = snapshot.parameter_specs[row.state_id][0]
        if row.target_optimizer != owner or row.source_optimizer not in {"prefix", "suffix"}:
            _refuse("optimizer_parameter_ownership", f"Wrong optimizer owner for {row.state_id!r}.")
        state = snapshot.optimizers[owner]
        action: _Action = "keep" if owner == row.source_optimizer else "transfer"
        if row != _filled_entry(replace(row, action=action), state):
            _refuse(
                "optimizer_slot_inventory", f"Invalid optimizer slot manifest for {row.state_id!r}."
            )


def _validate_snapshot(
    snapshot: OptimizerStateSnapshot,
    rows: tuple[ParameterOwnership, ...],
    by_owner: Mapping[str, Mapping[str, Any]],
) -> dict[str, OptimizerStateAdapter]:
    """Check all state integrity and native target metadata before any construction."""

    specs = {
        row.state_id: (row.owners[0], row.shape, row.dtype) for row in rows if len(row.owners) == 1
    }
    if len(specs) != len(rows) or snapshot.parameter_specs.keys() != specs.keys():
        _refuse(
            "optimizer_parameter_inventory",
            "Snapshot parameter IDs or ownership do not match the target.",
        )
    if snapshot.parameter_specs != specs:
        _refuse(
            "optimizer_parameter_signature", "Snapshot target ownership, shape, or dtype differs."
        )
    if not (snapshot.optimizers.keys() == snapshot.adapter_names.keys() == by_owner.keys()):
        _refuse(
            "optimizer_parameter_ownership", "Snapshot is missing a target optimizer or adapter."
        )
    adapters = {}
    for name, state in snapshot.optimizers.items():
        _state_inventory(state, by_owner[name], validate=True)
        adapter = _adapter_by_name(snapshot.backend, snapshot.adapter_names[name])
        try:
            adapter.validate_restore(state, by_owner[name])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _refuse(
                "optimizer_config_unsupported",
                f"Invalid optimizer configuration for {name!r}: {exc}",
                backend=snapshot.backend,
                cause=exc,
            )
        adapters[name] = adapter
    _validate_entries(snapshot)
    if snapshot.checksum != _snapshot_checksum(snapshot):
        _refuse("optimizer_snapshot_checksum", "Optimizer snapshot manifest checksum mismatch.")
    return adapters


def _check_factories(
    snapshot: OptimizerStateSnapshot, factories: Mapping[str, OptimizerFactory]
) -> None:
    """Check factory coverage for exactly the relevant target optimizer owners."""

    if factories.keys() - snapshot.optimizers.keys():
        _refuse(
            "optimizer_factory_ownership",
            "Factory provided for an optimizer with no target parameters.",
        )
    required = {
        owner
        for owner, name in snapshot.adapter_names.items()
        if not _has_builtin_factory(snapshot.backend, name)
    }
    if missing := required - factories.keys():
        _refuse(
            "optimizer_factory_required",
            f"Custom optimizer adapters require factories for {sorted(missing)!r}.",
        )


def _stage_optimizers(
    snapshot: OptimizerStateSnapshot,
    adapters: Mapping[str, OptimizerStateAdapter],
    by_owner: Mapping[str, Mapping[str, Any]],
    factories: Mapping[str, OptimizerFactory],
) -> dict[str, Any]:
    """Build all optimizers first, then restore only their independently owned state."""

    _check_factories(snapshot, factories)
    try:
        staged = {
            name: adapters[name].build_optimizer(
                state, MappingProxyType(by_owner[name]), factories.get(name)
            )
            for name, state in snapshot.optimizers.items()
        }
        if len({id(value) for value in staged.values()}) != len(staged):
            _refuse("optimizer_factory_not_fresh", "Each target needs a distinct fresh optimizer.")
        for name, state in snapshot.optimizers.items():
            adapters[name].restore_state(staged[name], state, by_owner[name])
    except SplitUnsupportedError:
        raise
    except Exception as exc:
        _refuse(
            "optimizer_restore_failed",
            f"Native optimizer construction or restoration failed: {exc}",
            snapshot.backend,
            cause=exc,
        )
    return staged


def _restore_optimizer_state(
    runtime: SplitRuntime,
    snapshot: OptimizerStateSnapshot,
    factories: Mapping[str, OptimizerFactory] | None,
) -> OptimizerStateMigrationResult:
    """Preflight all target state and publish only fully restored optimizers."""

    _validate_header(runtime, snapshot)
    with (
        pause_logging(),
        runtime._temporary_segment_state_snapshots(keep_on_success=True) as snapshots,
    ):
        all_rows, bindings = _inventory(runtime, snapshots)
        rows = tuple(row for row in all_rows if row.trainable)
        by_owner = _by_owner(rows, bindings)
        adapters = _validate_snapshot(snapshot, rows, by_owner)
        staged = _stage_optimizers(
            snapshot, adapters, by_owner, {} if factories is None else factories
        )
    restored = tuple(row.state_id for row in snapshot.entries if row.action != "initialize")
    return OptimizerStateMigrationResult(
        MappingProxyType(staged),
        restored,
        sum(len(slots) for state in snapshot.optimizers.values() for slots in state.slots.values()),
        sum(len(row.slot_names) for row in snapshot.entries if row.action == "initialize"),
        MappingProxyType(
            {
                action: sum(row.action == action for row in snapshot.entries)
                for action in ("keep", "transfer", "initialize", "unsupported")
            }
        ),
    )
