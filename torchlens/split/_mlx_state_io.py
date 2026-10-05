"""Named snapshots and coordinated state loading for MLX split runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .adapters._mlx_coordination import snapshot_mlx_values
from .errors import SplitErrorContext, SplitUnsupportedError


@dataclass(frozen=True)
class _MlxStateInventory:
    """Capture-time names and aliases, independent of later source-model replacements."""

    names: dict[str, int]
    defaults: dict[int, Any]


def initialize_mlx_runtime_state(runtime: Any) -> None:
    """Retain the native module's full named state without materializing segment replicas.

    Parameters
    ----------
    runtime
        Newly prepared or migrated MLX split runtime.
    """

    import mlx.nn as nn
    from mlx.utils import tree_flatten

    binding = runtime.segments.prefix._binding
    if binding.coordinator.inventory is not None or not isinstance(runtime.model, nn.Module):
        return
    names: dict[str, int] = {}
    defaults: dict[int, Any] = {}
    for name, value in tree_flatten(runtime.model.parameters()):
        source_id = binding.source_id(value)
        names[name] = source_id
        defaults.setdefault(source_id, value)
    binding.coordinator.inventory = _MlxStateInventory(names, defaults)


def inherit_mlx_runtime_state(runtime: Any, segments: Any) -> None:
    """Carry names and unconsumed loaded state across recuts and placement changes.

    Parameters
    ----------
    runtime
        Runtime supplying its current state.
    segments
        Newly materialized segments whose consumed state has already migrated.
    """

    previous = runtime.segments.prefix._binding.coordinator
    target = segments.prefix._binding.coordinator
    target.inventory = previous.inventory
    for source_id, value in previous.values.items():
        target.values.setdefault(source_id, value)
    target.mutable_sources.update(previous.mutable_sources)


def _inventory(runtime: Any) -> tuple[Any, _MlxStateInventory]:
    """Require a native named state inventory for this runtime."""

    coordinator = runtime.segments.prefix._binding.coordinator
    if not isinstance(coordinator.inventory, _MlxStateInventory):
        raise SplitUnsupportedError(
            "MLX split state_dict/load_state_dict require an mlx.nn.Module source model.",
            context=SplitErrorContext(
                backend="mlx",
                split_point=runtime.request.boundary,
                reason="named_state_unavailable",
            ),
        )
    return coordinator, coordinator.inventory


def mlx_runtime_state_dict(runtime: Any) -> dict[str, Any]:
    """Snapshot effective parameters and buffers, preserving every tied name.

    Parameters
    ----------
    runtime
        MLX split runtime whose current values are to be exported.

    Returns
    -------
    dict[str, Any]
        Evaluated independent arrays keyed by native flattened module paths.
    """

    import mlx.core as mx

    coordinator, inventory = _inventory(runtime)
    effective: dict[int, Any] = {}
    for binding in coordinator.bindings:
        binding.bound_values()
        for entry in binding.state.entries():
            if entry.source_id not in inventory.defaults:
                continue
            previous = effective.setdefault(entry.source_id, entry.value)
            if previous is not entry.value and not bool(
                mx.array_equal(previous, entry.value, equal_nan=True)
            ):
                raise SplitUnsupportedError(
                    "Cannot export divergent replicas of a tied MLX state value; "
                    "load synchronized state before exporting.",
                    context=SplitErrorContext(
                        backend="mlx",
                        split_point=runtime.request.boundary,
                        reason="divergent segment state replicas",
                    ),
                )
    values = {
        source_id: effective.get(source_id, coordinator.values.get(source_id, value))
        for source_id, value in inventory.defaults.items()
    }
    saved = snapshot_mlx_values(values)
    return {name: saved[source_id] for name, source_id in inventory.names.items()}


def load_mlx_runtime_state_dict(runtime: Any, state_dict: Mapping[str, Any]) -> None:
    """Validate a full state snapshot, then synchronize all eager and lazy segment bindings.

    Parameters
    ----------
    runtime
        MLX split runtime to update without writing its source model.
    state_dict
        Full flattened named state with capture-compatible shapes, dtypes, and aliases.
    """

    import mlx.core as mx

    coordinator, inventory = _inventory(runtime)
    context = SplitErrorContext(
        backend="mlx", split_point=runtime.request.boundary, reason="state_dict_invalid"
    )
    if not isinstance(state_dict, Mapping):
        raise SplitUnsupportedError("MLX split state_dict must be a mapping.", context=context)
    supplied = dict(state_dict)
    if supplied.keys() != inventory.names.keys():
        missing = sorted(inventory.names.keys() - supplied.keys())
        unexpected = sorted(supplied.keys() - inventory.names.keys(), key=repr)
        raise SplitUnsupportedError(
            f"MLX split state inventory mismatch: missing={missing!r}, unexpected={unexpected!r}.",
            context=context,
        )
    values: dict[int, Any] = {}
    for name, source_id in inventory.names.items():
        value, expected = supplied[name], inventory.defaults[source_id]
        if (
            not runtime.adapter.is_tensor(value)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise SplitUnsupportedError(
                f"MLX split state {name!r} must retain its captured array shape and dtype.",
                context=context,
            )
        previous = values.setdefault(source_id, value)
        if previous is not value and not bool(mx.array_equal(previous, value, equal_nan=True)):
            raise SplitUnsupportedError(
                f"MLX split tied state {name!r} received conflicting values.", context=context
            )
    coordinator.publish(snapshot_mlx_values(values), parameters=True, invalidate=True)
