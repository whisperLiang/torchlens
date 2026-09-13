"""Canonical-batch capture and batch-axis helpers.

TorchLens captures a model once at a small canonical batch (prefer ``B=1``,
fallback ``B=2``) and uses one ``B=2`` probe to authorize empirical batch
extrapolation.  This module owns that capture-time policy: it infers
batch-axis semantics from example inputs, rebatches nested input trees, and
never encodes an allowed runtime range.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Literal

from .errors import SplitUnsupportedError
from .shape_program import _escape_pointer, flatten_input_leaves

if TYPE_CHECKING:
    from .adapters.base import SplitBackendAdapter

CANONICAL_TRACE_BATCH = 1
FALLBACK_TRACE_BATCH = 2
# One counterpart sample supplies shape evidence and checks replay numerics.
# It does not prove correctness at untested batch sizes or hidden Python branches.
_WITNESS_PROBES_FROM_ONE = (2,)


@dataclass(frozen=True)
class BatchSpec:
    """Batch-axis semantics for one capture, without an allowed-value range.

    Parameters
    ----------
    axes:
        JSON Pointer path → batch axis index for every batched tensor leaf.
    inference:
        ``explicit`` when the caller declared axes; ``auto`` when they were
        inferred from top-level leading dimensions.
    user_batch_size:
        Leading batch observed on the caller's example inputs, if any.
    canonical_batch_size:
        Batch used for the capture that produced the reusable graph.
    """

    axes: Mapping[str, int]
    inference: Literal["auto", "explicit"] = "auto"
    user_batch_size: int | None = None
    canonical_batch_size: int = CANONICAL_TRACE_BATCH

    def __post_init__(self) -> None:
        """Validate axis declarations."""

        object.__setattr__(self, "axes", dict(self.axes))
        if self.canonical_batch_size < 1:
            raise ValueError("canonical_batch_size must be a positive integer")
        if self.user_batch_size is not None and self.user_batch_size < 1:
            raise ValueError("user_batch_size must be a positive integer when set")
        for path, axis in self.axes.items():
            if not isinstance(path, str) or not (
                path.startswith("/args/") or path.startswith("/kwargs/")
            ):
                raise ValueError("batch axis keys must be JSON Pointers rooted at /args or /kwargs")
            if not isinstance(axis, int):
                raise TypeError("batch axis values must be integer axis indexes")


def infer_user_batch_size(
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    *,
    adapter: SplitBackendAdapter,
    explicit_axes: Mapping[str, int] | None = None,
) -> int | None:
    """Return the unique batch size present on declared or inferred input axes."""

    leaves = flatten_input_leaves(inputs, input_kwargs, adapter=adapter)
    if not leaves:
        return None
    by_path = {leaf.path: leaf for leaf in leaves}
    axes = (
        infer_auto_axes_from_example(inputs, input_kwargs, adapter=adapter)
        if explicit_axes is None
        else dict(explicit_axes)
    )
    sizes: set[int] = set()
    for path, axis in axes.items():
        leaf = by_path.get(path)
        if leaf is None:
            continue
        shape = adapter.shape(leaf.value)
        if shape is None:
            continue
        normalized = axis if axis >= 0 else len(shape) + axis
        if 0 <= normalized < len(shape):
            sizes.add(int(shape[normalized]))
    return next(iter(sizes)) if len(sizes) == 1 else None


def rebatch_tree(
    value: Any,
    path: str,
    *,
    axes: Mapping[str, int],
    batch_size: int,
    adapter: SplitBackendAdapter,
    tensor_values: Mapping[str, Any] | None = None,
) -> Any:
    """Clone ``value`` while resizing declared batch tensor leaves."""

    if adapter.is_tensor(value):
        if tensor_values is not None:
            return tensor_values[path]
        if path in axes:
            return adapter.resize_batch(value, axes[path], batch_size)
        return adapter.clone(value)
    if isinstance(value, Mapping):
        resized = {
            key: rebatch_tree(
                item,
                f"{path}/{_escape_pointer(str(key))}",
                axes=axes,
                batch_size=batch_size,
                adapter=adapter,
                tensor_values=tensor_values,
            )
            for key, item in value.items()
        }
        if isinstance(value, dict):
            return type(value)(resized)
        return resized
    if isinstance(value, (list, tuple)):
        items = [
            rebatch_tree(
                item,
                f"{path}/{index}",
                axes=axes,
                batch_size=batch_size,
                adapter=adapter,
                tensor_values=tensor_values,
            )
            for index, item in enumerate(value)
        ]
        return type(value)(*items) if hasattr(type(value), "_fields") else type(value)(items)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            item.name: rebatch_tree(
                getattr(value, item.name),
                f"{path}/{_escape_pointer(item.name)}",
                axes=axes,
                batch_size=batch_size,
                adapter=adapter,
                tensor_values=tensor_values,
            )
            for item in fields(value)
        }
        return clone_dataclass_fields(value, updates)
    return value


def clone_dataclass_fields(value: Any, updates: Mapping[str, Any]) -> Any:
    """Copy dataclass state without re-executing its constructor.

    Parameters
    ----------
    value:
        Dataclass instance, including frozen Equinox modules and slotted types.
    updates:
        Cloned or resized field values. Custom constructors need not accept
        these field names, and init=False fields must also retain their state.

    Returns
    -------
    Any
        Independent instance with the original non-field state preserved.
    """

    cloned = copy(value)
    if cloned is value:
        raise SplitUnsupportedError("Cannot independently clone dataclass input state.")
    for name, item in updates.items():
        object.__setattr__(cloned, name, item)
    return cloned


def _rebatch_tensor_leaves(
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    *,
    axes: Mapping[str, int],
    batch_size: int,
    adapter: SplitBackendAdapter,
) -> dict[str, Any]:
    """Resize one input family while preserving supported shared-storage views.

    Repeated objects share one replacement. Torch views with identical byte
    geometry share a replacement storage but retain distinct object identity.
    Overlapping views with different geometry, unknown overlap, or conflicting
    axis declarations refuse before any model execution; independently resizing
    those inputs could change in-place reads and Python control flow.
    """

    from .._state import pause_logging

    leaves = flatten_input_leaves(inputs, input_kwargs, adapter=adapter)
    alias_of: dict[str, str] = {}
    with pause_logging():
        if adapter.name == "torch":
            from ..utils.alias_footprint import tensor_byte_footprint, touched_bytes_relation

            footprints = {leaf.path: tensor_byte_footprint(leaf.value) for leaf in leaves}
        for index, leaf in enumerate(leaves):
            identity_match = next(
                (previous for previous in leaves[:index] if previous.value is leaf.value), None
            )
            for previous in [identity_match] if identity_match is not None else leaves[:index]:
                same_object = leaf.value is previous.value
                if same_object:
                    compatible = True
                elif adapter.name == "torch":
                    left = footprints[leaf.path]
                    right = footprints[previous.path]
                    if (
                        left is not None
                        and right is not None
                        and (
                            left.device_key != right.device_key
                            or left.numel == 0
                            or right.numel == 0
                            or left.end_byte <= right.start_byte
                            or right.end_byte <= left.start_byte
                        )
                    ):
                        continue
                    relation = touched_bytes_relation(leaf.value, previous.value)
                    if relation == "disjoint":
                        continue
                    compatible = (
                        relation == "overlap"
                        and left is not None
                        and right is not None
                        and left.origin_byte == right.origin_byte
                        and left.shape == right.shape
                        and left.strides == right.strides
                        and leaf.value.dtype == previous.value.dtype
                    )
                else:
                    continue
                if not compatible or axes.get(leaf.path) != axes.get(previous.path):
                    raise SplitUnsupportedError(
                        "Cannot preserve input alias relationship while rebatching "
                        f"{previous.path!r} and {leaf.path!r}; overlapping views must have "
                        "identical geometry and batch axes. Use batch_axes={} for fixed-shape replay."
                    )
                alias_of[leaf.path] = previous.path
                break

        by_path = {leaf.path: leaf.value for leaf in leaves}
        replacements: dict[str, Any] = {}
        for leaf in leaves:
            source_path = alias_of.get(leaf.path)
            if source_path is None:
                replacements[leaf.path] = (
                    adapter.resize_batch(leaf.value, axes[leaf.path], batch_size)
                    if leaf.path in axes
                    else adapter.clone(leaf.value)
                )
            elif leaf.value is by_path[source_path]:
                replacements[leaf.path] = replacements[source_path]
            else:
                # The geometry check above only admits Torch views here.
                replacement = replacements[source_path].view_as(replacements[source_path])
                if not leaf.value.requires_grad:
                    replacement = replacement.detach()
                elif not replacement.requires_grad:
                    replacement.requires_grad_(True)
                replacements[leaf.path] = replacement
        return replacements


def rebatch_inputs(
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    *,
    axes: Mapping[str, int],
    batch_size: int,
    adapter: SplitBackendAdapter,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resize declared batch leaves together, preserving supported input aliases."""

    tensor_values = _rebatch_tensor_leaves(
        inputs, input_kwargs, axes=axes, batch_size=batch_size, adapter=adapter
    )
    resized_inputs = tuple(
        rebatch_tree(
            value,
            f"/args/{index}",
            axes=axes,
            batch_size=batch_size,
            adapter=adapter,
            tensor_values=tensor_values,
        )
        for index, value in enumerate(inputs)
    )
    resized_kwargs = {
        key: rebatch_tree(
            value,
            f"/kwargs/{_escape_pointer(str(key))}",
            axes=axes,
            batch_size=batch_size,
            adapter=adapter,
            tensor_values=tensor_values,
        )
        for key, value in dict(input_kwargs or {}).items()
    }
    return resized_inputs, resized_kwargs


def infer_explicit_axes_from_example(
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    *,
    adapter: SplitBackendAdapter,
    explicit_axes: Mapping[str, int],
) -> dict[str, int]:
    """Normalize caller-declared batch axes against the example input tree."""

    leaves = flatten_input_leaves(inputs, input_kwargs, adapter=adapter)
    by_path = {leaf.path: leaf for leaf in leaves}
    unknown = tuple(path for path in explicit_axes if path not in by_path)
    if unknown:
        raise SplitUnsupportedError(
            f"Unknown batch input paths: {unknown!r}; available paths are {tuple(by_path)!r}."
        )
    axes: dict[str, int] = {}
    for path, axis in explicit_axes.items():
        shape = adapter.shape(by_path[path].value)
        if shape is None:
            raise SplitUnsupportedError(f"Batch input {path!r} is not tensor-like.")
        normalized = axis if axis >= 0 else len(shape) + axis
        if not 0 <= normalized < len(shape):
            raise SplitUnsupportedError(f"Batch axis {axis} is invalid for {path!r} shape {shape}.")
        axes[path] = normalized
    return axes


def infer_auto_axes_from_example(
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    *,
    adapter: SplitBackendAdapter,
) -> dict[str, int]:
    """Infer only unambiguous top-level tensor batch inputs from the example.

    Nested tensor leaves are never auto-inferred: the caller must declare
    them. Every top-level tensor must have rank at least two and share the
    same leading extent. A vector's only axis may be features, so rank-one
    batches require explicit axes. Ambiguous inputs yield an empty mapping.
    """

    leaves = flatten_input_leaves(inputs, input_kwargs, adapter=adapter)
    top_level = [leaf for leaf in leaves if leaf.path.count("/") == 2]
    if not top_level:
        return {}
    leading: dict[str, int] = {}
    sizes: set[int] = set()
    for leaf in top_level:
        shape = adapter.shape(leaf.value)
        if shape is None or len(shape) < 2:
            return {}
        leading[leaf.path] = 0
        sizes.add(int(shape[0]))
    if len(sizes) != 1:
        return {}
    return leading


def resolve_batch_spec(
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    *,
    adapter: SplitBackendAdapter,
    explicit_axes: Mapping[str, int] | None = None,
) -> BatchSpec:
    """Resolve batch-axis semantics from the caller's own example inputs.

    Axis semantics are derived from the EXAMPLE the caller supplied, not from
    the small canonical capture: a canonical ``B=1`` capture makes every
    length-one axis look like a batch, so attribution happens while the
    original batch signal is still present.
    ``None`` selects auto-inference; an empty mapping explicitly keeps all
    input shapes static and never triggers canonical rebatching or a probe.
    """

    leaves = flatten_input_leaves(inputs, input_kwargs, adapter=adapter)
    by_path = {leaf.path: leaf for leaf in leaves}
    if explicit_axes is not None:
        axes = infer_explicit_axes_from_example(
            inputs,
            input_kwargs,
            adapter=adapter,
            explicit_axes=explicit_axes,
        )
        if not axes:
            return BatchSpec(axes={}, inference="explicit", user_batch_size=None)
        declared_sizes = {
            tuple(adapter.shape(by_path[path].value) or ())[axis] for path, axis in axes.items()
        }
        if len(declared_sizes) != 1:
            raise SplitUnsupportedError(
                "Declared batch inputs disagree in the example inputs: "
                f"{ {path: adapter.shape(by_path[path].value) for path in axes}!r}."
            )
        user_batch = int(next(iter(declared_sizes)))
        undeclared: dict[str, tuple[int, ...]] = {}
        for path, leaf in by_path.items():
            if path in axes:
                continue
            shape = tuple(adapter.shape(leaf.value) or ())
            candidates = tuple(index for index, dim in enumerate(shape) if dim == user_batch)
            if candidates:
                undeclared[path] = candidates
        if undeclared:
            raise SplitUnsupportedError(
                "Explicit batch inputs are incomplete: undeclared tensor paths contain the "
                f"example batch dimension {user_batch}: {undeclared!r}. Declare these paths "
                "in SplitFeatures.batch_axes."
            )
        return BatchSpec(axes=axes, inference="explicit", user_batch_size=user_batch)

    axes = infer_auto_axes_from_example(inputs, input_kwargs, adapter=adapter)
    if not axes:
        # Scalars/vectors have no safely inferable batch axis. Keep ordinary
        # unbatched calls (e.g. Linear(4, 3) on (4,)) usable without resizing.
        if all(
            leaf.path.count("/") == 2
            and (leaf_shape := adapter.shape(leaf.value)) is not None
            and len(leaf_shape) < 2
            for leaf in leaves
        ):
            return BatchSpec(axes={}, inference="auto", user_batch_size=None)
        if leaves:
            available = {
                leaf.path: adapter.shape(leaf.value)
                for leaf in leaves
                if adapter.shape(leaf.value) is not None
            }
            if available:
                raise SplitUnsupportedError(
                    "Dynamic batch could not be inferred conservatively. Declare "
                    f"SplitFeatures.batch_axes using one of {available!r}, or use "
                    "batch_axes={} to declare that all inputs are unbatched."
                )
        return BatchSpec(axes={}, inference="auto", user_batch_size=None)
    inferred_batch = infer_user_batch_size(
        inputs,
        input_kwargs,
        adapter=adapter,
        explicit_axes=axes,
    )
    return BatchSpec(axes=axes, inference="auto", user_batch_size=inferred_batch)


def canonical_batch_for(spec: BatchSpec) -> int:
    """Return the canonical capture batch for resolved batch semantics."""

    if not spec.axes:
        return spec.user_batch_size or CANONICAL_TRACE_BATCH
    return CANONICAL_TRACE_BATCH


def witness_probe_sizes(traced_batch_size: int) -> tuple[int, ...]:
    """Return B=2 only for a B=1 capture; fallback captures do not extrapolate."""

    if traced_batch_size == CANONICAL_TRACE_BATCH:
        return _WITNESS_PROBES_FROM_ONE
    return ()


__all__ = [
    "CANONICAL_TRACE_BATCH",
    "FALLBACK_TRACE_BATCH",
    "BatchSpec",
    "canonical_batch_for",
    "resolve_batch_spec",
    "infer_auto_axes_from_example",
    "infer_explicit_axes_from_example",
    "infer_user_batch_size",
    "rebatch_inputs",
    "rebatch_tree",
    "witness_probe_sizes",
]
