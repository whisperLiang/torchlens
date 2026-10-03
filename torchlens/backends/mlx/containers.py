"""Container traversal helpers for the MLX capture backend."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import replace
from typing import Any, Literal

from ...ir.container import ContainerSpec, DictKey, TupleIndex
from ...ir.op_record import amend_preview_output_parent_rebind


def iter_arrays_with_paths(
    value: object,
    is_tensor: Callable[[object], bool],
    path: tuple[object, ...] = (),
    *,
    sort_dict: bool = False,
) -> list[tuple[object, tuple[object, ...]]]:
    """Return MLX array leaves paired with their container paths.

    Parameters
    ----------
    value
        Candidate value or builtin container.
    is_tensor
        Backend-owned MLX array predicate.
    path
        Path accumulated by the recursive traversal.
    sort_dict
        Sort dictionary keys when matching input flattening order.
    """

    if is_tensor(value):
        return [(value, path)]
    if isinstance(value, (list, tuple)):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in iter_arrays_with_paths(item, is_tensor, (*path, index), sort_dict=sort_dict)
        ]
    if isinstance(value, dict):
        items = (
            value.items()
            if not sort_dict
            else sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return [
            leaf
            for key, item in items
            for leaf in iter_arrays_with_paths(item, is_tensor, (*path, key), sort_dict=sort_dict)
        ]
    return []


def output_container_spec(
    value: object, is_tensor: Callable[[object], bool]
) -> ContainerSpec | None:
    """Build a portable spec for builtin MLX output containers.

    Parameters
    ----------
    value
        Captured operation or model output.
    is_tensor
        Backend-owned MLX array predicate.
    """

    if is_tensor(value):
        return None
    if isinstance(value, tuple) and type(value) is tuple:
        kind: Literal["tuple", "list", "dict"] = "tuple"
        items = tuple(enumerate(value))
    elif isinstance(value, list):
        kind = "list"
        items = tuple(enumerate(value))
    elif isinstance(value, dict) and type(value) is dict:
        kind = "dict"
        items = tuple(value.items())
    else:
        return ContainerSpec(kind="literal", literal_value=value)
    child_specs = tuple(
        (DictKey(key) if kind == "dict" else TupleIndex(key), child_spec)
        for key, item in items
        if (child_spec := output_container_spec(item, is_tensor)) is not None
    )
    return ContainerSpec(
        kind=kind,
        length=len(value) if kind != "dict" else None,
        keys=tuple(key for key, _item in items) if kind == "dict" else (),
        child_specs=child_specs,
    )


def mark_output_occurrences(
    backend: Any, trace: Any, output: object
) -> tuple[tuple[object, ...], ...]:
    """Mark output producers once while retaining every final container occurrence.

    Parameters
    ----------
    backend
        MLX backend owning the array label store.
    trace
        Capture whose output producer inventory is being populated.
    output
        Native model output, including repeated array objects.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Paths aligned with the occurrence-preserving ``trace.output_layers``.
    """

    spec = output_container_spec(output, backend.is_tensor)
    paths: list[tuple[object, ...]] = []
    marked: set[str] = set()
    for value, path in iter_arrays_with_paths(output, backend.is_tensor):
        label = backend.tensor_store.get_label(value)
        if label is None:
            continue
        trace.output_layers.append(label)
        paths.append(path)
        if label in marked:
            continue
        marked.add(label)
        event = trace.capture_events.op_event_by_label_raw.get(label)
        if event is not None:
            trace.capture_events.append_amendment(
                amend_preview_output_parent_rebind(
                    event.seq,
                    label,
                    is_output_parent=True,
                    output=replace(
                        event.output,
                        container_path=path,
                        in_multi_output=bool(path),
                        container_spec=spec,
                    ),
                )
            )
    return tuple(paths)


def rebuild_mlx_module(module: Any, template: dict[str, Any]) -> Any:
    """Restore native module types without writing values into the captured model.

    Parameters
    ----------
    module
        Native module retained by the live call sidecar.
    template
        Captured module contents with replay array leaves substituted.
    """

    import mlx.nn as nn

    def rebuild(original: Any, value: Any) -> Any:
        """Restore module classes while retaining the captured container contents."""

        if isinstance(original, nn.Module) and isinstance(value, dict):
            restored = copy(original)
            if hasattr(original, "_no_grad"):
                restored._no_grad = set(original._no_grad)
            restored.clear()
            dict.update(
                restored, {key: rebuild(original.get(key), item) for key, item in value.items()}
            )
            return restored
        if isinstance(value, dict):
            return {
                key: rebuild(original.get(key) if isinstance(original, dict) else None, item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return type(value)(
                rebuild(
                    original[index]
                    if isinstance(original, (tuple, list)) and index < len(original)
                    else None,
                    item,
                )
                for index, item in enumerate(value)
            )
        return value

    return rebuild(module, template)


__all__ = [
    "iter_arrays_with_paths",
    "mark_output_occurrences",
    "output_container_spec",
    "rebuild_mlx_module",
]
