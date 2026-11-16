"""Builtin output-container metadata shared by eager preview backends."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal

from .._ir.container import ContainerSpec, DictKey, TupleIndex
from ._finalize import mark_output_label

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def output_container_spec(value: Any, is_tensor: Callable[[Any], bool]) -> ContainerSpec | None:
    """Describe builtin output slots with the canonical upstream container schema.

    Parameters
    ----------
    value
        Native model output or one of its children.
    is_tensor
        Backend-owned tensor predicate, without importing another runtime.

    Returns
    -------
    ContainerSpec | None
        The container description, or no child specification for a tensor.
    """

    if is_tensor(value):
        return None
    if type(value) in (list, tuple):
        kind: Literal["list", "tuple", "dict"] = "list" if type(value) is list else "tuple"
        items = tuple(enumerate(value))
    elif type(value) is dict:
        kind = "dict"
        items = tuple(value.items())
    else:
        return ContainerSpec(kind="literal", literal_value=value)
    return ContainerSpec(
        kind=kind,
        length=len(value) if kind != "dict" else None,
        keys=tuple(value) if kind == "dict" else (),
        child_specs=tuple(
            (DictKey(key) if kind == "dict" else TupleIndex(key), child)
            for key, item in items
            if (child := output_container_spec(item, is_tensor)) is not None
        ),
    )


def _output_leaves(
    value: Any, is_tensor: Callable[[Any], bool], path: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Visit native tensor occurrences in builtin container reconstruction order."""

    if is_tensor(value):
        yield path, value
    elif type(value) in (tuple, list):
        for index, child in enumerate(value):
            yield from _output_leaves(child, is_tensor, (*path, index))
    elif type(value) is dict:
        for key, child in value.items():
            yield from _output_leaves(child, is_tensor, (*path, key))


def mark_preview_outputs(
    trace: Trace,
    output: Any,
    is_tensor: Callable[[Any], bool],
    label_of: Callable[[Any], str | None],
) -> tuple[tuple[Any, ...], ...]:
    """Mark producing events and retain every final output occurrence.

    Parameters
    ----------
    trace
        Live capture owning the canonical event journal.
    output
        Native model output, including repeated tensor objects.
    is_tensor
        Backend-owned leaf predicate.
    label_of
        Backend-owned lookup of a tensor's recorded producer.

    Returns
    -------
    tuple
        Paths parallel to the occurrence-preserving ``trace.output_layers``.
        Missing producers stay missing for the backend's coverage tripwire.
    """

    specification = output_container_spec(output, is_tensor)
    paths: list[tuple[Any, ...]] = []
    marked: set[str] = set()
    for path, tensor in _output_leaves(output, is_tensor):
        label = label_of(tensor)
        if label is None:
            continue
        paths.append(path)
        if label in marked:
            trace.output_layers.append(label)
            continue
        marked.add(label)
        mark_output_label(trace, label, container_path=path, container_spec=specification)
    return tuple(paths)
