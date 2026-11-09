"""Describe Paddle model output containers without importing the Paddle runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...ir.container import ContainerSpec, DictKey, TupleIndex


def _paddle_output_container_spec(
    value: Any, is_tensor: Callable[[Any], bool]
) -> ContainerSpec | None:
    """Describe builtin final-output containers independently of native call outputs.

    Parameters
    ----------
    value:
        Model output or one of its nested children.
    is_tensor:
        Backend predicate identifying tensor leaves.

    Returns
    -------
    ContainerSpec | None
        Nested structure, or no child specification for a tensor leaf.
    """

    if is_tensor(value):
        return None
    if type(value) in (list, tuple):
        sequence_children = tuple(
            (TupleIndex(index), child)
            for index, item in enumerate(value)
            if (child := _paddle_output_container_spec(item, is_tensor)) is not None
        )
        return ContainerSpec(
            "list" if isinstance(value, list) else "tuple",
            length=len(value),
            child_specs=sequence_children,
        )
    if type(value) is dict:
        dict_children = tuple(
            (DictKey(key), child)
            for key, item in value.items()
            if (child := _paddle_output_container_spec(item, is_tensor)) is not None
        )
        return ContainerSpec("dict", keys=tuple(value), child_specs=dict_children)
    return ContainerSpec("literal", literal_value=value)
