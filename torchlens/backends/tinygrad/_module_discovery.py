"""Bounded object-module and parameter discovery through tinygrad model containers."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from .._finalize import join_module_address as _join_module_address
from ..registry import BackendUnsupportedError


def _iter_tinygrad_model_leaves(module: Any, address: str) -> list[tuple[str, Any]]:
    """Walk only public module attributes and their containers, guarding cycles.

    Parameters
    ----------
    module
        Root object for one module's attributes.
    address
        Address prefix for discovered leaves.

    Returns
    -------
    list[tuple[str, Any]]
        Addressed tensor and module candidates.
    """

    leaves: list[tuple[str, Any]] = []
    for name, value in getattr(module, "__dict__", {}).items():
        if not name.startswith("_"):
            _visit_tinygrad_model_value(value, _join_module_address(address, name), set(), leaves)
    return leaves


def _visit_tinygrad_model_value(
    value: Any, path: str, active: set[int], leaves: list[tuple[str, Any]]
) -> None:
    """Descend through containers, guarding cycles and leaving modules as leaves.

    Parameters
    ----------
    value
        Child value to inspect.
    path
        Structural address of the child.
    active
        Container identities on the current traversal branch.
    leaves
        Destination for addressed non-container values.
    """

    if inspect.ismodule(value) or inspect.isclass(value):
        return
    children = _tinygrad_container_children(value, path)
    if children is None:
        leaves.append((path, value))
        return
    if id(value) in active:
        return
    active.add(id(value))
    for component, child in children:
        _visit_tinygrad_model_value(child, f"{path}.{component}", active, leaves)
    active.remove(id(value))


def _tinygrad_container_children(value: Any, path: str) -> list[tuple[str, Any]] | None:
    """Return safe named children for one supported container.

    Parameters
    ----------
    value
        Candidate mapping, sequence, or set.
    path
        Structural address used for an ambiguous-order refusal.

    Returns
    -------
    list[tuple[str, Any]] | None
        Named children, or None when this is a leaf.
    """

    if isinstance(value, Mapping):
        children: list[tuple[str, Any]] = []
        used: set[str] = set()
        # Ordinary mappings preserve their author's insertion order. Sorting
        # custom keys by repr/id instead made equivalent dicts change paths.
        for index, key in enumerate(value):
            component = _tinygrad_mapping_component(key, index)
            while component in used:
                component = f"{component}_{index}"
            used.add(component)
            children.append((component, value[key]))
        return children
    if isinstance(value, (set, frozenset)):
        # A model may keep unrelated unordered metadata beside its modules.
        # Only members that contribute module or Tensor addresses need an
        # ordering contract; otherwise an ordinary set can block capture.
        items = sorted(
            (item for item in value if _has_tinygrad_model_leaf(item, set())),
            key=_tinygrad_container_sort_key,
        )
        keys = [_tinygrad_container_sort_key(item) for item in items]
        if len(keys) != len(set(keys)):
            raise BackendUnsupportedError(
                f"tinygrad module container {path!r} cannot assign a stable address to "
                "indistinguishable set members; use a list/tuple or give members distinct names."
            )
        return [(str(index), item) for index, item in enumerate(items)]
    if isinstance(value, tuple):
        fields = getattr(type(value), "_fields", None)
        if (
            isinstance(fields, tuple)
            and len(fields) == len(value)
            and all(isinstance(field, str) and field.isidentifier() for field in fields)
            and len(set(fields)) == len(fields)
        ):
            return list(zip(fields, value, strict=True))
    if isinstance(value, (list, tuple)):
        return [(str(index), item) for index, item in enumerate(value)]
    return None


def _has_tinygrad_model_leaf(value: Any, active: set[int]) -> bool:
    """Check whether one container member needs a module or Tensor address.

    Parameters
    ----------
    value
        Unordered member to inspect without assigning an address.
    active
        Object and container identities on the current traversal branch.

    Returns
    -------
    bool
        Whether the member contains discoverable module or Tensor state.
    """

    if _is_tinygrad_tensor(value):
        return True
    if (
        inspect.ismodule(value)
        or inspect.isclass(value)
        or inspect.isfunction(value)
        or inspect.ismethod(value)
        or id(value) in active
    ):
        return False
    if callable(value) and _is_known_tinygrad_nn_type(value):
        return True
    active.add(id(value))
    try:
        children: Iterable[Any]
        if isinstance(value, Mapping):
            children = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = value
        elif callable(value):
            children = (
                child
                for name, child in getattr(value, "__dict__", {}).items()
                if not name.startswith("_")
            )
        else:
            return False
        return any(_has_tinygrad_model_leaf(child, active) for child in children)
    finally:
        active.remove(id(value))


def _tinygrad_container_sort_key(value: Any) -> tuple[str, str]:
    """Give unordered members a stable semantic key without object identity.

    Parameters
    ----------
    value
        Mapping key or set element to sort.

    Returns
    -------
    tuple[str, str]
        Type and a stable value or declared member name.
    """

    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if type(value) in (str, bytes, int, float, bool, complex, type(None)):
        return (kind, repr(value))
    if isinstance(value, tuple):
        return (kind, repr(tuple(_tinygrad_container_sort_key(item) for item in value)))
    name = getattr(value, "__dict__", {}).get("name")
    return (kind, name if isinstance(name, str) else "")


def _tinygrad_mapping_component(key: Any, index: int) -> str:
    """Use readable safe keys, and ordinal paths for other mapping keys.

    Parameters
    ----------
    key
        Mapping key to address.
    index
        Stable position in this mapping's sorted key order.

    Returns
    -------
    str
        Safe dotted-path component.
    """

    if isinstance(key, str) and key.isidentifier() and not key.startswith("_"):
        return key
    return f"key_{index}"


def _is_tinygrad_module_like(value: Any, seen: set[int] | None = None) -> bool:
    """Return whether ``value`` is a tinygrad module-like callable object.

    Parameters
    ----------
    value
        Candidate object.
    seen
        Module identities already inspected on this discovery path.

    Returns
    -------
    bool
        True when the object matches the tinygrad module discovery heuristic.
    """

    if (
        inspect.ismodule(value)
        or inspect.isclass(value)
        or inspect.isfunction(value)
        or inspect.ismethod(value)
        or _is_tinygrad_tensor(value)
        or not callable(value)
    ):
        return False
    visited = set() if seen is None else seen
    if id(value) in visited:
        return False
    visited.add(id(value))
    if _is_known_tinygrad_nn_type(value):
        return True
    for _path, child in _iter_tinygrad_model_leaves(value, "self"):
        if _is_tinygrad_tensor(child) or _is_tinygrad_module_like(child, visited):
            return True
    return False


def _is_known_tinygrad_nn_type(value: Any) -> bool:
    """Return whether ``value`` is an instance of a known ``tinygrad.nn`` class.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True for known tinygrad neural-network helper classes.
    """

    try:
        import tinygrad.nn as tinygrad_nn
    except ImportError:
        return False
    known_types = tuple(
        attr for name in dir(tinygrad_nn) if isinstance((attr := getattr(tinygrad_nn, name)), type)
    )
    return isinstance(value, known_types)


def _is_tinygrad_tensor(value: Any) -> bool:
    """Return whether ``value`` is a tinygrad Tensor.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when ``value`` is a tinygrad ``Tensor``.
    """

    try:
        from tinygrad import Tensor
    except ImportError:
        return False
    return isinstance(value, Tensor)


def _iter_tinygrad_tensor_attrs(module: Any, address: str) -> list[tuple[str, Any]]:
    """Return tinygrad tensor attributes through supported containers.

    Parameters
    ----------
    module
        Candidate module object.
    address
        TorchLens module address.

    Returns
    -------
    list[tuple[str, Any]]
        Parameter address and tensor pairs.
    """

    return [
        (path, value)
        for path, value in _iter_tinygrad_model_leaves(module, address)
        if _is_tinygrad_tensor(value)
    ]
