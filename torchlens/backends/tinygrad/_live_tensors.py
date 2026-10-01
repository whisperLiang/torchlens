"""Identity-guarded live Tensor discovery for the tinygrad backend."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from ._module_discovery import _iter_tinygrad_tensor_attrs


def _tinygrad_callable_references(value: Any) -> list[Any]:
    """Collect the narrow closure, referenced globals, and object attributes.

    Parameters
    ----------
    value
        Callable or model object to inspect without exploring all globals.

    Returns
    -------
    list[Any]
        Reachable candidates for the live tensor visitor.
    """

    references: list[Any] = []
    for cell in getattr(value, "__closure__", None) or ():
        try:
            references.append(cell.cell_contents)
        except ValueError:
            continue
    code = getattr(value, "__code__", None)
    globals_dict = getattr(value, "__globals__", {})
    references.extend(
        globals_dict[name] for name in getattr(code, "co_names", ()) if name in globals_dict
    )
    references.extend(getattr(value, "__defaults__", None) or ())
    references.extend(getattr(value, "__dict__", {}).values())
    return references


def live_tinygrad_tensors_by_uop(model: Any) -> dict[int, Any]:
    """Collect identity-guarded Tensor handles reachable from a tinygrad callable.

    Parameters
    ----------
    model
        Captured callable, including its closed-over and model-owned tensors.

    Returns
    -------
    dict[int, Any]
        Live handles keyed by the exact pre-capture UOp identity.
    """

    try:
        from tinygrad import Tensor
    except ImportError:
        return {}

    found: dict[int, Any] = {}
    visited: set[int] = set()

    def visit(value: Any) -> None:
        """Visit a value once; a Tensor terminates recursive discovery."""

        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        if isinstance(value, Tensor):
            uop = getattr(value, "uop", None)
            if uop is not None:
                found.setdefault(id(uop), value)
            return
        if inspect.ismodule(value) or inspect.isclass(value):
            return
        children: Iterable[Any]
        if isinstance(value, Mapping):
            children = value.values()
        elif isinstance(value, (tuple, list, set, frozenset)):
            children = value
        else:
            children = _tinygrad_callable_references(value)
        for child in children:
            visit(child)

    visit(model)
    return found


def _tinygrad_parameter_tensors(
    tensors_by_uop: Mapping[int, Any], module_tree: Any | None
) -> list[Any]:
    """Choose named model tensors or trainable functional tensor sources.

    Parameters
    ----------
    tensors_by_uop
        Live Tensor handles found in the captured callable.
    module_tree
        Object-module discovery result, when the model exposes one.

    Returns
    -------
    list[Any]
        Model state handles whose BUFFER ancestors can be bound safely.
    """

    if module_tree is None:
        return [
            tensor
            for tensor in tensors_by_uop.values()
            if bool(getattr(tensor, "requires_grad", False))
        ]
    return [
        tensor
        for address, metadata in module_tree.metadata.items()
        if (module := metadata.get("_module_object")) is not None
        for _param_address, tensor in _iter_tinygrad_tensor_attrs(module, address)
    ]


def live_tinygrad_buffer_tensors_by_uop(
    tensors_by_uop: Mapping[int, Any], module_tree: Any | None
) -> dict[int, Any]:
    """Bind BUFFER ancestors of live parameters once at capture time.

    Parameters
    ----------
    tensors_by_uop
        Reachable live Tensor handles keyed by their UOp identity.
    module_tree
        Discovered object-module parameters, when present.

    Returns
    -------
    dict[int, Any]
        Unambiguous BUFFER identity to live Tensor bindings.
    """

    try:
        from tinygrad.uop.ops import Ops
    except ImportError:
        return {}

    bindings: dict[int, Any] = {}
    ambiguous: set[int] = set()
    for tensor in _tinygrad_parameter_tensors(tensors_by_uop, module_tree):
        try:
            lineage = tensor.uop.toposort()
        except Exception:
            continue
        for uop in lineage:
            if uop.op is not Ops.BUFFER:
                continue
            key = id(uop)
            existing = bindings.get(key)
            if existing is None:
                bindings[key] = tensor
            elif existing is not tensor:
                ambiguous.add(key)
    for tensor in tensors_by_uop.values():
        uop = getattr(tensor, "uop", None)
        if uop is not None and uop.op is Ops.BUFFER:
            bindings.setdefault(id(uop), tensor)
    for key in ambiguous:
        bindings.pop(key, None)
    return bindings


__all__ = ["live_tinygrad_buffer_tensors_by_uop", "live_tinygrad_tensors_by_uop"]
