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


def _safe_tinygrad_buffer_ancestor(tensor: Any, ops: Any) -> tuple[int, Any] | None:
    """Find a BUFFER only through value-preserving tensor operations.

    Parameters
    ----------
    tensor
        Live model tensor that may be backed by one BUFFER.
    ops
        The installed tinygrad operation enum.

    Returns
    -------
    tuple[int, Any] | None
        Binding priority and BUFFER UOp, or None when values may have changed.
    """

    uop = getattr(tensor, "uop", None)
    rank = 0
    while uop is not None:
        if uop.op is ops.BUFFER:
            return rank, uop
        if uop.op not in {ops.COPY, ops.CONTIGUOUS, ops.RESHAPE}:
            return None
        src = tuple(getattr(uop, "src", ()) or ())
        if not src:
            return None
        uop = src[0]
        rank = 1
    return None


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
        Unambiguous, value-preserving BUFFER identity to live Tensor bindings.
    """

    try:
        from tinygrad.uop.ops import Ops
    except ImportError:
        return {}

    candidates: dict[int, list[tuple[int, Any]]] = {}
    for tensor in _tinygrad_parameter_tensors(tensors_by_uop, module_tree):
        ancestor = _safe_tinygrad_buffer_ancestor(tensor, Ops)
        if ancestor is not None:
            rank, uop = ancestor
            candidates.setdefault(id(uop), []).append((rank, tensor))
    for tensor in tensors_by_uop.values():
        uop = getattr(tensor, "uop", None)
        if uop is not None and uop.op is Ops.BUFFER:
            candidates.setdefault(id(uop), []).append((0, tensor))
    bindings: dict[int, Any] = {}
    for key, matches in candidates.items():
        best_rank = min(rank for rank, _tensor in matches)
        best = {id(tensor): tensor for rank, tensor in matches if rank == best_rank}
        if len(best) == 1:
            bindings[key] = next(iter(best.values()))
    return bindings


__all__ = ["live_tinygrad_buffer_tensors_by_uop", "live_tinygrad_tensors_by_uop"]
