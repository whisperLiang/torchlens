"""Transactional state snapshots for function-based native split probes."""

from __future__ import annotations

import dis
import types
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any


def _restore_tensor(value: Any, snapshot: Any) -> None:
    """Restore a mutable framework tensor when its backend exposes a mutator."""

    assign = getattr(value, "assign", None)
    if callable(assign):
        assign(snapshot)
        return
    set_value = getattr(value, "set_value", None)
    if callable(set_value):
        set_value(snapshot)
        return
    data = getattr(value, "data", None)
    copy_ = getattr(data, "copy_", None)
    if callable(copy_):
        copy_(snapshot)


def _tensor_snapshot(value: Any, adapter: Any) -> Any | None:
    """Copy a mutable backend tensor, or return ``None`` for immutable values."""

    if not adapter.is_tensor(value):
        return None
    if not any(
        callable(getattr(value, name, None)) for name in ("assign", "set_value")
    ) and not callable(getattr(getattr(value, "data", None), "copy_", None)):
        return None
    try:
        return adapter.clone(adapter.detach(value))
    except Exception:
        return None


@contextmanager
def callable_capture_state(model: Any, adapter: Any) -> Iterator[None]:
    """Restore mutable state reachable from a Python function after a probe.

    Parameters
    ----------
    model:
        Function, bound method, or callable state reachable from the function.
    adapter:
        Native split adapter used to identify mutable framework variables.

    Yields
    ------
    None
        A transactional scope for an internal native forward.

    Notes
    -----
    ``deepcopy(function)`` returns the original function, so its globals and
    closure cells remain shared. This scope snapshots those reachable values
    without copying imported modules or executing arbitrary user code.
    """

    seen: set[int] = set()
    objects: list[tuple[Any, dict[str, Any]]] = []
    lists: list[tuple[list[Any], list[Any]]] = []
    mappings: list[tuple[dict[Any, Any], dict[Any, Any]]] = []
    sets: list[tuple[set[Any], set[Any]]] = []
    tensors: list[tuple[Any, Any]] = []
    global_slots: dict[tuple[int, str], tuple[dict[str, Any], str, Any]] = {}
    cells: dict[int, tuple[types.CellType, Any]] = {}
    missing = object()

    def visit(value: Any) -> None:
        """Record one reachable value and recurse through supported state."""

        if id(value) in seen:
            return
        seen.add(id(value))
        snapshot = _tensor_snapshot(value, adapter)
        if snapshot is not None:
            tensors.append((value, snapshot))
            return
        if isinstance(value, types.MethodType):
            visit(value.__self__)
            visit(value.__func__)
        elif isinstance(value, types.FunctionType):
            for instruction in dis.get_instructions(value):
                if instruction.opname not in {"LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL"}:
                    continue
                name = str(instruction.argval)
                slot = (id(value.__globals__), name)
                if slot in global_slots:
                    continue
                contents = value.__globals__.get(name, missing)
                global_slots[slot] = (value.__globals__, name, contents)
                if contents is not missing:
                    visit(contents)
            for cell in value.__closure__ or ():
                if id(cell) in cells:
                    continue
                try:
                    contents = cell.cell_contents
                except ValueError:
                    contents = missing
                cells[id(cell)] = (cell, contents)
                if contents is not missing:
                    visit(contents)
            visit(value.__defaults__)
            visit(value.__kwdefaults__)
        elif isinstance(value, partial):
            visit(value.func)
            visit(value.args)
            visit(value.keywords)
        elif isinstance(value, dict):
            saved_mapping = dict(value)
            mappings.append((value, saved_mapping))
            for item in saved_mapping.values():
                visit(item)
        elif isinstance(value, list):
            saved_items = list(value)
            lists.append((value, saved_items))
            for item in saved_items:
                visit(item)
        elif isinstance(value, set):
            saved_members = set(value)
            sets.append((value, saved_members))
            for item in saved_members:
                visit(item)
        elif isinstance(value, (tuple, frozenset)):
            for item in value:
                visit(item)
        elif not isinstance(value, (type, types.ModuleType)):
            attrs = getattr(value, "__dict__", None)
            if isinstance(attrs, dict):
                saved = dict(attrs)
                objects.append((value, saved))
                for item in saved.values():
                    visit(item)

    visit(model)
    try:
        yield
    finally:
        for value, snapshot in tensors:
            _restore_tensor(value, snapshot)
        for value, saved_attrs in reversed(objects):
            attrs = vars(value)
            for name in tuple(attrs):
                if name not in saved_attrs:
                    del attrs[name]
            attrs.update(saved_attrs)
        for mapping, saved_mapping in mappings:
            mapping.clear()
            mapping.update(saved_mapping)
        for items, saved_items in lists:
            items[:] = saved_items
        for members, saved_members in sets:
            members.clear()
            members.update(saved_members)
        for namespace, name, contents in global_slots.values():
            if contents is missing:
                namespace.pop(name, None)
            else:
                namespace[name] = contents
        for cell, contents in cells.values():
            if contents is missing:
                del cell.cell_contents
            else:
                cell.cell_contents = contents


__all__ = ["callable_capture_state"]
