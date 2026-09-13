"""Isolate model-owned tinygrad state during split's internal forwards."""

from __future__ import annotations

import dis
import types
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any


@contextmanager
def tinygrad_capture_state(model: Any) -> Iterator[None]:
    """Restore reachable model attributes and isolate mutable tensor storage.

    Parameters
    ----------
    model
        Callable object, bound method, or Python function whose referenced
        globals, closure cells, and defaults may own tinygrad model state.

    Yields
    ------
    None
        An internal capture scope with private, alias-preserving state buffers.

    Notes
    -----
    Copying a Python function returns the same function and does not isolate
    its model. In particular, a canonical B=1 capture must not leave a newly
    initialized batch-shaped cache for the subsequent B=2 witness. Existing
    cache attributes are retained, never guessed or reset by name.
    """

    from tinygrad import Tensor
    from tinygrad.uop.ops import Ops, UOp

    seen: set[int] = set()
    objects: list[tuple[Any, dict[str, Any]]] = []
    lists: list[tuple[list[Any], list[Any]]] = []
    mappings: list[tuple[dict[Any, Any], dict[Any, Any]]] = []
    sets: list[tuple[set[Any], set[Any]]] = []
    tensors: list[tuple[Any, Any, Any, Any]] = []
    missing = object()
    global_slots: dict[tuple[int, str], tuple[dict[str, Any], str, Any]] = {}
    cells: dict[int, tuple[types.CellType, Any]] = {}

    def visit(value: Any) -> None:
        """Record only state reachable from the supplied model callable."""

        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Tensor):
            tensors.append((value, value.uop, value.grad, value.is_param))
            visit(value.grad)
        elif isinstance(value, types.MethodType):
            visit(value.__self__)
            visit(value.__func__)
        elif isinstance(value, types.FunctionType):
            for instruction in dis.get_instructions(value):
                if instruction.opname not in {"LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL"}:
                    continue
                name = instruction.argval
                slot = (id(value.__globals__), name)
                if slot not in global_slots:
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
            snapshot = dict(value)
            mappings.append((value, snapshot))
            for item in snapshot.values():
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
                saved_attrs = {name: item for name, item in attrs.items() if name != "_tl"}
                objects.append((value, saved_attrs))
                for item in saved_attrs.values():
                    visit(item)

    visit(model)
    try:
        # Raw STORE effects can write a Buffer without changing its Tensor's
        # UOp. Work on private storage instead of attempting to undo writes;
        # the retained capture keeps its private buffers after this scope ends.
        replacements: dict[Any, Any] = {}
        private_buffers: dict[int, Any] = {}
        for _tensor, uop, _grad, _is_param in tensors:
            for node in uop.toposort():
                if node.op is not Ops.BUFFER or node in replacements:
                    continue
                buffer = node.buffer
                if id(buffer) not in private_buffers:
                    if buffer.is_allocated():
                        private_buffers[id(buffer)] = Tensor(node).clone().realize().uop
                    else:
                        private_buffers[id(buffer)] = UOp.new_buffer(
                            node.device, node.size, node.dtype
                        )
                replacements[node] = private_buffers[id(buffer)]
        for tensor, uop, _grad, _is_param in tensors:
            if replacements:
                tensor.uop = uop.substitute(replacements)
        yield
    finally:
        for tensor, uop, grad, is_param in tensors:
            tensor.uop, tensor.grad, tensor.is_param = uop, grad, is_param
        for value, saved_attrs in reversed(objects):
            attrs = vars(value)
            for name in tuple(attrs):
                if name != "_tl" and name not in saved_attrs:
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
