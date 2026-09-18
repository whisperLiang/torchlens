"""Shared reachable Python-state snapshots for native split probe transactions."""

from __future__ import annotations

import dis
import types
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any


class PythonStateSnapshot:
    """Snapshot reachable containers and bindings without copying callable identities."""

    def __init__(
        self,
        visit_tensor: Callable[[Any, Callable[[Any], None]], bool],
        *,
        preserve_tl: bool = False,
    ) -> None:
        """Configure backend-owned tensor handling and optional live metadata retention."""

        self.visit_tensor = visit_tensor
        self.preserve_tl = preserve_tl
        self.seen: set[int] = set()
        self.objects: list[tuple[Any, dict[str, Any]]] = []
        self.lists: list[tuple[list[Any], list[Any]]] = []
        self.mappings: list[tuple[dict[Any, Any], dict[Any, Any]]] = []
        self.sets: list[tuple[set[Any], set[Any]]] = []
        self.global_slots: dict[tuple[int, str], tuple[dict[str, Any], str, Any]] = {}
        self.cells: dict[int, tuple[types.CellType, Any]] = {}
        self.missing = object()

    def visit(self, value: Any) -> None:
        """Visit each identity once, delegating native tensor state to its backend."""

        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        # Pass the visitor rather than closing over this snapshot in the
        # backend callback: that would retain tensor snapshots in a GC cycle.
        if self.visit_tensor(value, self.visit):
            return
        if isinstance(value, types.MethodType):
            self.visit(value.__self__)
            self.visit(value.__func__)
        elif isinstance(value, types.FunctionType):
            self._visit_function(value)
        elif isinstance(value, partial):
            self.visit(value.func)
            self.visit(value.args)
            self.visit(value.keywords)
        elif isinstance(value, (dict, list, set, tuple, frozenset)):
            self._visit_container(value)
        elif not isinstance(value, (type, types.ModuleType)):
            self._visit_object(value)

    def _visit_function(self, value: types.FunctionType) -> None:
        """Record referenced globals, closure cells, and argument defaults."""

        for instruction in dis.get_instructions(value):
            if instruction.opname not in {"LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL"}:
                continue
            name = str(instruction.argval)
            slot = (id(value.__globals__), name)
            if slot in self.global_slots:
                continue
            contents = value.__globals__.get(name, self.missing)
            self.global_slots[slot] = (value.__globals__, name, contents)
            if contents is not self.missing:
                self.visit(contents)
        self._visit_cells(value)
        self.visit(value.__defaults__)
        self.visit(value.__kwdefaults__)

    def _visit_cells(self, value: types.FunctionType) -> None:
        """Record each closure binding once, including empty cells."""

        for cell in value.__closure__ or ():
            if id(cell) in self.cells:
                continue
            try:
                contents = cell.cell_contents
            except ValueError:
                contents = self.missing
            self.cells[id(cell)] = (cell, contents)
            if contents is not self.missing:
                self.visit(contents)

    def _visit_container(self, value: Any) -> None:
        """Record mutable membership before traversing the saved children."""

        children: Iterable[Any]
        if isinstance(value, dict):
            saved_mapping = dict(value)
            self.mappings.append((value, saved_mapping))
            children = saved_mapping.values()
        elif isinstance(value, list):
            saved_items = list(value)
            self.lists.append((value, saved_items))
            children = saved_items
        elif isinstance(value, set):
            saved_members = set(value)
            self.sets.append((value, saved_members))
            children = saved_members
        else:
            children = value
        for item in children:
            self.visit(item)

    def _visit_object(self, value: Any) -> None:
        """Snapshot plain attributes, preserving tinygrad's live capture metadata."""

        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            saved = {
                name: item
                for name, item in attrs.items()
                if not (self.preserve_tl and name == "_tl")
            }
            self.objects.append((value, saved))
            for item in saved.values():
                self.visit(item)

    def restore(self) -> None:
        """Restore Python state after the backend has restored its tensor state."""

        self._restore_objects()
        for mapping, saved_mapping in self.mappings:
            mapping.clear()
            mapping.update(saved_mapping)
        for items, saved_items in self.lists:
            items[:] = saved_items
        for members, saved_members in self.sets:
            members.clear()
            members.update(saved_members)
        self._restore_bindings()

    def _restore_objects(self) -> None:
        """Restore attributes in reverse discovery order without changing identities."""

        for value, saved_attrs in reversed(self.objects):
            attrs = vars(value)
            for name in tuple(attrs):
                if name not in saved_attrs and not (self.preserve_tl and name == "_tl"):
                    del attrs[name]
            attrs.update(saved_attrs)

    def _restore_bindings(self) -> None:
        """Restore or delete global and nonlocal bindings according to capture truth."""

        for namespace, name, contents in self.global_slots.values():
            if contents is self.missing:
                namespace.pop(name, None)
            else:
                namespace[name] = contents
        for cell, contents in self.cells.values():
            if contents is self.missing:
                del cell.cell_contents
            else:
                cell.cell_contents = contents
