"""TraceCore: per-trace ownership and cross-domain coordination.

TraceCore supplies lifetime, row-id domains, transactions, and cross-domain
coordination ONLY — each domain (columns per kind, pools, edges, groups,
payloads, overlays) owns its typed table and narrow API, so the core never
becomes a generic dict-driven god store.

Fork is core-level COW: the fork shares the frozen base (columns, pools,
edges, payload registrations) and receives its OWN overlay and facade
cache; base immutability makes sharing safe by construction.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Iterator

from .columns import ColumnBuilder
from .overlays import MISSING, RowOverlay
from .payloads import PayloadArena
from .groups import MembershipGroups
from .pools import ClosurePool, InternPool
from .relations import EdgeTable


class KindTable:
    """Typed column block for one record kind."""

    __slots__ = ("kind", "_columns", "_n_rows", "_frozen")

    def __init__(self, kind: str) -> None:
        """Create an empty table for ``kind``."""

        self.kind = kind
        self._columns: dict[str, ColumnBuilder] = {}
        self._n_rows = 0
        self._frozen = False

    def __len__(self) -> int:
        """Return the row count."""

        return self._n_rows

    @property
    def frozen(self) -> bool:
        """Return whether the table is frozen."""

        return self._frozen

    def column(self, field: str, codec: str = "object") -> ColumnBuilder:
        """Return (creating on first use) the column for ``field``."""

        column = self._columns.get(field)
        if column is None:
            if self._frozen:
                raise RuntimeError(f"{self.kind} table is frozen")
            column = ColumnBuilder(codec)
            for _ in range(self._n_rows):
                column.append_missing()
            self._columns[field] = column
        return column

    def columns(self) -> Iterator[tuple[str, ColumnBuilder]]:
        """Iterate (field, column) pairs."""

        return iter(self._columns.items())

    def new_row(self) -> int:
        """Append one row across every column and return its row id."""

        if self._frozen:
            raise RuntimeError(f"{self.kind} table is frozen")
        row = self._n_rows
        self._n_rows += 1
        for column in self._columns.values():
            column.append_missing()
        return row

    def set(self, row: int, field: str, value: Any) -> None:
        """Write one cell during BUILDING."""

        self.column(field).set(row, value)

    def get(self, row: int, field: str) -> Any:
        """Read one cell from the base (no overlay)."""

        column = self._columns.get(field)
        return None if column is None else column.get(row)

    def freeze(self) -> None:
        """Freeze every column; further writes go to the overlay."""

        for column in self._columns.values():
            column.freeze()
        self._frozen = True


class TraceCore:
    """One per-trace semantic store."""

    __slots__ = (
        "tables",
        "ops",
        "pool",
        "closures",
        "edges",
        "groups",
        "payloads",
        "overlay",
        "_facades",
        "_facade_factory",
        "backward_epochs",
        "__weakref__",
    )

    def __init__(self) -> None:
        """Create an empty core."""

        self.tables: dict[str, KindTable] = {}
        # The Op row store (op_store.OpRowStore) once the M5 ingress binds it;
        # None until materialize step 0 creates it for a captured run.
        self.ops: Any = None
        self.pool = InternPool()
        self.closures = ClosurePool()
        self.edges: dict[str, EdgeTable] = {}
        self.groups: dict[str, MembershipGroups] = {}
        self.payloads = PayloadArena()
        self.overlay = RowOverlay()
        # Strong facade cache first (byte-identical lifetime parity with the
        # object graph); flips weak-valued at M11 per the design's
        # strong-then-weak sequencing once the lifetime oracle proves no
        # observable dependency on unreferenced-facade survival.
        self._facades: dict[tuple[str, int], Any] = {}
        self._facade_factory: Callable[[str, int], Any] | None = None
        self.backward_epochs: list[Any] = []

    def table(self, kind: str) -> KindTable:
        """Return (creating on first use) the table for ``kind``."""

        table = self.tables.get(kind)
        if table is None:
            table = KindTable(kind)
            self.tables[kind] = table
        return table

    def edge_table(self, family: str) -> EdgeTable:
        """Return (creating on first use) one edge family table."""

        edges = self.edges.get(family)
        if edges is None:
            edges = EdgeTable()
            self.edges[family] = edges
        return edges

    def read(self, kind: str, row: int, field: str) -> Any:
        """Overlay-aware cell read."""

        overlaid = self.overlay.read(row, f"{kind}.{field}", MISSING)
        if overlaid is not MISSING:
            return overlaid
        return self.table(kind).get(row, field)

    def write(self, kind: str, row: int, field: str, value: Any) -> None:
        """Public write: base while building, overlay after freeze."""

        table = self.table(kind)
        if table.frozen:
            self.overlay.write(row, f"{kind}.{field}", value)
        else:
            table.set(row, field, value)

    def set_facade_factory(self, factory: Callable[[str, int], Any]) -> None:
        """Install the facade constructor used by ``facade()``."""

        self._facade_factory = factory

    def facade(self, kind: str, row: int) -> Any:
        """Return the identity-cached facade for one row."""

        key = (kind, row)
        cached = self._facades.get(key)
        if cached is None:
            if self._facade_factory is None:
                raise RuntimeError("no facade factory installed")
            cached = self._facade_factory(kind, row)
            self._facades[key] = cached
        return cached

    def freeze(self) -> None:
        """Freeze every kind table and edge family."""

        for table in self.tables.values():
            table.freeze()
        for family, edges in self.edges.items():
            del family
            if len(edges):
                n_rows = max(len(table) for table in self.tables.values())
                edges.freeze(n_rows, n_rows)

    def fork(self) -> "TraceCore":
        """COW fork: share the frozen base, detach mutation state."""

        child = TraceCore.__new__(TraceCore)
        child.tables = self.tables
        child.ops = self.ops
        child.pool = self.pool
        child.closures = self.closures
        child.edges = self.edges
        child.payloads = self.payloads
        child.overlay = RowOverlay()
        for key, value in self.overlay.snapshot().items():
            child.overlay.write(key[0], key[1], value)
        child._facades = {}
        child._facade_factory = self._facade_factory
        child.backward_epochs = []
        return child

    def weak_self(self) -> "weakref.ref[TraceCore]":
        """Return a weak reference to this core (facade back-pointer)."""

        return weakref.ref(self)
