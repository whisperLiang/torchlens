"""Columnar row store backing the ``Op`` facade (the M5 seam).

One ``OpRowStore`` per captured trace holds every stored ``Op`` field cell;
``Op`` itself is a two-word ``(_core, _row)`` facade whose generated data
descriptors read and write cells here. ``DetachedOpStore`` is the single-row
twin used by ``Op.copy()``, pickle restoration, fork shells, and preview
backends that construct ops outside the event-materialize ingress.

Storage phases:

* BUILDING (postprocess steps 0-20): rows are plain Python lists (one
  ``[_MISSING] * n_fields`` allocation per op), so cell reads/writes are
  list indexing and the per-op footprint matches the former slot layout.
* FROZEN (after step 20, where the standalone compaction passes ran): rows
  transpose once into per-field columns. Numeric/bool columns pack into
  numpy arrays when every present value is an exact ``bool``/``int``/
  ``float`` (subclasses such as ``Bytes`` stay object-backed so reads keep
  their exact public type); everything else stays an object column. Later
  writes land in a sparse overlay; deletes on object columns release the
  cell value in place so op removal keeps freeing payloads.

An unset cell is ``_MISSING`` (never ``None`` -- ``None`` is a real stored
value); facade descriptors translate ``_MISSING`` into the exact
``AttributeError`` an unset ``__slots__`` member raised.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

#: Unset-cell sentinel; distinct from None (a real storable value).
_MISSING = object()

#: CSR-backed relation cell sentinel (M6): the value lives in the store's
#: dataflow edge table; the facade descriptor rematerializes an immutable
#: tuple view on first access and caches it back through ``cell_set``.
#: Distinct from ``_MISSING`` so present/absent semantics stay exact —
#: ``cell_del`` on a ``_CSR`` cell clears to genuinely absent.
_CSR = object()

#: Shared-fact cell sentinel (M7): the value lives in the store's fact
#: blocks (the FunctionCall / ParamAlias group tables); the facade
#: descriptor hydrates the exact public container type for the row on first
#: access and caches it back. Same present/absent semantics as ``_CSR``.
_FACT = object()

#: Overlay-miss sentinel local to cell reads.
_NO_OVERLAY = object()

#: Minimum row count for the freeze-time columnar transpose. Below this the
#: per-trace fixed cost of ~190 column objects (~29 KB measured) exceeds any
#: packing win, so small sealed stores keep their row-major cells (write and
#: delete semantics are identical either way; ``new_row`` refuses once sealed).
_TRANSPOSE_MIN_ROWS = 512


class OpStoreLayout:
    """Immutable field-name-to-field-id layout shared by every op store.

    Parameters
    ----------
    names:
        Stored field names in declared state order (the ``_OP_SLOT_NAMES``
        order: FIELD_ORDER minus computed property-backed names, plus the
        dynamic runtime slots).
    """

    __slots__ = ("names", "fid_by_name", "n_fields")

    def __init__(self, names: tuple[str, ...]) -> None:
        """Freeze the layout for ``names``."""

        self.names = tuple(names)
        self.fid_by_name = {name: fid for fid, name in enumerate(self.names)}
        self.n_fields = len(self.names)


class _FrozenColumn:
    """One frozen per-field column with presence tracking.

    Parameters
    ----------
    values:
        numpy array (packed) or Python list (object column).
    present:
        ``bytearray`` presence mask, or ``None`` when every row is present.
    packed:
        Whether ``values`` is a numpy array requiring ``.item()`` on read.
    """

    __slots__ = ("values", "present", "packed")

    def __init__(self, values: Any, present: bytearray | None, packed: bool) -> None:
        """Bind backing storage produced by ``OpRowStore.freeze``."""

        self.values = values
        self.present = present
        self.packed = packed

    def get(self, row: int) -> Any:
        """Return the cell value at ``row``, or ``_MISSING``."""

        present = self.present
        if present is not None and not present[row]:
            return _MISSING
        value = self.values[row]
        if self.packed:
            return value.item()
        return value

    def clear(self, row: int) -> None:
        """Release the cell at ``row`` (object columns drop the reference)."""

        if self.present is None:
            self.present = bytearray(b"\x01" * len(self.values))
        self.present[row] = 0
        if not self.packed:
            self.values[row] = None


class OpRowStore:
    """Per-trace row store for every ``Op`` of one captured run.

    Parameters
    ----------
    layout:
        Shared field layout (one module-level instance per process).
    """

    __slots__ = (
        "layout",
        "_rows",
        "_columns",
        "_overlay",
        "_n_rows",
        "_sealed",
        "dataflow_edges",
        "ref_labels",
        "fact_blocks",
    )

    def __init__(self, layout: OpStoreLayout) -> None:
        """Create an empty building-phase store."""

        self.layout = layout
        self._rows: list[list[Any]] | None = []
        self._columns: list[_FrozenColumn] | None = None
        self._overlay: dict[int, Any] = {}
        self._n_rows = 0
        self._sealed = False
        # The M6 dataflow family: bound at the freeze-time relation
        # conversion (same EdgeTable object registered in the owning
        # TraceCore's edge registry) together with the row -> reference-label
        # table the facade descriptors use to rematerialize views.
        self.dataflow_edges: Any = None
        self.ref_labels: dict[int, str] | None = None
        # The M7 shared-fact blocks (FunctionCall / ParamAlias group
        # tables), bound by the same freeze-time conversion.
        self.fact_blocks: Any = None

    def __len__(self) -> int:
        """Return the number of rows ever appended (removed rows included)."""

        return self._n_rows

    @property
    def frozen(self) -> bool:
        """Return whether the store has been sealed by ``freeze()``."""

        return self._sealed

    def new_row(self) -> int:
        """Append one all-``_MISSING`` row and return its row id."""

        rows = self._rows
        if rows is None or self._sealed:
            raise RuntimeError("op store is frozen; no new rows may be appended")
        row = self._n_rows
        self._n_rows = row + 1
        rows.append([_MISSING] * self.layout.n_fields)
        return row

    def adopt_row(self, cells: list[Any]) -> int:
        """Append one pre-built row (the bulk ingress path) and return its id.

        ``cells`` must be layout-ordered with ``_MISSING`` for unset fields;
        the store takes ownership of the list.
        """

        rows = self._rows
        if rows is None or self._sealed:
            raise RuntimeError("op store is frozen; no new rows may be appended")
        if len(cells) != self.layout.n_fields:
            raise ValueError("row cell count does not match the layout")
        row = self._n_rows
        self._n_rows = row + 1
        rows.append(cells)
        return row

    def cell_get(self, row: int, fid: int) -> Any:
        """Return one cell value, or ``_MISSING`` when unset."""

        rows = self._rows
        if rows is not None:
            return rows[row][fid]
        overlay = self._overlay
        if overlay:
            value = overlay.get(row * self.layout.n_fields + fid, _NO_OVERLAY)
            if value is not _NO_OVERLAY:
                return value
        columns = self._columns
        assert columns is not None
        return columns[fid].get(row)

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Write one cell (base while building, overlay after freeze)."""

        rows = self._rows
        if rows is not None:
            rows[row][fid] = value
        else:
            self._overlay[row * self.layout.n_fields + fid] = value

    def cell_del(self, row: int, fid: int) -> bool:
        """Delete one cell; return whether it was previously set.

        Frozen object columns release the value in place (so a removed op's
        payloads keep being freed); packed columns tombstone via the overlay.
        """

        rows = self._rows
        if rows is not None:
            row_cells = rows[row]
            if row_cells[fid] is _MISSING:
                return False
            row_cells[fid] = _MISSING
            return True
        if self.cell_get(row, fid) is _MISSING:
            return False
        key = row * self.layout.n_fields + fid
        columns = self._columns
        assert columns is not None
        column = columns[fid]
        if column.packed:
            self._overlay[key] = _MISSING
        else:
            self._overlay.pop(key, None)
            column.clear(row)
        return True

    def items(self, row: int) -> Iterator[tuple[str, Any]]:
        """Yield ``(field_name, value)`` for every set cell in layout order."""

        names = self.layout.names
        for fid, name in enumerate(names):
            value = self.cell_get(row, fid)
            if value is not _MISSING:
                yield name, value

    def retained_bytes(self) -> int:
        """Return shallow structural bytes retained by this store.

        Counts the store object, row lists (building) or column
        backings/presence masks/overlay (frozen) -- never the cell VALUES,
        matching the former shallow per-op ``sys.getsizeof`` accounting.
        """

        import sys

        total = sys.getsizeof(self)
        rows = self._rows
        if rows is not None:
            total += sys.getsizeof(rows)
            for row_cells in rows:
                total += sys.getsizeof(row_cells)
            return total
        total += sys.getsizeof(self._overlay)
        columns = self._columns
        assert columns is not None
        total += sys.getsizeof(columns)
        for column in columns:
            total += sys.getsizeof(column)
            total += sys.getsizeof(column.values)
            if column.present is not None:
                total += sys.getsizeof(column.present)
        return total

    def rows_building(self) -> list[list[Any]] | None:
        """Return the raw building-phase rows, or ``None`` once frozen.

        Exposed for the trace-level metadata compaction sweep, which pools
        immutable values column-by-column without the attribute protocol.
        """

        return self._rows

    def freeze(self) -> None:
        """Seal the store; transpose into columns when the trace is large.

        Sealing always stops row appends. The physical columnar transpose
        (numeric packing) only pays for itself past ``_TRANSPOSE_MIN_ROWS``;
        smaller sealed stores keep their row-major cells with identical
        read/write/delete behavior.
        """

        rows = self._rows
        if rows is None or self._sealed:
            self._sealed = True
            return
        self._sealed = True
        if self._n_rows < _TRANSPOSE_MIN_ROWS:
            return
        n_fields = self.layout.n_fields
        columns: list[_FrozenColumn] = []
        for fid in range(n_fields):
            column_values = [row_cells[fid] for row_cells in rows]
            columns.append(_freeze_column(column_values))
        self._columns = columns
        self._rows = None


def _freeze_column(column_values: list[Any]) -> _FrozenColumn:
    """Build one frozen column, packing exact-typed numerics when safe.

    Packing requires every PRESENT value to be an exact ``bool``, exact
    ``int`` (within int64 range), or exact ``float`` -- one uniform type per
    column. Subclass instances (``Bytes``, ``Flops``, ``Duration``...),
    ``None`` values, and mixed types keep the object backing so every read
    returns the exact object/type that was stored.
    """

    present: bytearray | None = None
    any_missing = False
    for value in column_values:
        if value is _MISSING:
            any_missing = True
            break
    if any_missing:
        present = bytearray(
            0 if value is _MISSING else 1 for value in column_values
        )

    value_cls: type | None = None
    uniform = True
    for value in column_values:
        if value is _MISSING:
            continue
        cls = value.__class__
        if cls is not bool and cls is not int and cls is not float:
            uniform = False
            break
        if value_cls is None:
            value_cls = cls
        elif cls is not value_cls:
            uniform = False
            break
    if uniform and value_cls is not None:
        dtype = {bool: np.bool_, int: np.int64, float: np.float64}[value_cls]
        fill: Any = False if value_cls is bool else value_cls(0)
        try:
            packed_values: Any = np.array(
                [fill if value is _MISSING else value for value in column_values],
                dtype=dtype,
            )
        except OverflowError:
            pass
        else:
            return _FrozenColumn(packed_values, present, True)
    object_values = [None if value is _MISSING else value for value in column_values]
    return _FrozenColumn(object_values, present, False)


class DetachedOpStore:
    """Single-row op store for detached facades (copy/pickle/fork/preview).

    Parameters
    ----------
    layout:
        Shared field layout.
    """

    __slots__ = ("layout", "_cells")

    #: Detached rows never carry CSR-backed relations or shared-fact cells
    #: (class-level constants so the facade descriptors can probe both store
    #: kinds uniformly).
    dataflow_edges = None
    ref_labels = None
    fact_blocks = None

    def __init__(self, layout: OpStoreLayout) -> None:
        """Create an empty single-row store."""

        self.layout = layout
        self._cells: list[Any] = [_MISSING] * layout.n_fields

    def adopt_row(self, cells: list[Any]) -> int:
        """Adopt one pre-built layout-ordered cell list as the single row."""

        if len(cells) != self.layout.n_fields:
            raise ValueError("row cell count does not match the layout")
        self._cells = cells
        return 0

    def __len__(self) -> int:
        """Return the row count (always one)."""

        return 1

    @property
    def frozen(self) -> bool:
        """Return ``False``: detached rows never freeze."""

        return False

    def cell_get(self, row: int, fid: int) -> Any:
        """Return one cell value, or ``_MISSING`` when unset."""

        return self._cells[fid]

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Write one cell."""

        self._cells[fid] = value

    def cell_del(self, row: int, fid: int) -> bool:
        """Delete one cell; return whether it was previously set."""

        cells = self._cells
        if cells[fid] is _MISSING:
            return False
        cells[fid] = _MISSING
        return True

    def items(self, row: int) -> Iterator[tuple[str, Any]]:
        """Yield ``(field_name, value)`` for every set cell in layout order."""

        names = self.layout.names
        cells = self._cells
        for fid, name in enumerate(names):
            value = cells[fid]
            if value is not _MISSING:
                yield name, value

    def retained_bytes(self) -> int:
        """Return shallow structural bytes retained by this store."""

        import sys

        return sys.getsizeof(self) + sys.getsizeof(self._cells)
