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

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from .groups import GroupRef, MembershipGroups

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

    __slots__ = ("fid_by_name", "n_fields", "names")

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

    __slots__ = ("packed", "present", "values")

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
        "_columns",
        "_cow_shared",
        "_n_rows",
        "_overlay",
        "_rows",
        "_sealed",
        "dataflow_edges",
        "fact_blocks",
        "layout",
        "ref_labels",
    )

    def __init__(self, layout: OpStoreLayout) -> None:
        """Create an empty building-phase store."""

        self.layout = layout
        self._rows: list[list[Any]] | None = []
        self._columns: list[_FrozenColumn] | None = None
        self._overlay: dict[int, Any] = {}
        self._n_rows = 0
        self._sealed = False
        # Sticky flag set by the first OpStoreView taken over this store
        # (M11 COW fork). Once shared, deletes must stop releasing object
        # column cells IN PLACE — a fork reads the same column backing —
        # and tombstone through the overlay instead. Memory-only effect:
        # a removed op's payloads on a forked-from trace are released with
        # the store rather than at removal time.
        self._cow_shared = False
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
        if column.packed or self._cow_shared:
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


#: Live write-audit collectors keyed by audited store id (M10 step
#: contracts). Populated only while a postprocess step audit is active.
_AUDIT_COLLECTORS: dict[int, set[int]] = {}


class _AuditedOpRowStore(OpRowStore):
    """Write-recording twin used ONLY during env-gated step audits.

    ``begin_cell_write_audit`` swaps a store's ``__class__`` to this subclass
    (layout-identical: empty ``__slots__``), so the un-audited hot path pays
    ZERO extra cost — no per-write branch exists on ``OpRowStore`` itself.
    """

    __slots__ = ()

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Record the written column id, then perform the write."""

        collector = _AUDIT_COLLECTORS.get(id(self))
        if collector is not None:
            collector.add(fid)
        OpRowStore.cell_set(self, row, fid, value)

    def cell_del(self, row: int, fid: int) -> bool:
        """Record the deleted column id, then perform the delete."""

        collector = _AUDIT_COLLECTORS.get(id(self))
        if collector is not None:
            collector.add(fid)
        return OpRowStore.cell_del(self, row, fid)


def begin_cell_write_audit(store: OpRowStore) -> None:
    """Start recording column writes on ``store`` (idempotent)."""

    if store.__class__ is OpRowStore:
        store.__class__ = _AuditedOpRowStore
    _AUDIT_COLLECTORS.setdefault(id(store), set())


def end_cell_write_audit(store: OpRowStore) -> set[str]:
    """Stop recording and return the written column NAMES."""

    observed = _AUDIT_COLLECTORS.pop(id(store), set())
    if store.__class__ is _AuditedOpRowStore:
        store.__class__ = OpRowStore  # type: ignore[assignment]
    names = store.layout.names
    return {names[fid] for fid in observed}


class DetachedOpStore:
    """Single-row op store for detached facades (copy/pickle/fork/preview).

    Parameters
    ----------
    layout:
        Shared field layout.
    """

    __slots__ = ("_cells", "layout")

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


#: Exact types returned uncopied (and untranslated) by the COW copier.
_COW_ATOMIC = frozenset(
    {str, int, float, bool, bytes, complex, type(None)}
)


def cow_copy_value(value: Any, translate: Callable[[Any], Any] | None) -> Any:
    """Structurally copy one COW-read value for fork-side isolation.

    Exact builtin containers are rebuilt (so fork-side in-place mutation can
    never reach the shared base); ``translate`` maps parent record facades to
    their fork facades (returning ``None`` for non-records); every other
    object — tensors, callables, interned immutables, quantity subclasses —
    is returned by identity, preserving the fork's payload-sharing contract.
    All-identity tuples/frozensets return the original object so interned
    immutable views stay shared.
    """

    cls = value.__class__
    if cls in _COW_ATOMIC:
        return value
    if cls is dict:
        return {
            cow_copy_value(key, translate): cow_copy_value(item, translate)
            for key, item in value.items()
        }
    if cls is list:
        return [cow_copy_value(item, translate) for item in value]
    if cls is set:
        return {cow_copy_value(item, translate) for item in value}
    if cls is tuple or cls is frozenset:
        copied = [cow_copy_value(item, translate) for item in value]
        for original, item_copy in zip(value, copied):
            if original is not item_copy:
                return cls(copied)
        return value
    if translate is not None:
        mapped = translate(value)
        if mapped is not None:
            return mapped
    return value


class _ViewFactBlocks:
    """Fact-block adapter isolating hydrated containers for one fork view."""

    __slots__ = ("_base", "_view")

    def __init__(self, base: Any, view: OpStoreView) -> None:
        """Bind the base fact blocks and the owning view."""

        self._base = base
        self._view = view

    def hydrate(self, row: int, name: str) -> Any:
        """Hydrate one shared fact, translating record members for the fork."""

        return cow_copy_value(
            self._base.hydrate(row, name), self._view.record_translator
        )


class OpStoreView:
    """Per-fork COW view over one sealed base store (the M11 fork substrate).

    The view shares the base's frozen storage and isolates everything mutable:

    * Fork writes and deletes land in the view's own overlay, never the base.
    * The base's post-freeze overlay is SNAPSHOT at construction (row-major
      sealed bases snapshot their row lists instead), so parent writes after
      the fork stay invisible in both directions.
    * Exact builtin mutable containers are eagerly copied into the view
      overlay at fork time (``isolate_mutable_cells``, run by the fork
      builder once the record translator is installed), so isolation holds
      in BOTH directions from fork time — the copy-on-first-read in
      ``_isolate`` remains as the read-cache backstop (tensors/callables
      inside stay shared by identity — the payload-sharing fork contract).
    * ``GroupRef`` cells translate to the fork core's cloned group tables, so
      removal scrub on either trace never reaches the other.
    * Record facades inside containers and hydrated fact blocks translate
      through ``record_translator`` (installed by the fork builder) to the
      fork's own facades.

    Non-builtin mutable cell values (custom objects) are shared by identity —
    the same residual the shallow fork path already accepted for
    replay-unaffected fields.
    """

    __slots__ = (
        "_base_overlay",
        "_base_rows",
        "_group_refs",
        "_group_tables",
        "_overlay",
        "base",
        "fact_blocks",
        "record_translator",
    )

    def __init__(
        self,
        base: "OpRowStore | OpStoreView",
        group_tables: dict[int, MembershipGroups] | None = None,
    ) -> None:
        """Snapshot ``base`` (which must be sealed) into a COW view.

        ``base`` may itself be an ``OpStoreView`` (fork of a fork): the new
        view flattens onto the ROOT store, snapshotting the parent view's
        effective overlay (its private base snapshot plus its own writes) —
        the parent view never mutates its base snapshot, so sharing the
        row snapshot list is safe.
        """

        if not base.frozen:
            raise RuntimeError("OpStoreView requires a sealed base store")
        self.base: OpRowStore
        self._base_rows: list[list[Any]] | None
        self._base_overlay: dict[int, Any]
        if isinstance(base, OpStoreView):
            self.base = base.base
            self._base_rows = base._base_rows
            self._base_overlay = {**base._base_overlay, **base._overlay}
        else:
            self.base = base
            base._cow_shared = True
            base_rows = base._rows
            if base_rows is not None:
                # Sealed row-major base: writes/deletes mutate rows in
                # place, so the view snapshots the row lists (cheap under
                # the transpose threshold) and ignores the live cells
                # thereafter.
                self._base_rows = [list(row_cells) for row_cells in base_rows]
                self._base_overlay = {}
            else:
                self._base_rows = None
                self._base_overlay = dict(base._overlay)
        self._overlay: dict[int, Any] = {}
        self._group_tables = group_tables
        self._group_refs: dict[tuple[int, int], GroupRef] = {}
        self.record_translator: Callable[[Any], Any] | None = None
        self.fact_blocks = (
            _ViewFactBlocks(self.base.fact_blocks, self)
            if self.base.fact_blocks is not None
            else None
        )

    def __len__(self) -> int:
        """Return the base row count."""

        return len(self.base)

    @property
    def layout(self) -> OpStoreLayout:
        """Return the shared field layout."""

        return self.base.layout

    @property
    def dataflow_edges(self) -> Any:
        """Return the shared (frozen) dataflow edge table."""

        return self.base.dataflow_edges

    @property
    def ref_labels(self) -> Any:
        """Return the shared row -> reference-label table."""

        return self.base.ref_labels

    @property
    def frozen(self) -> bool:
        """Return ``True``: views only exist over sealed bases."""

        return True

    def new_row(self) -> int:
        """Refuse: fork views never append rows."""

        raise RuntimeError("op store view is frozen; no new rows may be appended")

    def adopt_row(self, cells: list[Any]) -> int:
        """Refuse: fork views never adopt rows."""

        raise RuntimeError("op store view is frozen; no new rows may be appended")

    def rows_building(self) -> None:
        """Return ``None``: a view is never in the building phase."""

        return

    def _translate_group_ref(self, ref: GroupRef) -> GroupRef:
        """Return the fork-side ref for one shared group cell."""

        tables = self._group_tables
        if tables is None:
            return ref
        clone = tables.get(id(ref.groups))
        if clone is None:
            return ref
        key = (id(ref.groups), ref.group_id)
        fork_ref = self._group_refs.get(key)
        if fork_ref is None:
            fork_ref = GroupRef(clone, ref.group_id)
            self._group_refs[key] = fork_ref
        return fork_ref

    def _isolate(self, key: int, value: Any) -> Any:
        """Isolate one base-read value, caching fork copies in the overlay."""

        if value is _MISSING or value is _CSR or value is _FACT:
            return value
        cls = value.__class__
        if cls in _COW_ATOMIC:
            return value
        if cls is GroupRef:
            fork_ref = self._translate_group_ref(value)
            if fork_ref is not value:
                self._overlay[key] = fork_ref
            return fork_ref
        if cls is dict or cls is list or cls is set or cls is tuple or cls is frozenset:
            copied = cow_copy_value(value, self.record_translator)
            self._overlay[key] = copied
            return copied
        translate = self.record_translator
        if translate is not None:
            mapped = translate(value)
            if mapped is not None:
                self._overlay[key] = mapped
                return mapped
        return value

    def cell_get(self, row: int, fid: int) -> Any:
        """Return one cell value with fork-side isolation applied."""

        key = row * self.base.layout.n_fields + fid
        value = self._overlay.get(key, _NO_OVERLAY)
        if value is not _NO_OVERLAY:
            return value
        value = self._base_overlay.get(key, _NO_OVERLAY)
        if value is _NO_OVERLAY:
            base_rows = self._base_rows
            if base_rows is not None:
                value = base_rows[row][fid]
            else:
                columns = self.base._columns
                assert columns is not None
                value = columns[fid].get(row)
        return self._isolate(key, value)

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Write one cell into the fork overlay."""

        self._overlay[row * self.base.layout.n_fields + fid] = value

    def cell_del(self, row: int, fid: int) -> bool:
        """Tombstone one cell in the fork overlay; report prior presence."""

        if self.cell_get(row, fid) is _MISSING:
            return False
        self._overlay[row * self.base.layout.n_fields + fid] = _MISSING
        return True

    def items(self, row: int) -> Iterator[tuple[str, Any]]:
        """Yield ``(field_name, value)`` for every set cell in layout order."""

        names = self.base.layout.names
        for fid, name in enumerate(names):
            value = self.cell_get(row, fid)
            if value is not _MISSING:
                yield name, value

    def isolate_mutable_cells(self) -> None:
        """Eagerly isolate every mutable-container cell into the fork overlay.

        Called once by the fork builder AFTER the record translator is
        installed. Copy-on-first-read alone left a window where a PARENT's
        in-place container mutation between fork time and the fork's first
        read of that cell leaked into the fork; pre-isolating restores the
        deepcopy fork's snapshot semantics in both directions (fork writes
        were already overlay-isolated). Cells already written or isolated
        stay untouched; atomic values, sentinels, and shared-by-identity
        payloads (tensors, callables) never enter the overlay.
        """

        n_fields = self.base.layout.n_fields
        overlay = self._overlay
        base_overlay = self._base_overlay
        base_rows = self._base_rows
        atomic = _COW_ATOMIC
        translate = self.record_translator

        def _isolate_eager(key: int, value: Any) -> None:
            # ``_isolate`` minus the unconditional read-cache: identity
            # results (interned tuples/frozensets of atomics, untranslated
            # records) stay OUT of the overlay so the eager sweep does not
            # materialize a per-fork copy of every immutable view.
            cls = value.__class__
            if cls is GroupRef:
                fork_ref = self._translate_group_ref(value)
                if fork_ref is not value:
                    overlay[key] = fork_ref
                return
            if cls is dict or cls is list or cls is set or cls is tuple or cls is frozenset:
                copied = cow_copy_value(value, translate)
                if copied is not value:
                    overlay[key] = copied
                return
            if translate is not None:
                mapped = translate(value)
                if mapped is not None:
                    overlay[key] = mapped

        if base_rows is not None:
            for row, row_cells in enumerate(base_rows):
                row_key = row * n_fields
                for fid, value in enumerate(row_cells):
                    if value.__class__ in atomic:
                        continue
                    key = row_key + fid
                    if key not in overlay:
                        _isolate_eager(key, value)
        else:
            columns = self.base._columns
            assert columns is not None
            n_rows = len(self.base)
            for fid in range(n_fields):
                column = columns[fid]
                for row in range(n_rows):
                    key = row * n_fields + fid
                    if key in overlay:
                        continue
                    value = base_overlay.get(key, _NO_OVERLAY)
                    if value is _NO_OVERLAY:
                        value = column.get(row)
                    if value.__class__ in atomic:
                        continue
                    _isolate_eager(key, value)

    def retained_bytes(self) -> int:
        """Return shallow structural bytes retained by this view alone."""

        import sys

        total = sys.getsizeof(self) + sys.getsizeof(self._overlay)
        total += sys.getsizeof(self._base_overlay)
        if self._base_rows is not None:
            total += sys.getsizeof(self._base_rows)
            for row_cells in self._base_rows:
                total += sys.getsizeof(row_cells)
        return total
