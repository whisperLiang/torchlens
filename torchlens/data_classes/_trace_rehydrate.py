"""Load-time rehydration of a coreless trace into the columnar store (F9).

``_trace_core`` is ``FieldPolicy.DROP``, so every unpickled Trace (plain
pickle AND ``.tlspec`` bundle loads, which route through the same
``__setstate__``) used to arrive as a dict-backed ISLAND: one detached
single-row store per record, no shared edge table, no group sharing —
outside the single-truth architecture entirely.

``rehydrate_trace_core`` runs at the end of ``Trace.__setstate__``: it
adopts every restored ``Op`` into one fresh per-trace ``OpRowStore``, the
non-Op record facades into their kind tables, then replays the standard
relation freeze (dataflow -> edge-occurrence CSR, interned views, shared
``GroupRef`` group tables — whose conversion explicitly coalesces the
equal-but-distinct members a pickle round-trip produces — and fact
pooling) and seals the store exactly like the capture-time freeze seam
(columnar transpose above the row threshold).

Boundaries (documented, tested):

* Backward records (``GradFn``/``GradFnCall``/``BackwardPass``) stay
  detached-backed — loaded traces have no event stream, so there is no
  epoch to rebuild; a NEW backward on the restored trace binds fresh
  epochs through the normal projection path.
* Rehydration is strictly best-effort: any inconsistency (mixed store
  ownership, layout drift) aborts and leaves the load exactly as the
  coreless island behaved — a load never fails because of it.
* Compaction passes are not re-run: pickle already preserves shared
  identity within one artifact, so pooled metadata stays pooled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trace import Trace

#: Record kind table names keyed by facade class name (M8 vocabulary).
_KIND_BY_CLASS_NAME = {
    "Param": "param",
    "Buffer": "buffer",
    "Module": "module",
    "ModuleCall": "module_call",
    "FuncCallLocation": "func_call_location",
}


def rehydrate_trace_core(trace: "Trace") -> bool:
    """Adopt a coreless trace's detached records into a fresh sealed core.

    Returns ``True`` when a core was built and sealed, ``False`` when the
    trace already has a core, has no ops, or rehydration aborted (the
    coreless-island behavior is preserved on abort).
    """

    if trace.__dict__.get("_trace_core") is not None:
        return False
    from .._trace_core.core import TraceCore
    from .._trace_core.op_store import DetachedOpStore, OpRowStore
    from .._trace_core.record_rows import CORE_KEY, adopt_records, detach_record
    from .._trace_core.relation_views import freeze_trace_relation_views
    from ._trace_fork import _iter_parent_ops, _iter_parent_records
    from .op import _OP_STORE_LAYOUT, _object_setattr

    ops = list(_iter_parent_ops(trace))
    if not ops:
        return False

    store = OpRowStore(_OP_STORE_LAYOUT)
    adopted: list[tuple[Any, Any, int]] = []
    records_by_kind: dict[str, dict[int, Any]] = {}
    try:
        for op in ops:
            try:
                bound = object.__getattribute__(op, "_core")
            except AttributeError:
                return False
            if bound is store:
                continue
            if not isinstance(bound, DetachedOpStore):
                # Mixed/foreign ownership: not a plain coreless load.
                return False
            if bound.layout is not _OP_STORE_LAYOUT:
                return False
            row = store.adopt_row(bound._cells)
            adopted.append((op, bound, row))
            _object_setattr(op, "_core", store)
            _object_setattr(op, "_row", row)

        core = TraceCore()
        core.ops = store
        for record in _iter_parent_records(trace):
            kind = _KIND_BY_CLASS_NAME.get(type(record).__name__)
            if kind is None:
                continue
            records_by_kind.setdefault(kind, {}).setdefault(id(record), record)
        for kind, records in records_by_kind.items():
            adopt_records(core, kind, records.values())

        trace.__dict__["_trace_core"] = core
        freeze_trace_relation_views(trace)
        store.freeze()
        for kind_store in core.kind_rows.values():
            kind_store.freeze()
    except Exception:
        # Best-effort: restore the detached bindings and stay coreless.
        for op, bound, _row in adopted:
            _object_setattr(op, "_core", bound)
            _object_setattr(op, "_row", 0)
        for records in records_by_kind.values():
            for record in records.values():
                record_store = record.__dict__.get(CORE_KEY)
                if record_store is not None and not isinstance(
                    record_store, DetachedOpStore
                ):
                    detach_record(record)
        trace.__dict__.pop("_trace_core", None)
        return False
    return True
