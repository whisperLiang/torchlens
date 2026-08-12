"""M4: standalone unit suite for the _trace_core substrate.

Zero production consumers yet — these tests prove the substrate honors the
converged semantic contracts (docs/reference/trace_core_design.md section
3.2/3.5) before the M5 Op seam, including executable prototypes of every
hard seam: pool injectivity, parallel-edge order, payload identity, overlay
COW + transactional rollback, core-level fork isolation, and facade
identity caching.
"""

from __future__ import annotations

import gc

import pytest
import torch

from torchlens._trace_core import (
    ColumnBuilder,
    EdgeTable,
    InternPool,
    PayloadArena,
    RowOverlay,
    TraceCore,
    Transaction,
)
from torchlens._trace_core.pools import ClosurePool
from torchlens.data_classes.op import Bytes


@pytest.mark.smoke
def test_column_build_freeze_round_trip() -> None:
    """Values and explicit missings survive build -> freeze byte-exactly."""

    column = ColumnBuilder("int64")
    column.append(7)
    column.append(None)
    column.append(-3)
    column.set(0, 9)
    frozen = column.freeze()
    assert list(frozen) == [9, None, -3]
    assert column.get(1) is None
    with pytest.raises(RuntimeError):
        column.set(0, 1)

    objects = ColumnBuilder("object")
    objects.append("label")
    objects.append(None)
    frozen_objects = objects.freeze()
    assert list(frozen_objects) == ["label", None]


@pytest.mark.smoke
def test_pool_exact_type_injectivity() -> None:
    """True / 1 / 1.0 / Bytes(1) never collapse (shipped _pool_key contract)."""

    pool = InternPool()
    ids = {pool.intern(value) for value in (True, 1, 1.0, Bytes(1))}
    assert len(ids) == 4
    assert pool.intern(True) == pool.intern(True)
    assert pool.value(pool.intern("x")) == "x"


@pytest.mark.smoke
def test_closure_pool_shares_and_materializes_fresh() -> None:
    """One entry per distinct closure; materialization is a fresh set."""

    closures = ClosurePool()
    first = closures.intern(["a", "b"])
    second = closures.intern({"b", "a"})
    assert first == second
    assert len(closures) == 1
    materialized = closures.materialize(first)
    materialized.add("c")
    assert closures.materialize(first) == {"a", "b"}


@pytest.mark.smoke
def test_edge_table_parallel_edges_and_order() -> None:
    """Parallel edges stay distinct; CSR preserves insertion order."""

    edges = EdgeTable()
    edges.add(0, 1, arg_position=0)
    edges.add(0, 1, arg_position=1)  # parallel edge, second arg slot
    edges.add(0, 2, arg_position=2)
    edges.add(3, 1)
    edges.freeze(n_source_rows=4, n_target_rows=4)

    out = list(edges.out_edges(0))
    assert [(e.target, e.arg_position) for e in out] == [(1, 0), (1, 1), (2, 2)]
    assert edges.sources_of(1) == [0, 0, 3]
    with pytest.raises(RuntimeError):
        EdgeTable().out_edges(0).__next__()


@pytest.mark.smoke
def test_payload_arena_identity_contract() -> None:
    """Same object = same handle; equal-but-distinct stays distinct;
    in-place mutation visible; replacement never disturbs aliases."""

    arena = PayloadArena()
    tensor = torch.zeros(3)
    equal_twin = torch.zeros(3)
    handle = arena.register(tensor)
    assert arena.register(tensor) == handle
    assert arena.register(equal_twin) != handle
    assert arena.value(handle) is tensor

    tensor.add_(1.0)  # in-place mutation stays visible through the handle
    assert torch.equal(arena.value(handle), torch.ones(3))

    replacement = arena.replace(handle, torch.full((3,), 5.0))
    assert replacement != handle
    assert arena.value(handle) is tensor  # original alias undisturbed


@pytest.mark.smoke
def test_overlay_and_transaction_rollback() -> None:
    """Overlay reads shadow the base; rollback restores atomically."""

    overlay = RowOverlay()
    overlay.write(0, "op.out", "patched")
    assert overlay.read(0, "op.out") == "patched"

    txn = Transaction({"main": overlay})
    overlay.write(0, "op.out", "mutated")
    overlay.write(1, "op.grad", "extra")
    txn.rollback()
    assert overlay.read(0, "op.out") == "patched"
    assert not overlay.has(1, "op.grad")


@pytest.mark.smoke
def test_core_write_routing_and_freeze() -> None:
    """Writes hit the base while building, the overlay after freeze."""

    core = TraceCore()
    table = core.table("op")
    row = table.new_row()
    core.write("op", row, "func_name", "relu")
    assert core.read("op", row, "func_name") == "relu"

    core.freeze()
    core.write("op", row, "func_name", "patched")
    assert core.read("op", row, "func_name") == "patched"
    assert table.get(row, "func_name") == "relu"  # base untouched


@pytest.mark.smoke
def test_core_fork_isolation_prototype() -> None:
    """COW fork: shared frozen base, isolated overlays both directions."""

    core = TraceCore()
    row = core.table("op").new_row()
    core.write("op", row, "func_name", "relu")
    core.freeze()
    core.write("op", row, "annotations", "parent-note")

    fork = core.fork()
    assert fork.read("op", row, "func_name") == "relu"
    assert fork.read("op", row, "annotations") == "parent-note"

    fork.write("op", row, "func_name", "fork-patch")
    assert core.read("op", row, "func_name") == "relu"
    core.write("op", row, "func_name", "parent-patch")
    assert fork.read("op", row, "func_name") == "fork-patch"
    assert fork.tables is core.tables  # base is shared, not copied


@pytest.mark.smoke
def test_facade_identity_cache_prototype() -> None:
    """Repeated facade lookups return the SAME object (strong cache)."""

    core = TraceCore()
    row = core.table("op").new_row()
    core.set_facade_factory(lambda kind, r: {"kind": kind, "row": r})
    first = core.facade("op", row)
    assert core.facade("op", row) is first

    fork = core.fork()
    fork_facade = fork.facade("op", row)
    assert fork_facade is not first  # fork has its own facade cache


@pytest.mark.smoke
def test_core_collectable_after_drop() -> None:
    """A dropped core (with facades and payloads) is garbage-collected."""

    core = TraceCore()
    core.table("op").new_row()
    core.payloads.register(torch.zeros(2))
    core.set_facade_factory(lambda kind, r: object())
    core.facade("op", 0)
    ref = core.weak_self()
    del core
    gc.collect()
    assert ref() is None


@pytest.mark.smoke
def test_partial_core_prototype() -> None:
    """A never-frozen core stays readable (partial-capture escape path)."""

    core = TraceCore()
    row = core.table("op").new_row()
    core.write("op", row, "func_name", "conv2d")
    # No freeze: the builder is still authoritative and readable.
    assert core.read("op", row, "func_name") == "conv2d"
    assert not core.table("op").frozen


@pytest.mark.smoke
def test_op_store_view_cow_isolation() -> None:
    """M11 store view: bidirectional write isolation + container copy-on-read."""

    from torchlens._trace_core.op_store import (
        _MISSING,
        OpRowStore,
        OpStoreLayout,
        OpStoreView,
    )

    layout = OpStoreLayout(("alpha", "beta", "items"))
    store = OpRowStore(layout)
    row = store.new_row()
    store.cell_set(row, 0, "base")
    store.cell_set(row, 2, {"k": [1, 2]})
    store.freeze()

    view = OpStoreView(store)
    view.cell_set(row, 0, "fork")
    assert store.cell_get(row, 0) == "base"
    store.cell_set(row, 0, "parent-after")
    assert view.cell_get(row, 0) == "fork"
    assert store.cell_get(row, 0) == "parent-after"

    forked_container = view.cell_get(row, 2)
    assert forked_container == {"k": [1, 2]}
    forked_container["k"].append(3)
    assert store.cell_get(row, 2) == {"k": [1, 2]}
    assert view.cell_get(row, 2)["k"] == [1, 2, 3]

    assert view.cell_del(row, 2)
    assert view.cell_get(row, 2) is _MISSING
    assert store.cell_get(row, 2) == {"k": [1, 2]}
    with pytest.raises(RuntimeError):
        view.new_row()


@pytest.mark.smoke
def test_op_store_view_translates_group_refs() -> None:
    """Group cells resolve through the fork's cloned tables, scrub-isolated."""

    from torchlens._trace_core.groups import GroupRef, MembershipGroups
    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout

    core = TraceCore()
    table = MembershipGroups(frozenset)
    core.groups["equivalent_ops"] = table
    group_id = table.add({"a", "b"})
    ref = GroupRef(table, group_id)

    layout = OpStoreLayout(("equivalent_ops",))
    store = OpRowStore(layout)
    row = store.new_row()
    store.cell_set(row, 0, ref)
    store.freeze()
    core.ops = store

    fork_core = core.fork()
    fork_ref = fork_core.ops.cell_get(row, 0)
    assert isinstance(fork_ref, GroupRef)
    assert fork_ref is not ref
    assert fork_ref.view() == frozenset({"a", "b"})
    assert fork_ref.view() is not ref.view()

    # Scrub on the parent never reaches the fork, and vice versa.
    table.replace(group_id, {"a"})
    assert fork_ref.view() == frozenset({"a", "b"})
    fork_core.groups["equivalent_ops"].replace(group_id, set())
    assert ref.view() == frozenset({"a"})


@pytest.mark.smoke
def test_core_transaction_rolls_back_stores_and_epochs() -> None:
    """One transaction restores core overlay, store overlays, and epochs."""

    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout

    core = TraceCore()
    layout = OpStoreLayout(("alpha",))
    store = OpRowStore(layout)
    row = store.new_row()
    store.cell_set(row, 0, "committed")
    store.freeze()
    core.ops = store
    kind_row = core.table("module").new_row()
    core.freeze()
    core.write("module", kind_row, "address", "committed-address")

    txn = core.transaction()
    store.cell_set(row, 0, "dirty")
    core.write("module", kind_row, "address", "dirty-address")
    core.backward_epochs.append(object())
    txn.rollback()

    assert store.cell_get(row, 0) == "committed"
    assert core.read("module", kind_row, "address") == "committed-address"
    assert core.backward_epochs == []


@pytest.mark.smoke
def test_facade_cache_weak_valued_with_strong_fallback() -> None:
    """The facade cache never pins unreferenced weak-able facades (M11 flip)."""

    import weakref

    class _WeakFacade:
        """Weak-referenceable facade stand-in."""

    core = TraceCore()
    row = core.table("op").new_row()
    core.set_facade_factory(lambda kind, r: _WeakFacade())
    first = core.facade("op", row)
    assert core.facade("op", row) is first
    facade_ref = weakref.ref(first)
    del first
    gc.collect()
    assert facade_ref() is None, "weak-valued cache must not pin a dropped facade"
    assert isinstance(core.facade("op", row), _WeakFacade)

    # Facade classes that refuse weak references (Op) stay strongly held.
    strong_core = TraceCore()
    strong_row = strong_core.table("op").new_row()
    strong_core.set_facade_factory(lambda kind, r: {"row": r})
    held = strong_core.facade("op", strong_row)
    assert strong_core.facade("op", strong_row) is held
    del held
    gc.collect()
    assert strong_core.facade("op", strong_row) == {"row": strong_row}
