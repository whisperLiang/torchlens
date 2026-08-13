"""Duplicate/empty mutable-container cell pooling (the M14 memory slice).

Covers the ``PooledCell`` contract end to end on a real capture: pooled
cells exist on the finished trace (the memory win is real), reads hydrate
fresh exact-type containers with stable identity and per-row isolation,
forks and pickles never observe the internal encoding, aliased containers
refuse to pool, and the empty/duplicate thresholds hold.
"""

from __future__ import annotations

import pickle
from collections import defaultdict

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core.op_store import PooledCell
from torchlens.data_classes.op import (
    _container_pool_key,
    _pool_container_cells,
)

pytestmark = pytest.mark.smoke


class _Stack(nn.Module):
    def __init__(self, blocks: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


@pytest.fixture(scope="module")
def stack_trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(_Stack(), torch.zeros(1, 4))


def _pooled_cells_by_field(store) -> dict[str, int]:
    counts: dict[str, int] = {}
    names = store.layout.names
    rows = getattr(store, "_rows", None)
    if rows is not None:
        for row_cells in rows:
            for fid, value in enumerate(row_cells):
                if value.__class__ is PooledCell:
                    counts[names[fid]] = counts.get(names[fid], 0) + 1
    else:
        for fid, column in enumerate(store._columns):
            if not column.packed:
                for value in column.values:
                    if value.__class__ is PooledCell:
                        counts[names[fid]] = counts.get(names[fid], 0) + 1
    return counts


def test_capture_pools_op_and_kind_cells(stack_trace: tl.Trace) -> None:
    """A finished capture holds PooledCells for the allowlisted op fields
    and the repetitive kind-table cells (the measured win exists)."""

    core = stack_trace.__dict__["_trace_core"]
    op_counts = _pooled_cells_by_field(core.ops)
    n_ops = len(stack_trace.ops.keys())
    for field in ("annotations", "interventions", "var_names", "func_autocast_state"):
        assert op_counts.get(field, 0) >= n_ops - 1, (field, op_counts)
    module_counts = _pooled_cells_by_field(core.kind_rows["module"])
    assert module_counts.get("forward_hooks", 0) >= 2, module_counts


def test_hydration_identity_and_row_isolation(stack_trace: tl.Trace) -> None:
    """First read hydrates the exact public type, caches it back (stable
    identity), and never shares mutable state across rows."""

    op_a = stack_trace["relu_1_2"]
    op_b = stack_trace["relu_2_4"]
    hydrated = op_a.annotations
    assert type(hydrated) is dict and hydrated == {}
    assert op_a.annotations is hydrated
    hydrated["marker"] = 1
    assert op_b.annotations == {} and op_b.annotations is not hydrated

    autocast_a = op_a.func_autocast_state
    autocast_a["cpu"]["injected"] = True
    assert "injected" not in op_b.func_autocast_state["cpu"]

    entry_keys = op_a.module_entry_arg_keys
    assert type(entry_keys) is defaultdict and entry_keys.default_factory is list

    module = stack_trace.modules["layers.0"]
    hooks = module.forward_hooks
    assert type(hooks) is list and hooks == []
    assert module.forward_hooks is hooks
    hooks.append("shadow")
    assert stack_trace.modules["layers.1"].forward_hooks == []


def test_fork_never_shares_hydrated_containers(stack_trace: tl.Trace) -> None:
    """A COW fork hydrates its own containers; mutations never cross."""

    fork = stack_trace.fork()
    base_op = stack_trace["relu_3_6"]
    fork_op = fork["relu_3_6"]
    fork_notes = fork_op.annotations
    fork_notes["fork_only"] = True
    assert "fork_only" not in base_op.annotations
    base_op.annotations["base_only"] = True
    assert "base_only" not in fork_op.annotations


def test_pickle_round_trip_never_leaks_encoding(stack_trace: tl.Trace) -> None:
    """Op pickle state carries real containers, never PooledCell."""

    layer = stack_trace["relu_4_8"]
    op = next(op for _, op in layer.ops.items())
    state = op.__getstate__()
    assert "annotations" in state and "func_autocast_state" in state
    for name, value in state.items():
        assert value.__class__ is not PooledCell, name
    restored = pickle.loads(pickle.dumps(stack_trace["linear_2_3"]))
    assert restored.annotations == {}
    assert restored.func_autocast_state == op.func_autocast_state

    module_restored = pickle.loads(pickle.dumps(stack_trace.modules["layers.2"]))
    assert module_restored.forward_hooks == []


def test_alias_guard_refuses_shared_containers() -> None:
    """A container reachable from two swept cells never pools."""

    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout

    layout = OpStoreLayout(("a", "b", "c"))
    store = OpRowStore(layout)
    shared = {"x": 1}
    for _ in range(4):
        row = store.new_row()
        store.cell_set(row, 0, shared)  # aliased: one object, many cells
        store.cell_set(row, 1, {"x": 1})  # equal but distinct: pools
        store.cell_set(row, 2, [])  # empty: pools
    _pool_container_cells([(store, None)], {})
    rows = store.rows_building()
    assert all(row_cells[0] is shared for row_cells in rows)
    pooled_b = {id(row_cells[1]) for row_cells in rows}
    assert len(pooled_b) == 1 and rows[0][1].__class__ is PooledCell
    assert rows[0][2].__class__ is PooledCell


def test_duplicate_threshold_and_unpoolable_content() -> None:
    """Non-empty content below 3 occurrences, unknown member types, and
    subclassed containers keep their original cells."""

    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout

    class Custom:
        pass

    layout = OpStoreLayout(("a",))
    store = OpRowStore(layout)
    pair_one = {"k": 1}
    pair_two = {"k": 1}
    tensorful = {"t": torch.zeros(1)}
    custom = {"c": Custom()}
    for value in (pair_one, pair_two, tensorful, custom):
        row = store.new_row()
        store.cell_set(row, 0, value)
    _pool_container_cells([(store, None)], {})
    rows = store.rows_building()
    assert rows[0][0] is pair_one and rows[1][0] is pair_two
    assert rows[2][0] is tensorful and rows[3][0] is custom


def test_container_pool_key_injectivity() -> None:
    """Keys are class-tagged: bool/int/float never collapse, dict order and
    container classes distinguish, defaultdict factories key separately."""

    def key(value):
        return _container_pool_key(value, 0, [])

    assert key({"a": True}) != key({"a": 1})
    assert key([1.0]) != key([1])
    assert key({"a": 1, "b": 2}) != key({"b": 2, "a": 1})
    assert key([1, 2]) != key({1: 2})
    dd_list = defaultdict(list)
    dd_set = defaultdict(set)
    assert key(dd_list) != key(dd_set) != key({})
    assert key({"nested": [1, {2}]}) is not None
    cyclic: list = []
    cyclic.append(cyclic)
    assert key(cyclic) is None
