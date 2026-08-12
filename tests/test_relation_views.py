"""Unit suite for the M6 relation freeze (``_trace_core/relation_views.py``).

Covers the conversion mechanics directly on synthetic stores — edge
projection, differential CSR clearing vs explicit-view fallback, label
resolution failure modes — plus the end-to-end behavioral surface on real
captures: immutable views, CSR rematerialization identity, idempotence,
legacy load coercion, and removal scrubbing over immutable views.
"""

from __future__ import annotations

import pickle

import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core.core import TraceCore
from torchlens._trace_core.op_store import _CSR, _MISSING, OpRowStore, OpStoreLayout
from torchlens._trace_core.relation_views import (
    DATAFLOW_FAMILY,
    freeze_op_relation_views,
    freeze_trace_relation_views,
    frozenset_view,
    intern_view,
    materialize_dataflow_view,
    tuple_view,
)

_SEED = 20260812


def _make_layout() -> OpStoreLayout:
    """Build a minimal layout carrying the relation fields under test."""

    return OpStoreLayout(
        (
            "layer_label",
            "parents",
            "children",
            "parent_arg_positions",
            "modules",
            "input_ancestors",
        )
    )


def _make_store(rows: list[dict[str, object]]) -> tuple[TraceCore, OpRowStore]:
    """Build a building-phase store from per-row field dicts."""

    layout = _make_layout()
    core = TraceCore()
    store = OpRowStore(layout)
    core.ops = store
    for fields in rows:
        cells: list[object] = [_MISSING] * layout.n_fields
        for name, value in fields.items():
            cells[layout.fid_by_name[name]] = value
        store.adopt_row(cells)
    return core, store


def _resolver(store: OpRowStore) -> dict[str, int]:
    """Map each row's ``layer_label`` cell to its row id."""

    fid = store.layout.fid_by_name["layer_label"]
    rows = store.rows_building()
    assert rows is not None
    return {
        cells[fid]: row for row, cells in enumerate(rows) if cells[fid] is not _MISSING
    }


class TestInternPool:
    """The view intern pool."""

    def test_equal_views_share_one_instance(self) -> None:
        pool: dict = {}
        first = tuple_view(["a", "b"], pool)
        second = tuple_view(("a", "b"), pool)
        assert first is second
        assert isinstance(first, tuple)

    def test_empty_collapses_to_shared_singleton(self) -> None:
        pool: dict = {}
        assert tuple_view([], pool) is tuple_view([], pool)
        assert frozenset_view(set(), pool) is frozenset_view(set(), pool)

    def test_unhashable_view_returned_unpooled(self) -> None:
        pool: dict = {}
        value = ([1, 2],)
        assert intern_view(value, pool) is value
        assert not pool


class TestFreezeOpRelationViews:
    """Direct conversion mechanics on synthetic stores."""

    def test_linear_chain_projects_and_clears(self) -> None:
        core, store = _make_store(
            [
                {"layer_label": "a", "parents": [], "children": ["b"]},
                {
                    "layer_label": "b",
                    "parents": ["a"],
                    "children": ["c"],
                    "parent_arg_positions": {"args": {0: "a"}, "kwargs": {}},
                },
                {
                    "layer_label": "c",
                    "parents": ["b"],
                    "children": [],
                    "parent_arg_positions": {"args": {0: "b"}, "kwargs": {}},
                },
            ]
        )
        stats = freeze_op_relation_views(core, store, _resolver(store).get)
        assert stats.edges == 2
        assert stats.unresolved_labels == 0
        assert stats.explicit_view_cells == 0
        rows = store.rows_building()
        assert rows is not None
        parents_fid = store.layout.fid_by_name["parents"]
        children_fid = store.layout.fid_by_name["children"]
        # Every populated dataflow cell cleared into the CSR.
        assert rows[1][parents_fid] is _CSR
        assert rows[1][children_fid] is _CSR
        assert materialize_dataflow_view(store, 1, "parents") == ("a",)
        assert materialize_dataflow_view(store, 1, "children") == ("c",)
        assert materialize_dataflow_view(store, 0, "children") == ("b",)
        assert materialize_dataflow_view(store, 2, "children") == ()

    def test_edge_occurrences_carry_arg_positions(self) -> None:
        core, store = _make_store(
            [
                {"layer_label": "x", "parents": [], "children": ["y"]},
                {
                    "layer_label": "y",
                    "parents": ["x"],
                    "children": [],
                    "parent_arg_positions": {
                        "args": {0: "x", 1: "x"},
                        "kwargs": {},
                    },
                },
            ]
        )
        freeze_op_relation_views(core, store, _resolver(store).get)
        edges = core.edge_table(DATAFLOW_FAMILY)
        # One occurrence per attributed argument position: x feeds y twice.
        occurrences = [edge for edge in edges.in_edges(1)]
        assert len(occurrences) == 2
        assert {edge.arg_position for edge in occurrences} == {
            ("args", 0),
            ("args", 1),
        }
        # The rematerialized public view still dedups to one label.
        assert materialize_dataflow_view(store, 1, "parents") == ("x",)

    def test_unresolved_label_keeps_explicit_view(self) -> None:
        core, store = _make_store(
            [
                {
                    "layer_label": "a",
                    "parents": ["ghost"],
                    "children": [],
                },
            ]
        )
        stats = freeze_op_relation_views(core, store, _resolver(store).get)
        assert stats.unresolved_labels == 1
        assert stats.explicit_view_cells >= 1
        rows = store.rows_building()
        assert rows is not None
        parents_fid = store.layout.fid_by_name["parents"]
        value = rows[0][parents_fid]
        assert value == ("ghost",)
        assert isinstance(value, tuple)

    def test_non_dataflow_families_convert_in_place(self) -> None:
        core, store = _make_store(
            [
                {
                    "layer_label": "a",
                    "parents": [],
                    "children": [],
                    "modules": ["m1:1", "m2:1"],
                    "input_ancestors": {"a"},
                },
            ]
        )
        freeze_op_relation_views(core, store, _resolver(store).get)
        rows = store.rows_building()
        assert rows is not None
        modules = rows[0][store.layout.fid_by_name["modules"]]
        ancestors = rows[0][store.layout.fid_by_name["input_ancestors"]]
        assert modules == ("m1:1", "m2:1") and isinstance(modules, tuple)
        assert ancestors == frozenset({"a"}) and isinstance(ancestors, frozenset)

    def test_children_order_mismatch_keeps_explicit_view(self) -> None:
        # Staging children in NON-chronological order cannot be reproduced by
        # the row-major edge emission; the differential check must keep the
        # staging order as an explicit view rather than silently reorder.
        core, store = _make_store(
            [
                {"layer_label": "a", "parents": [], "children": ["c", "b"]},
                {"layer_label": "b", "parents": ["a"], "children": []},
                {"layer_label": "c", "parents": ["a"], "children": []},
            ]
        )
        stats = freeze_op_relation_views(core, store, _resolver(store).get)
        rows = store.rows_building()
        assert rows is not None
        children_fid = store.layout.fid_by_name["children"]
        value = rows[0][children_fid]
        assert value == ("c", "b")
        assert isinstance(value, tuple)
        assert stats.explicit_view_cells >= 1


class _ChainCNN(nn.Module):
    """Deterministic conv chain for end-to-end assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.head = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv(x))
        return self.head(y.mean(dim=(2, 3)))


def _capture() -> tl.Trace:
    torch.manual_seed(_SEED)
    return tl.trace(_ChainCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4))


class TestFinishedTraceSurface:
    """End-to-end immutable-view surface on real captures."""

    def test_dataflow_round_trip_matches_capture(self) -> None:
        trace = _capture()
        op = trace.ops["relu_1_2"]
        assert op.parents == ("conv2d_1_1",)
        assert op.children == ("mean_1_3",)
        assert op.parents is op.parents

    def test_csr_backed_cells_rematerialize_lazily(self) -> None:
        trace = _capture()
        store = trace.__dict__["_trace_core"].ops
        assert store.dataflow_edges is not None
        assert len(store.dataflow_edges) > 0
        # Uninspected rows hold the CSR sentinel, not per-row containers.
        rows_or_none = store.rows_building()
        parents_fid = store.layout.fid_by_name["parents"]
        if rows_or_none is not None:
            assert any(cells[parents_fid] is _CSR for cells in rows_or_none)

    def test_freeze_is_idempotent(self) -> None:
        trace = _capture()
        assert freeze_trace_relation_views(trace) is None

    def test_removal_scrubs_immutable_views(self) -> None:
        trace = _capture()
        removed = trace.ops["relu_1_2"]
        surviving_parent = trace.ops["conv2d_1_1"]
        assert "relu_1_2" in surviving_parent.children
        trace._remove_log_entry(removed, remove_references=True)
        assert "relu_1_2" not in surviving_parent.children
        assert isinstance(surviving_parent.children, tuple)

    def test_pickle_round_trip_preserves_views(self) -> None:
        trace = _capture()
        clone = pickle.loads(pickle.dumps(trace))
        op = clone.ops["relu_1_2"]
        assert op.parents == ("conv2d_1_1",)
        assert isinstance(op.parents, tuple)
        assert isinstance(op.input_ancestors, frozenset)

    def test_legacy_list_state_coerces_to_views_on_load(self) -> None:
        trace = _capture()
        op = trace.ops["relu_1_2"]
        state = op.__getstate__()
        # Regress the state to the pre-M6 container types.
        state["parents"] = list(state["parents"])
        state["children"] = list(state["children"])
        state["input_ancestors"] = set(state["input_ancestors"])
        restored = object.__new__(type(op))
        restored.__setstate__(state)
        assert isinstance(restored.parents, tuple)
        assert isinstance(restored.children, tuple)
        assert isinstance(restored.input_ancestors, frozenset)

    def test_layer_aggregate_views(self) -> None:
        trace = _capture()
        layer = trace["relu_1_2"]
        assert isinstance(layer.parents, tuple)
        assert isinstance(layer.children, tuple)
        assert isinstance(layer.modules, tuple)

    def test_staging_phase_stays_mutable_for_detached_building(self) -> None:
        # The building-phase write path must stay raw: a fresh building store
        # accepts and returns the mutable staging containers unchanged.
        layout = _make_layout()
        store = OpRowStore(layout)
        cells: list[object] = [_MISSING] * layout.n_fields
        store.adopt_row(cells)
        staging = ["raw"]
        store.cell_set(0, layout.fid_by_name["parents"], staging)
        assert store.cell_get(0, layout.fid_by_name["parents"]) is staging
