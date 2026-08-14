"""Oracle-independence probes: catch corruption the tautological pairs miss.

b9 R74/75-3: a large metadata-invariant class re-derives its expectation from
the SAME canonical CSR edge table the checked fields come from (both sides of
the ancestry recompute read ``_trace_core/relation_views.py``), so a corrupt
edge table can yield a CONSISTENT pair and sail through that tier. The
genuinely independent roots are (a) on LIVE traces, the sealed capture-time
edge witness (``capture_edge_survival``), and (b) everywhere, the replay-side
sweeps that start from SAVED VALUES instead of recorded edges (the Round-26
inverse orphan-arg sweep and the dropped-edge capture-identity witness).

This file plants the exact corruption shape -- a SYMMETRIC edge drop,
scrubbed from ``parents``, the parent's ``children``, AND
``parent_arg_positions`` together so no recorded-edge self-consistency pair
can trip -- and pins that BOTH independent roots go red on it.

RAW-JOURNAL RE-DERIVATION (design note, b9 R74/75-3 second half). The
structural fix direction for the tautological class: where a raw capture
journal exists (``trace.capture_events``, session-time), invariant
expectations for ancestry/topology should be re-derivable from the JOURNAL's
parent records rather than from the sealed edge table, giving the metadata
tier a second root on live traces that also survives the edge-table freeze.
On LOADED artifacts the journal is gone (``FieldPolicy.DROP``), so the loaded
metadata tier legitimately collapses to self-consistency -- the honest
boundary is: live = journal + edge table + capture witness, loaded =
self-consistency plus the value-rooted replay sweeps pinned below. Building
the journal-side re-derivation is capture/postprocess territory
(FW2-CAPTURE); this file pins the halves that exist today.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens.validation import check_metadata_invariants
from torchlens.validation.diagnostics import TRACE_FAILURE_ATTR

pytestmark = pytest.mark.smoke


class _TwoStage(nn.Module):
    """Two-stage model whose mid-edge is the drop target."""

    def __init__(self) -> None:
        """Build the two linear stages."""

        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear -> relu -> linear.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Logits.
        """

        return self.fc2(torch.relu(self.fc1(x)))


def _drop_edge_symmetrically(log, child_label: str, parent_label: str) -> None:
    """Remove one parent edge from every recorded relation surface at once.

    Parameters
    ----------
    log:
        Finished trace to corrupt.
    child_label:
        Label of the edge's consumer.
    parent_label:
        Label of the edge's producer.
    """

    child = log.layer_dict_all_keys[child_label]
    parent = log.layer_dict_all_keys[parent_label]
    base_parent = parent_label.split(":", 1)[0]
    base_child = child_label.split(":", 1)[0]
    child.parents = [p for p in child.parents if p.split(":", 1)[0] != base_parent]
    parent.children = [c for c in parent.children if c.split(":", 1)[0] != base_child]
    # A stealthy corruption conforms the derived flag too, or the cheap
    # graph_topology pair (children vs has_children) catches it before any
    # independence question arises.
    parent.has_children = len(parent.children) > 0
    for positions in (child.parent_arg_positions or {}).values():
        stale = [
            argloc
            for argloc, owner in positions.items()
            if isinstance(owner, str) and owner.split(":", 1)[0] == base_parent
        ]
        for argloc in stale:
            positions.pop(argloc, None)
    # Since the multiplicity witness (b9-opus R75-1) the canonical CSR
    # edge-occurrence table is a checked surface too: a stealthy symmetric
    # drop must scrub the occurrence there as well, or the cheap
    # count-consistency pair catches it before any independence question
    # arises. capture_edge_survival roots in the sealed CAPTURE-TIME
    # witness, which no post-hoc scrub can conform.
    core = log.__dict__.get("_trace_core")
    store = getattr(core, "ops", None) if core is not None else None
    edges = getattr(store, "dataflow_edges", None) if store is not None else None
    label_rows = getattr(core, "label_rows", {}) if core is not None else {}
    source_row = label_rows.get(base_parent)
    target_row = label_rows.get(base_child)
    if edges is not None and source_row is not None and target_row is not None:
        kept = [
            edges.edge(edge_id)
            for edge_id in range(len(edges))
            if not (
                edges.edge(edge_id).source == source_row
                and edges.edge(edge_id).target == target_row
            )
        ]
        edges._frozen = False
        edges._sources = [edge.source for edge in kept]
        edges._targets = [edge.target for edge in kept]
        edges._use_kinds = [edge.use_kind for edge in kept]
        edges._arg_positions = [edge.arg_position for edge in kept]
        edges._seqs = [edge.seq for edge in kept]
        edges._by_source = None
        edges._by_target = None
        edges.freeze(len(store), len(store))


def _planted_edge_drop_trace():
    """Return a finished trace with one mid-graph edge symmetrically dropped.

    Returns
    -------
    tuple
        ``(trace, ground_truth_outputs)``.
    """

    log = trace_fn(_TwoStage(), torch.randn(2, 6), layers_to_save="all", save_arg_values=True)
    outputs = [log.layer_dict_all_keys[label].out for label in log.output_layers]
    relu_label = next(op.label for op in log.compute_ops if op.func_name == "relu")
    consumer = next(iter(log.layer_dict_all_keys[relu_label].children))
    _drop_edge_symmetrically(log, consumer, relu_label)
    return log, outputs


class _DoubleConsume(nn.Module):
    """Model with GENUINE edge multiplicity: one tensor consumed twice."""

    def __init__(self) -> None:
        """Build the single linear stage."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the linear output at BOTH arg positions of one add.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            ``h + h`` for ``h = lin(x)`` -- two occurrences of one edge.
        """

        h = self.lin(x)
        return h + h


def _duplicate_edge_occurrence(log, child_label: str):
    """Append a copy of one existing CSR edge occurrence and re-freeze.

    This is the b9-opus R75-1 plant shape: parents/children views dedup by
    label and ``parent_arg_positions`` is per-position storage, so a
    DUPLICATED occurrence changes no label view and no arg map -- only the
    canonical edge-occurrence table itself carries the extra multiplicity.

    Parameters
    ----------
    log:
        Finished trace to corrupt.
    child_label:
        Label of the consumer whose first in-edge is duplicated.

    Returns
    -------
    Edge
        The duplicated occurrence (for diagnostics).
    """

    core = log.__dict__["_trace_core"]
    store = core.ops
    edges = store.dataflow_edges
    row = core.label_rows[child_label]
    edge = next(iter(edges.in_edges(row)))
    edges._frozen = False
    edges._by_source = None
    edges._by_target = None
    edges.add(edge.source, edge.target, use_kind=edge.use_kind, arg_position=edge.arg_position)
    edges.freeze(len(store), len(store))
    return edge


def test_duplicated_edge_occurrence_is_caught_on_a_live_trace():
    """A DUPLICATED edge occurrence in the CSR table must fail invariants.

    b9-opus R75-1: retargeting and novel (src, tgt) adds were caught, but
    duplicating an existing occurrence was 0/6 silent -- the label views
    dedup and the arg map is per-position, so no recorded pair disagreed.
    The multiplicity witness cross-checks per-(child, parent) occurrence
    counts against the independently stored ``parents`` +
    ``parent_arg_positions`` roots and must name the child and parent.
    """

    log = trace_fn(_TwoStage(), torch.randn(2, 6))
    try:
        check_metadata_invariants(log)  # pristine trace is green
        relu_label = next(op.label for op in log.compute_ops if op.func_name == "relu")
        consumer = next(iter(log.layer_dict_all_keys[relu_label].children)).split(":", 1)[0]
        _duplicate_edge_occurrence(log, consumer)
        with pytest.raises(Exception, match="multiplicity") as excinfo:
            check_metadata_invariants(log)
        message = str(excinfo.value)
        assert "edge_use_parent_arg_consistency" in message
        assert consumer in message, "failure must name the corrupted child"
        assert relu_label.split(":", 1)[0] in message, "failure must name the parent"
    finally:
        log.cleanup()


def test_genuine_double_consumption_passes_multiplicity_witness():
    """Legitimate parallel edges (``h + h``) must NOT trip the witness.

    A genuine double consumption records TWO occurrences of one
    (parent, child) edge, matched by two ``parent_arg_positions`` entries.
    The witness compares COUNTS across independent roots, so honest
    multiplicity stays green -- pin that, and pin that the trace really
    carries the parallel edges (the positive half is not vacuous).
    """

    log = trace_fn(_DoubleConsume(), torch.randn(2, 4))
    try:
        core = log.__dict__["_trace_core"]
        edges = core.ops.dataflow_edges
        add_label = next(op.label for op in log.compute_ops if op.func_name in ("add", "__add__"))
        add_row = core.label_rows[add_label.split(":", 1)[0]]
        sources = [edge.source for edge in edges.in_edges(add_row)]
        # The freeze emits one occurrence per parents-list entry per
        # attributed arg position; a double consumption whose staging parents
        # list also repeats the label records 2x2 = 4 parallel occurrences.
        # The witness only requires count CONSISTENCY, so any >= 2 parallel
        # multiplicity proves the positive fixture is not vacuous.
        assert len(sources) >= 2 and len(set(sources)) == 1, (
            "expected the add op to carry parallel occurrences of ONE parent "
            f"edge; got sources {sources!r} -- the positive fixture went vacuous"
        )
        check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_symmetric_edge_drop_is_caught_on_a_live_trace():
    """A both-sides edge drop is caught by an INDEPENDENT root, live.

    The plant is the post-witness silent edge-drop class: parents, children,
    and the arg map are scrubbed TOGETHER, so every pair that re-derives one
    recorded-edge surface from another sees a coherent (weaker) graph. On a
    LIVE trace the sealed capture-time edge witness (``capture_edge_survival``,
    r29 F3b) must reconcile recorded edges against what capture witnessed --
    pin that it names the plant.
    """

    log, _ = _planted_edge_drop_trace()
    try:
        with pytest.raises(Exception, match="capture_edge_survival"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_symmetric_edge_drop_is_caught_by_value_rooted_replay():
    """The replay pipeline also goes red on the same plant, from saved VALUES.

    Scope narrowed (b9-sol R75-2): this proves the value-rooted half on a
    payload-bearing LIVE trace only. It does NOT reach loaded artifacts —
    ``Op.func`` is ``FieldPolicy.DROP`` at every save level, so loaded torch
    replay validation refuses typed before any sweep runs (pinned below by
    ``test_loaded_artifact_replay_validation_refuses_typed``), and the
    session-only capture witness is also gone on load. The loaded half of
    edge-drop detection is therefore DARK today; that gap is pinned honestly
    by the strict-xfail
    ``test_symmetric_edge_drop_on_a_loaded_artifact_known_gap``.
    Metadata invariants are disabled here to isolate the value-rooted path.
    """

    log, outputs = _planted_edge_drop_trace()
    try:
        status = log.validate_saved_outs(outputs, validate_metadata=False)
        assert not status, (
            "a symmetric edge drop validated clean through the value-rooted "
            "sweeps: the loaded-artifact half of edge-drop detection is dark"
        )
        failure = getattr(log, TRACE_FAILURE_ATTR, None)
        assert failure is not None, "validation failed without recording a failure"
    finally:
        log.cleanup()


def _loaded_edge_drop_trace(tmp_path):
    """Return a LOADED trace with the plant applied post-load, plus outputs.

    Builds the same _TwoStage capture as ``_planted_edge_drop_trace``, round
    trips it through a real ``Trace.save`` / ``tl.load``, then applies the
    symmetric edge drop to the LOADED trace (relation views normalize the
    assignments; the CSR scrub is a no-op on the detached loaded core).

    Parameters
    ----------
    tmp_path:
        Directory for the bundle.

    Returns
    -------
    tuple
        ``(loaded_trace, ground_truth_outputs)``.
    """

    import torchlens as tl

    log = trace_fn(_TwoStage(), torch.randn(2, 6), layers_to_save="all", save_arg_values=True)
    outputs = [log.layer_dict_all_keys[label].out for label in log.output_layers]
    path = tmp_path / "edge_drop_plant.tlspec"
    log.save(path)
    log.cleanup()
    loaded = tl.load(path)
    relu_label = next(op.label for op in loaded.compute_ops if op.func_name == "relu")
    consumer = next(iter(loaded.layer_dict_all_keys[relu_label].children))
    _drop_edge_symmetrically(loaded, consumer, relu_label)
    return loaded, outputs


def test_loaded_artifact_replay_validation_refuses_typed(tmp_path) -> None:
    """Loaded torch replay validation refuses TYPED, never silently passes.

    b9-sol R75-2 reality check: ``Op.func`` is ``FieldPolicy.DROP`` at every
    save level, so a loaded torch artifact cannot run the value-rooted replay
    sweeps at all. The honest floor pinned here is that the refusal is LOUD —
    a ``TorchLensIOError`` naming the unresolved computational functions — so
    a planted loaded corruption can never validate clean through a path that
    silently skipped the work.
    """

    from torchlens._io import TorchLensIOError

    loaded, outputs = _loaded_edge_drop_trace(tmp_path)
    with pytest.raises(TorchLensIOError, match="unresolved computational functions"):
        loaded.validate_forward_pass(outputs, validate_metadata=False)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "b9-sol R75-2 / b9-fable R75-1 known gap: the loaded half of "
        "edge-drop detection is DARK. Op.func is FieldPolicy.DROP at every "
        "save level so loaded torch replay refuses typed before any "
        "value-rooted sweep runs, and the capture-time edge witness "
        "(capture_edge_survival) is session-only so the loaded metadata tier "
        "cannot see the plant either. This test asserts the DESIRED loaded "
        "behavior; strict xfail flips red the day a loaded-capable replay or "
        "witness lands, forcing the pin to become a real regression test."
    ),
)
def test_symmetric_edge_drop_on_a_loaded_artifact_known_gap(tmp_path) -> None:
    """DESIRED: a loaded artifact's validation path catches the same plant."""

    loaded, outputs = _loaded_edge_drop_trace(tmp_path)
    status = loaded.validate_forward_pass(outputs, validate_metadata=False)
    assert not status, (
        "a symmetric edge drop on a LOADED artifact validated clean through "
        "the loaded-provider validation path"
    )
    failure = getattr(loaded, TRACE_FAILURE_ATTR, None)
    assert failure is not None, "validation failed without recording a failure"
