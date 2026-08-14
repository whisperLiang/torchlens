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

    This is the half that survives on LOADED artifacts, where the session-only
    capture witness is gone (``FieldPolicy.DROP``): the dropped parent's saved
    argument value sits unattributed in the child's saved args, and the
    Round-26 inverse orphan-arg sweep / dropped-edge witness must name it.
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
