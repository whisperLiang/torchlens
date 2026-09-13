"""Repeated tinygrad operations retain their identities through portable I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_tinygrad_recurrence_grouping import _repeated_trace

import torchlens as tl
from torchlens.validation import check_metadata_invariants

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.smoke]


@pytest.mark.parametrize("level,lazy", [("portable", False), ("portable", True), ("audit", False)])
def test_recurrent_trace_roundtrip_preserves_passes(tmp_path: Path, level: str, lazy: bool) -> None:
    """Save/load preserves distinct passes, graph relations, and payload policy.

    Parameters
    ----------
    tmp_path:
        Private directory for the original and re-saved bundles.
    level:
        Whether the artifact retains payloads or only audit metadata.
    lazy:
        Whether payloads are materialized only on explicit access.
    """

    trace = _repeated_trace()
    assert check_metadata_invariants(trace) is True
    grouped = [op for op in trace.layer_list if op.func_name == "where" and op.num_passes > 1]
    assert [op.pass_index for op in grouped] == [1, 2]
    expected = {op.label: op.out.numpy().copy() for op in grouped}
    relations = {
        op.label: (op.layer_label, op.pass_index, tuple(op.parents), tuple(op.children))
        for op in trace.layer_list
    }
    equivalence_members = {frozenset(labels) for labels in trace.op_equivalence_classes.values()}
    path = tmp_path / "recurrent.tlspec"
    tl.save(trace, path, level=level)
    loaded = tl.load(path, lazy=lazy)

    assert loaded.output_layers == trace.output_layers == [grouped[-1].label]
    assert loaded.output_ops[0] is loaded[grouped[-1].label]
    assert {
        op.label: (op.layer_label, op.pass_index, tuple(op.parents), tuple(op.children))
        for op in loaded.layer_list
    } == relations
    assert {frozenset(labels) for labels in loaded.op_equivalence_classes.values()} == (
        equivalence_members
    )
    assert loaded.validation_replay_status.reason == "loaded_trace_runtime_capture_stripped"
    for label, value in expected.items():
        op = loaded[label]
        if level == "audit":
            assert op.out is None and op.out_ref is None
        elif lazy:
            assert op.out is None and op.out_ref is not None
            np.testing.assert_array_equal(op.out_ref.materialize().numpy(), value)
        else:
            np.testing.assert_array_equal(op.out.numpy(), value)

    # Saving must leave the source's live replay sidecars and payloads intact.
    assert trace.validate_forward_pass([]) is True
    resaved_path = tmp_path / "resaved.tlspec"
    tl.save(loaded, resaved_path, level="audit")
    resaved = tl.load(resaved_path)
    assert resaved.output_layers == trace.output_layers
    assert resaved.op_labels == trace.op_labels
