"""P3 gates: the decomposed producer behind TORCHLENS_CAPTURE_PRODUCER.

* Cross-leg parity: legacy vs decomposed runs of the same scenario compare
  EMPTY through Check A (structural; bijective relabelings green by
  construction) with Check B clean within each leg — journal, store, and
  artifact layers.
* ``_raw_event_shape_hash`` is byte-equal across legs (same capture topology
  through either record shape).
* Journal shape conformance: the decomposed leg emits ``OpRecord`` for every
  entry (sources and buffers included) with seq stamped on the core; the
  legacy leg emits only compat ``OpEvent``s.
* Stage-applicability conformance (DoR 3.1): the sparse pipeline NEVER
  writes ``_capture_parent_edge_truth`` on either leg; freeze/append are the
  only universal tail stages in the checked-in matrix.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.events import OpEvent
from torchlens.ir.op_record import OpRecord

from ._comparator import check_a, check_b
from ._models import scenario_by_name
from ._snapshot import run_scenario

pytestmark = pytest.mark.smoke

_PRODUCER_ENV = "TORCHLENS_CAPTURE_PRODUCER"

# Cross-leg scenario slice: exhaustive + reference-mode + predicate/fastlog +
# buffers + manual-edge cover every commit pipeline and the ingest joins.
_CROSS_LEG_SCENARIOS = (
    "cnn_exhaustive",
    "cnn_reference",
    "cnn_predicate",
    "cnn_record",
    "buffer_output_exhaustive",
    "manual_edge_exhaustive",
)


def _run_leg(
    monkeypatch: pytest.MonkeyPatch, scenario_name: str, producer: str, tmp: Path
) -> Any:
    monkeypatch.setenv(_PRODUCER_ENV, producer)
    try:
        return run_scenario(scenario_by_name(scenario_name), tmp)
    finally:
        monkeypatch.delenv(_PRODUCER_ENV, raising=False)


@pytest.mark.heavy
def test_cross_leg_parity_two_check_discharge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy vs decomposed legs: Check A empty cross-leg, Check B clean per leg."""

    for scenario_name in _CROSS_LEG_SCENARIOS:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = _run_leg(monkeypatch, scenario_name, "legacy", Path(tmp) / "legacy")
            decomposed = _run_leg(
                monkeypatch, scenario_name, "decomposed", Path(tmp) / "decomposed"
            )
        a_diffs = check_a(legacy.snapshot, decomposed.snapshot)
        assert not a_diffs, (
            f"{scenario_name}: cross-leg structural diffs (first 5): {a_diffs[:5]}"
        )
        for leg_name, run in (("legacy", legacy), ("decomposed", decomposed)):
            b_diffs = check_b(run)
            assert not b_diffs, (
                f"{scenario_name}:{leg_name}: within-leg attestation diffs: {b_diffs[:5]}"
            )
        legacy_hash = getattr(legacy.trace, "_raw_event_shape_hash", None)
        decomposed_hash = getattr(decomposed.trace, "_raw_event_shape_hash", None)
        assert legacy_hash is not None
        assert legacy_hash == decomposed_hash, (
            f"{scenario_name}: _raw_event_shape_hash differs across legs"
        )


def test_decomposed_journal_holds_records_with_core_seq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decomposed leg journals OpRecords everywhere; legacy stays OpEvent."""

    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
    x = torch.randn(2, 4)

    monkeypatch.setenv(_PRODUCER_ENV, "decomposed")
    decomposed_trace = tl.trace(model, x)
    monkeypatch.setenv(_PRODUCER_ENV, "legacy")
    legacy_trace = tl.trace(model, x)

    decomposed_events = decomposed_trace._capture_events.op_events
    assert decomposed_events, "vacuous: no journal rows"
    assert all(isinstance(event, OpRecord) for event in decomposed_events), {
        type(event).__name__ for event in decomposed_events
    }
    seqs = [event.core.seq for event in decomposed_events]
    assert all(seq > 0 for seq in seqs), "unstamped record reached the journal"
    assert seqs == sorted(seqs)
    # grad-fn single ownership: no record carries a handle attribute at all
    assert all(
        getattr(event, "grad_fn_handle", None) is None for event in decomposed_events
    )
    legacy_events = legacy_trace._capture_events.op_events
    assert all(isinstance(event, OpEvent) for event in legacy_events)


def test_unknown_producer_value_refuses() -> None:
    from torchlens.backends.torch.ops import _resolve_record_producer

    import os

    os.environ[_PRODUCER_ENV] = "not_a_producer"
    try:
        with pytest.raises(ValueError, match="not a known capture producer"):
            _resolve_record_producer()
    finally:
        del os.environ[_PRODUCER_ENV]


@pytest.mark.parametrize("producer", ["legacy", "decomposed"])
def test_sparse_pipeline_never_writes_edge_truth(
    monkeypatch: pytest.MonkeyPatch, producer: str
) -> None:
    """Conformance (DoR 3.1): edge-truth seal is an exhaustive-only stage."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)
    monkeypatch.setenv(_PRODUCER_ENV, producer)

    recording = tl.record(model, x, save=tl.func("relu"))
    truth = getattr(recording, "_capture_parent_edge_truth", None)
    assert not truth, f"sparse pipeline wrote edge truth under {producer}"

    exhaustive_trace = tl.trace(model, x)
    assert exhaustive_trace.__dict__.get("_capture_parent_edge_truth"), (
        "exhaustive pipeline must seal edge truth"
    )


def test_commit_stage_matrix_shape() -> None:
    from torchlens.capture.projections import COMMIT_STAGE_MATRIX

    exhaustive = COMMIT_STAGE_MATRIX["exhaustive"]
    sparse = COMMIT_STAGE_MATRIX["sparse"]
    # the ONE universal tail: freeze -> append, in order, in both pipelines
    for stages in (exhaustive, sparse):
        assert stages.index("freeze") + 1 == stages.index("append")
    for exhaustive_only in ("edge_truth_seal", "grad_handle_index", "live_op_view"):
        assert exhaustive_only in exhaustive
        assert exhaustive_only not in sparse
    for sparse_only in ("demanded_enrichment", "halt_evaluation"):
        assert sparse_only in sparse
        assert sparse_only not in exhaustive
