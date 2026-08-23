"""Shared-root corruption plant for the capture-oracle characterizer.

b9-sol R75-1: ``_characterize._snapshot_events`` derives ground-truth events
from the captured journal THROUGH the shared inverse oracle adapter
(``producer_parity._oracle_adapter.op_event_from_record``) — the same root the
production ingest migration relies on — so a common-mode adapter defect would
populate BOTH sides of a naive self-comparison. The existing non-vacuity test
only mutates an already-built golden dict; it never proves a live adapter
defect surfaces.

This test plants the defect at the shared root itself: it monkeypatches the
adapter to silently DROP one parent edge during a real in-process
characterization run, then asserts (a) the characterization output diverges
from the committed golden FILE (the golden catches adapter drift), and (b) an
independent first-principles root breaks (PlainCNN's relu must consume exactly
the conv output). ``_characterize.py`` production behavior is untouched.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.heavy

_GOLDEN = Path(__file__).with_name("goldens") / "plain_cnn__exhaustive.json"
_CASE = "plain_cnn__exhaustive"


def _parent_sequences(events: list[dict[str, Any]]) -> list[list[str]]:
    """Project each event's raw parent-label list.

    Parameters
    ----------
    events:
        Characterization ``ground_truth.events`` rows.

    Returns
    -------
    list[list[str]]
        Per-event ``identity.parent_labels_raw`` lists.
    """

    return [list(event["identity"]["parent_labels_raw"]) for event in events]


def test_adapter_corruption_diverges_from_golden_and_breaks_independent_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent-dropping adapter defect is caught by golden AND hand pins."""

    import producer_parity._oracle_adapter as adapter_module

    from ._characterize import characterize_case

    real_adapter = adapter_module.op_event_from_record
    corrupted_calls: list[str] = []

    def parent_dropping_adapter(record: Any, *, grad_fn_handle: Any = None) -> Any:
        """Delegate to the real adapter, then silently drop one parent edge."""

        event = real_adapter(record, grad_fn_handle=grad_fn_handle)
        if event.parents:
            corrupted_calls.append(event.label_raw)
            event = dataclasses.replace(event, parents=event.parents[:-1])
        return event

    monkeypatch.setattr(adapter_module, "op_event_from_record", parent_dropping_adapter)
    corrupted_record = characterize_case(_CASE)

    assert corrupted_calls, "the plant never fired -- the corruption run is vacuous"

    golden_events = json.loads(_GOLDEN.read_text(encoding="utf-8"))["record"]["ground_truth"][
        "events"
    ]
    golden_parents = _parent_sequences(golden_events)
    assert any(golden_parents), "golden has no parent edges -- divergence check is vacuous"

    corrupted_events = corrupted_record["ground_truth"]["events"]
    corrupted_parents = _parent_sequences(corrupted_events)

    # (a) The committed golden catches the adapter drift: the ground-truth
    # event projection no longer matches the golden file's parent structure,
    # and specifically MISSES edges the golden records (a drop, not a rename).
    assert corrupted_parents != golden_parents, (
        "a parent-dropping adapter defect produced golden-identical events; "
        "the golden cannot catch shared-adapter drift"
    )
    golden_edge_count = sum(len(parents) for parents in golden_parents)
    corrupted_edge_count = sum(len(parents) for parents in corrupted_parents)
    assert corrupted_edge_count < golden_edge_count

    # (b) The independent first-principles root breaks: PlainCNN.forward is
    # exactly relu(conv(x)), so the relu event must consume exactly the conv
    # event's output. Under the plant it no longer does.
    by_func = {event["identity"]["func_name"]: event for event in corrupted_events}
    assert set(by_func) == {"conv2d", "relu"}
    conv_label = by_func["conv2d"]["identity"]["label_raw"]
    assert by_func["relu"]["identity"]["parent_labels_raw"] != [conv_label], (
        "the hand-derived relu<-conv fact survived the planted adapter defect"
    )
