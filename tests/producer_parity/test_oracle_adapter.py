"""Inverse-adapter identity proof: inserting the adapter changes nothing."""

from __future__ import annotations

import pytest

import torchlens as tl

from ._models import SmallCNN, _cnn_input
from ._oracle_adapter import op_event_from_record

pytestmark = pytest.mark.smoke


def test_adapter_is_identity_on_op_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every LEGACY journal record passes through the adapter unchanged."""

    import torchlens.postprocess as postprocess_module
    import torchlens.postprocess._materialize as materialize_module

    # This test pins the compat-OpEvent identity leg; force the legacy
    # producer regardless of the ambient switch.
    monkeypatch.setenv("TORCHLENS_CAPTURE_PRODUCER", "legacy")

    captured: list = []
    original = materialize_module.materialize_from_events

    def spy(trace, events):
        captured.extend(events.op_events)
        original(trace, events)

    monkeypatch.setattr(postprocess_module, "materialize_from_events", spy)
    monkeypatch.setattr(materialize_module, "materialize_from_events", spy)
    tl.trace(SmallCNN(), _cnn_input())

    assert captured
    for event in captured:
        assert op_event_from_record(event) is event
