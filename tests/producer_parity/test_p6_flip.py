"""P6 flip: the decomposed producer is the default; legacy is the escape hatch.

The internal ``TORCHLENS_CAPTURE_PRODUCER`` switch inverts at P6 (DoR: default
= decomposed). An unset environment must resolve ``decomposed`` and journal
``OpRecord`` rows end-to-end; ``legacy`` stays available as the explicit
escape hatch until the P7 deletion retires it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.events import OpEvent
from torchlens.ir.op_record import OpRecord

_PRODUCER_ENV = "TORCHLENS_CAPTURE_PRODUCER"

pytestmark = pytest.mark.smoke


def test_default_resolves_decomposed(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens.backends.torch.ops import _resolve_record_producer

    monkeypatch.delenv(_PRODUCER_ENV, raising=False)
    assert _resolve_record_producer() == "decomposed"
    monkeypatch.setenv(_PRODUCER_ENV, "legacy")
    assert _resolve_record_producer() == "legacy"


def test_default_capture_journals_records_and_legacy_hatch_journals_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)

    monkeypatch.delenv(_PRODUCER_ENV, raising=False)
    default_trace = tl.trace(model, x)
    default_events = default_trace._capture_events.op_events
    assert default_events, "vacuous: no journal rows"
    assert all(isinstance(event, OpRecord) for event in default_events), {
        type(event).__name__ for event in default_events
    }

    monkeypatch.setenv(_PRODUCER_ENV, "legacy")
    hatch_trace = tl.trace(model, x)
    hatch_events = hatch_trace._capture_events.op_events
    assert hatch_events, "vacuous: no journal rows"
    assert all(isinstance(event, OpEvent) for event in hatch_events)
