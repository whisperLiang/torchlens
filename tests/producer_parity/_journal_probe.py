"""Shared journal-grabbing probe for the P4 test files.

Intercepts step-0 materialization to hand tests the REAL journal entries a
tiny capture produced (per producer leg), without depending on postprocess
internals beyond the one compat seam every path routes through.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

import torchlens as tl

_PRODUCER_ENV = "TORCHLENS_CAPTURE_PRODUCER"
_SEED = 20260812


class ProbeCNN(nn.Module):
    """Conv + relu + fc: enough op/facet variety for template events."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 60)
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        return self.fc(h.flatten(1))


def probe_input() -> torch.Tensor:
    torch.manual_seed(_SEED + 61)
    return torch.randn(2, 1, 4, 4)


def grab_journal_events(monkeypatch: Any, producer: str) -> list[Any]:
    """Return raw journal entries from a tiny capture at the step-0 seam."""

    monkeypatch.setenv(_PRODUCER_ENV, producer)
    postprocess_module = importlib.import_module("torchlens.postprocess")
    materialize_module = importlib.import_module("torchlens.postprocess._materialize")
    original = materialize_module.materialize_from_events
    grabbed: list[Any] = []

    def observing(trace: Any, events: Any) -> None:
        if not grabbed:
            grabbed.extend(events.op_events)
        original(trace, events)

    monkeypatch.setattr(postprocess_module, "materialize_from_events", observing)
    monkeypatch.setattr(materialize_module, "materialize_from_events", observing)
    tl.trace(ProbeCNN(), probe_input())
    monkeypatch.delenv(_PRODUCER_ENV, raising=False)
    assert grabbed, "journal interception grabbed no events"
    return grabbed
