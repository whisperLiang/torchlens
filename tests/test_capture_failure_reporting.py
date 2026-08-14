"""Failure-arm reporting: routed warning, partial_log note, empty_cache gate.

Pins the B8-35/B8-44/B8-45/R16-4b fixes in ``backends/torch/backend.py``:

- The historical unconditional stdout banner ("Feature extraction failed;
  returning model and environment to normal") was factually false on
  rescue-recovered captures and supported return_partial flows, corrupted
  machine-readable stdout, and was unfilterable. It is now ONE accurate
  routed ``RuntimeWarning`` naming the failure and the recovery surface.
- The SUCCESS path of partial-log attachment adds an ``add_note`` telling the
  user ``exc.partial_log`` exists (previously only the two attachment-FAILURE
  arms carried notes).
- ``cleanup_forward_memory`` clears the CUDA allocator cache only when the
  capture actually touched CUDA (the capture-touched-CUDA predicate), never
  unconditionally.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _BoomError(RuntimeError):
    pass


class _FailingModel(nn.Module):
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        raise _BoomError("forward exploded")


def test_failed_capture_emits_routed_warning_not_stdout_banner(capsys):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(_BoomError):
            tl.trace(_FailingModel(), torch.randn(1, 2))
    out = capsys.readouterr().out
    assert "Feature extraction failed" not in out, "stdout banner must be gone"
    routed = [
        record
        for record in records
        if record.category is RuntimeWarning and "capture attempt failed" in str(record.message)
    ]
    assert len(routed) == 1, [str(record.message) for record in records]
    assert "_BoomError" in str(routed[0].message)
    assert "partial_log" in str(routed[0].message)


@pytest.mark.skipif(
    not hasattr(BaseException, "add_note"),
    reason="BaseException.add_note requires Python 3.11+",
)
def test_failed_capture_success_path_notes_partial_log():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_BoomError) as exc_info:
            tl.trace(_FailingModel(), torch.randn(1, 2))
    exc = exc_info.value
    assert getattr(exc, "partial_log", None) is not None
    notes = getattr(exc, "__notes__", [])
    assert any("partial_log" in note for note in notes), notes


def test_cleanup_forward_memory_gated_on_capture_touched_cuda(monkeypatch):
    from torchlens.backends.torch import backend as backend_module
    from torchlens.backends.torch.backend import TorchBackend

    calls: list[str] = []
    monkeypatch.setattr(backend_module, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))

    cpu_trace = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    TorchBackend().cleanup_forward_memory(cpu_trace)
    assert calls == [], "a CPU-only capture must never clear the CUDA allocator cache"
