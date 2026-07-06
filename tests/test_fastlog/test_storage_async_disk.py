"""CPU-async-compatible disk-mode tests for fastlog."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import CaptureSpec
from torchlens.options import StreamingOptions


class AsyncDiskModel(nn.Module):
    """Small model for CPU-async storage compatibility."""

    def __init__(self) -> None:
        """Initialize the layer."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.linear(x).relu()


def test_cpu_async_disk_storage_uses_sync_finalized_bundle(tmp_path: Path) -> None:
    """``save_mode='cpu_async'`` is accepted and finalizes disk-only bundles."""

    bundle_path = tmp_path / "cpu_async.tlfast"

    recording = tl.fastlog.record(
        AsyncDiskModel(),
        torch.ones(1, 3),
        default_op=CaptureSpec(save_mode="cpu_async"),
        streaming=StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )
    loaded = tl.fastlog.load(bundle_path)

    assert recording.bundle_path == bundle_path
    assert (bundle_path / "manifest.json").exists()
    assert (bundle_path / "fastlog_index.jsonl").exists()
    assert not list(tmp_path.glob("cpu_async.tlfast.tmp.*"))
    assert len(loaded) == len(recording)
    assert any(record.ram_payload is None and record.disk_payload is not None for record in recording)
    assert all(record.spec.save_mode == "cpu_async" for record in recording)
