"""Regression tests for r18d report hardening (report/_explain.py, report/_profile.py).

Each test fixes a defect that only surfaces off the plain feed-forward happy path.
Class theme: aggregate multi-pass ``Layer`` handling + ``getattr``-default field drift.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torchlens as tl


class _RecurrentLinear(nn.Module):
    """One Linear applied three times -> a multi-pass (recurrent) layer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


# --------------------------------------------------------------------------- H1
def test_profile_recurrent_call_and_module_levels_no_crash() -> None:
    """profile() at every level survives a recurrent model and counts every pass."""

    log = tl.trace(_RecurrentLinear(), torch.randn(2, 4))

    # All three levels must build without leaking the multi-pass ValueError.
    op_frame = log.profile(level="op").to_pandas()
    call_frame = log.profile(level="call").to_pandas()
    module_frame = log.profile(level="module").to_pandas()

    # op level: 3 linear + 3 relu passes + input + output boundary mirrors.
    assert len(op_frame) == 8

    # The root call aggregates every executed pass (8 ops), not the 4 bare
    # layer labels it stores -- the silent-undercount half of the defect.
    assert call_frame["op_count"].max() == 8

    # The recurrent submodule is invoked three times -> three per-pass ops.
    assert 3 in module_frame["op_count"].tolist()

    # Real per-pass metrics were summed, not skipped.
    assert call_frame["flops"].notna().any()
    assert call_frame["time"].notna().any()
