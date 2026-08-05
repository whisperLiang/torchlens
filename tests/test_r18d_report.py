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


class _SelectiveTinyModel(nn.Module):
    """Add + relu + sigmoid; boundary/input ops go unsaved under a relu-only save."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.relu(x + 1.0))


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


# --------------------------------------------------------------------------- M2
def test_profile_device_column_is_populated() -> None:
    """The device column reports the real device (device_ref), never a blanket None."""

    frame = tl.trace(nn.Linear(4, 4), torch.randn(2, 4)).profile().to_pandas()
    assert "device" in frame.columns
    assert frame["device"].notna().all()
    assert set(frame["device"]) == {"cpu"}


# --------------------------------------------------------------------------- M1
def test_explain_selective_save_reports_without_crashing() -> None:
    """explain() honors its unknown/clean contract on a selective-save trace."""

    log = tl.trace(_SelectiveTinyModel(), torch.randn(2, 4), save=tl.func("relu"))

    saved = [
        layer.layer_label
        for layer in log.layer_list
        if getattr(layer, "has_saved_activation", False)
    ]
    unsaved = [
        layer.layer_label
        for layer in log.layer_list
        if not getattr(layer, "has_saved_activation", False)
    ]
    # Precondition: the input boundary is genuinely unsaved (the crash trigger).
    assert unsaved and saved

    report_json = tl.report.explain(log, format="json")
    assert isinstance(report_json, dict)
    assert isinstance(report_json["first_nonfinite"], str)

    report_text = tl.report.explain(log, format="text")
    assert isinstance(report_text, str)
    assert "Shared parameters" in report_text


def test_explain_selective_save_still_flags_saved_nonfinite() -> None:
    """Gating on saved payloads must not blind the anomaly scan to a saved NaN."""

    x = torch.randn(2, 4)
    x[0, 0] = float("nan")
    log = tl.trace(_SelectiveTinyModel(), x, save=tl.func("relu"))

    report_json = tl.report.explain(log, format="json")
    # relu(nan) == nan and relu IS saved, so the scan must still detect it.
    assert "non-finite" in report_json["first_nonfinite"].lower()
