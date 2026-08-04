"""Tests for the TorchLens doctor utility."""

from __future__ import annotations

import importlib.metadata

from packaging.requirements import Requirement
import pytest

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
from torchlens.utils import _DOCTOR_EXCLUDED_EXTRAS
from torchlens.utils._torch_compat import get_torch_capability_snapshot


def test_doctor_returns_sane_report() -> None:
    """Doctor returns structured checks and a printable report."""

    report = tl.utils.doctor()
    assert report.checks
    names = {check.name for check in report.checks}
    assert {
        "pytorch",
        "runtime capabilities",
        "torch wrapper bindings",
        "cuda",
        "graphviz",
        "safetensors",
        "extras",
        "model fingerprint",
    } <= names
    assert all(check.status in {"PASS", "FAIL", "SKIP", "WARN"} for check in report.checks)
    text = report.show()
    assert "TorchLens doctor report" in text
    assert "pytorch" in text


def test_doctor_surfaces_every_runtime_capability() -> None:
    """Doctor capability row stays in lockstep with defined capability flags."""

    report = tl.utils.doctor()
    row = next(check for check in report.checks if check.name == "runtime capabilities")
    expected = set(get_torch_capability_snapshot()) | set(get_tf_capability_snapshot())
    surfaced = {part.split("=", 1)[0] for part in row.detail.split(";")[0].split(", ")}

    assert surfaced == expected


def test_doctor_warns_on_stale_torch_wrapper_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doctor reports torch namespace attrs that point at original callables."""

    wrap_torch()
    original_relu = _state._decorated_to_orig[id(__import__("torch").relu)]
    monkeypatch.setattr("torch.relu", original_relu)

    report = tl.utils.doctor()
    row = next(check for check in report.checks if check.name == "torch wrapper bindings")

    assert row.status == "WARN"
    assert "torch.relu" in row.detail


def test_declared_extra_probes_track_packaging_metadata() -> None:
    """Doctor extra probes must stay in lockstep with declared package extras."""

    distribution = importlib.metadata.distribution("torchlens")
    expected = {
        extra
        for extra in (distribution.metadata.get_all("Provides-Extra") or [])
        if extra not in _DOCTOR_EXCLUDED_EXTRAS
    }
    requirement_extras = set()
    for requirement_line in distribution.requires or ():
        requirement = Requirement(requirement_line)
        if requirement.marker is None:
            continue
        requirement_extras.update(tl.utils._extras_from_requirement_marker(requirement))

    probes = tl.utils._declared_extra_probes()

    assert set(probes) == expected
    assert requirement_extras <= set(probes)
    assert {"jax", "mlx", "paddle", "profiler", "sae", "tensorflow", "tf", "tinygrad"} <= set(
        probes
    )
