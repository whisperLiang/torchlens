"""Byte-identity oracle over the whole public record-object surface."""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import pytest

from ._snapshot import canonical_dump
from ._stages import MODEL_AXES, build_stage_snapshots

_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_UPDATE_ENV = "TORCHLENS_UPDATE_SURFACE_ORACLE"


def _golden_path(model_axis: str) -> Path:
    """Return the golden file path for one model axis."""

    return _GOLDEN_DIR / f"{model_axis}.json"


def _diff_summary(expected: str, actual: str, limit: int = 60) -> str:
    """Return a bounded unified diff between two canonical dumps."""

    diff_lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="golden",
            tofile="actual",
            lineterm="",
        )
    )
    body = "\n".join(diff_lines[:limit])
    if len(diff_lines) > limit:
        body += f"\n... ({len(diff_lines) - limit} more diff lines)"
    return body


@pytest.mark.smoke
@pytest.mark.parametrize("model_axis", MODEL_AXES)
def test_public_surface_matches_golden(model_axis: str) -> None:
    """The full public object surface is byte-identical to the golden.

    Any diff is a public behavior change: root-cause it as a regression in
    the storage re-plumbing. Re-snapshot ONLY for an intended, documented
    public change, via ``TORCHLENS_UPDATE_SURFACE_ORACLE=1``.
    """

    snapshots = build_stage_snapshots(model_axis)
    actual = canonical_dump(snapshots)
    golden_path = _golden_path(model_axis)
    if os.environ.get(_UPDATE_ENV) == "1":
        _GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(actual + "\n")
        pytest.skip(f"updated golden {golden_path.name}")
    assert golden_path.exists(), (
        f"missing golden {golden_path}; generate with {_UPDATE_ENV}=1"
    )
    expected = golden_path.read_text().rstrip("\n")
    if actual != expected:
        raise AssertionError(
            f"public surface diverged from golden for {model_axis}:\n"
            + _diff_summary(expected, actual)
        )


@pytest.mark.smoke
def test_surface_snapshot_is_deterministic() -> None:
    """Two independent capture+snapshot passes are byte-identical.

    Guards the oracle itself: if this fails, a nondeterministic value leaked
    into the canonical form and the oracle needs a normalization rule, not a
    golden refresh.
    """

    first = canonical_dump(build_stage_snapshots("plain_cnn"))
    second = canonical_dump(build_stage_snapshots("plain_cnn"))
    assert first == second
