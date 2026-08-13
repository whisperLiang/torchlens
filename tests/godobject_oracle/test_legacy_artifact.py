"""Frozen legacy-artifact gate: pre-refactor .tlspec files must always load.

The two committed artifacts under ``goldens/`` were produced at the
pre-columnar baseline (tree ``db2bc7a5``) and are NEVER regenerated: they ARE
the old format. Every wave of the columnar re-plumbing must load them through
the (eventually one-way legacy) decoder with a byte-identical public surface
and, for the runnable artifact, a VERIFIED faithful run.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import pytest
import torch
from surface_oracle._snapshot import canonical_dump, snapshot_trace_surface

import torchlens as tl

from .test_aliases import _SEED

_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_UPDATE_ENV = "TORCHLENS_UPDATE_SURFACE_ORACLE"

_ANALYSIS_ARTIFACT = _GOLDEN_DIR / "legacy_baseline_cnn.tlspec"
_RUNNABLE_ARTIFACT = _GOLDEN_DIR / "legacy_baseline_cnn_runnable.tlspec"
_LOADED_SURFACE_GOLDEN = _GOLDEN_DIR / "legacy_baseline_cnn_loaded.json"


@pytest.mark.smoke
def test_legacy_analysis_artifact_loads_byte_identically() -> None:
    """The frozen analysis artifact loads with an identical public surface."""

    assert _ANALYSIS_ARTIFACT.exists(), "frozen legacy artifact missing"
    loaded = tl.load(str(_ANALYSIS_ARTIFACT))
    actual = canonical_dump(snapshot_trace_surface(loaded))
    from _oracle_env import resolve_env_golden

    golden_path, record_on_missing = resolve_env_golden(_GOLDEN_DIR, _LOADED_SURFACE_GOLDEN.name)
    if os.environ.get(_UPDATE_ENV) == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual + "\n")
        pytest.skip("updated legacy loaded-surface golden")
    if record_on_missing and not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual + "\n")
        pytest.skip(f"recorded first-run loaded-surface golden for this environment: {golden_path}")
    assert golden_path.exists(), f"missing loaded-surface golden; generate with {_UPDATE_ENV}=1"
    expected = golden_path.read_text().rstrip("\n")
    if actual != expected:
        diff = "\n".join(
            list(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="golden",
                    tofile="actual",
                    lineterm="",
                )
            )[:60]
        )
        raise AssertionError(f"legacy loaded surface diverged:\n{diff}")


@pytest.mark.smoke
def test_legacy_runnable_artifact_runs_verified() -> None:
    """The frozen runnable artifact stages state and replays VERIFIED."""

    assert _RUNNABLE_ARTIFACT.exists(), "frozen runnable artifact missing"
    loaded = tl.load(str(_RUNNABLE_ARTIFACT))
    torch.manual_seed(_SEED)
    x = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)
    result = loaded.run(inputs=x)
    assert result.report.path_faithfulness.name == "VERIFIED"
