"""Byte-identity oracle over the whole public record-object surface.

Snapshots are generated in a fresh subprocess per model axis (the capture-
oracle isolation pattern): in-process generation constructed every axis after
the first on already-wrapped torch, so the goldens silently froze wrap-state
artifacts and the oracle's verdict depended on what ran earlier in the
session (b10 R78-1, the SF-53 generator-design gap).
"""

from __future__ import annotations

import difflib
import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ._stages import MODEL_AXES

_ROOT = Path(__file__).parents[2]
_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_UPDATE_ENV = "TORCHLENS_UPDATE_SURFACE_ORACLE"


def _run_worker(model_axes: tuple[str, ...]) -> dict[str, str]:
    """Generate canonical dumps in one isolated Python subprocess."""

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    python_paths = (str(_ROOT / "tests"), str(_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, *((existing_pythonpath,) if existing_pythonpath else ()))
    )
    completed = subprocess.run(
        [sys.executable, "-m", "surface_oracle._worker", *model_axes],
        cwd=_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"surface worker produced no JSON for {model_axes}")
    import json

    dumps = json.loads(lines[-1])
    assert isinstance(dumps, dict) and set(dumps) == set(model_axes)
    return dumps


@functools.lru_cache(maxsize=1)
def _worker_dumps() -> dict[str, str]:
    """One batch generation for the whole axis family (models prebuilt pre-wrap)."""

    return _run_worker(MODEL_AXES)


def _worker_dump(model_axis: str) -> str:
    """Return one axis's isolated canonical dump."""

    return _worker_dumps()[model_axis]


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


# heavy, not smoke (r3settle2 budget lint): the family's ONE batch
# subprocess (all six axes generated pre-wrap in a single isolated
# interpreter, the SF-53 design) costs ~7-9s attributed to the first
# cell -- over the 5s smoke partition, inside heavy's 5-20s band.
@pytest.mark.heavy
@pytest.mark.parametrize("model_axis", MODEL_AXES)
def test_public_surface_matches_golden(model_axis: str) -> None:
    """The full public object surface is byte-identical to the golden.

    Any diff is a public behavior change: root-cause it as a regression in
    the storage re-plumbing. Re-snapshot ONLY for an intended, documented
    public change, via ``TORCHLENS_UPDATE_SURFACE_ORACLE=1``.
    """

    actual = _worker_dump(model_axis)
    from _oracle_env import (
        flag_armed,
        require_env_golden,
        require_update_reason,
        resolve_env_golden,
        write_provenance,
    )

    # No wrap-state guard here: generation runs in an isolated subprocess
    # with a clean interpreter (SF-53 is closed structurally for this family).
    if flag_armed(os.environ, _UPDATE_ENV):
        # Reason BEFORE bytes (b10 R78 round-4): the write used to precede
        # require_update_reason, so a reasonless update run FAILED but had
        # already rebaselined the committed goldens in the working tree.
        reason = require_update_reason(_UPDATE_ENV)
        golden_path, _ = resolve_env_golden(_GOLDEN_DIR, f"{model_axis}.json")
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual + "\n")
        write_provenance(golden_path.parent, "tests/surface_oracle", _UPDATE_ENV, reason)
        pytest.skip(f"updated golden {golden_path.name}; re-run without {_UPDATE_ENV} to verify")
    golden_path = require_env_golden(_GOLDEN_DIR, f"{model_axis}.json", _UPDATE_ENV)
    if not golden_path.exists():
        reason = require_update_reason(_UPDATE_ENV)
        golden_path.write_text(actual + "\n")
        write_provenance(golden_path.parent, "tests/surface_oracle", _UPDATE_ENV, reason)
        pytest.skip(f"recorded first-run surface golden for this environment: {golden_path}")
    expected = golden_path.read_text().rstrip("\n")
    if actual != expected:
        raise AssertionError(
            f"public surface diverged from golden for {model_axis}:\n"
            + _diff_summary(expected, actual)
        )


# heavy with the family above: first-caller cache fill plus the fresh
# single-axis pass exceeds the smoke partition on its own.
@pytest.mark.heavy
def test_surface_snapshot_is_deterministic() -> None:
    """Two independent isolated generation passes are byte-identical.

    Guards the oracle itself: if this fails, a nondeterministic value leaked
    into the canonical form and the oracle needs a normalization rule, not a
    golden refresh. Comparing two SUBPROCESS runs (not two same-session
    snapshots) also covers process-level state the historical in-process
    double-snapshot was structurally blind to (b10 R78-1).
    """

    first = _worker_dump("plain_cnn")
    second = _run_worker(("plain_cnn",))["plain_cnn"]
    assert first == second
