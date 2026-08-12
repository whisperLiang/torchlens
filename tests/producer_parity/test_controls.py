"""Legacy-vs-legacy parity controls (P0 gate: controls empty, shims armed).

The in-process control doubles as the shim-perturbation control: every run
here executes with the attestation shims ARMED, so an empty diff also proves
the shims observe without perturbing (the same scenario compared against an
unshimmed run in ``test_shim_perturbation_control``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ._comparator import check_a, check_b, compare_runs
from ._models import SCENARIOS, scenario_by_name
from ._snapshot import run_scenario
from ._worker import load_snapshot

pytestmark = pytest.mark.heavy

_CONTROL_SCENARIOS = tuple(scenario.name for scenario in SCENARIOS)


def _format(diffs: list) -> str:
    return "\n".join(
        f"[{d.check}/{d.kind}] {d.layer}:{d.anchor}:{d.path} {d.detail}" for d in diffs[:40]
    )


@pytest.mark.parametrize("scenario_name", _CONTROL_SCENARIOS)
def test_in_process_control(scenario_name: str, tmp_path: Path) -> None:
    """Two same-leg runs compare empty through both checks (shims armed)."""

    scenario = scenario_by_name(scenario_name)
    run_left = run_scenario(scenario, tmp_path / "left")
    run_right = run_scenario(scenario, tmp_path / "right")
    diffs = compare_runs(run_left, run_right)
    assert not diffs, f"legacy-vs-legacy control not empty:\n{_format(diffs)}"


def test_shim_perturbation_control(tmp_path: Path) -> None:
    """A shimmed run and an unshimmed run are structurally identical."""

    scenario = scenario_by_name("cnn_exhaustive")
    shimmed = run_scenario(scenario, tmp_path / "shimmed", arm_shims=True)
    bare = run_scenario(scenario, tmp_path / "bare", arm_shims=False)
    diffs = check_a(shimmed.snapshot, bare.snapshot)
    assert not diffs, f"shims perturbed the capture:\n{_format(diffs)}"


def test_check_b_clean_on_every_scenario(tmp_path: Path) -> None:
    """Within-leg attestation is green on unmutated captures."""

    for scenario in SCENARIOS:
        run = run_scenario(scenario, tmp_path / scenario.name)
        diffs = check_b(run)
        assert not diffs, f"{scenario.name}: Check B not clean:\n{_format(diffs)}"


@pytest.mark.slow
def test_cross_process_control(tmp_path: Path) -> None:
    """A fresh-interpreter run (all ids relabeled) compares empty: the real
    bijective-relabel green-proof plus Check B in the worker."""

    scenario = scenario_by_name("cnn_exhaustive")
    local = run_scenario(scenario, tmp_path / "local")

    out_path = tmp_path / "worker.json"
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'tests'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    subprocess.run(
        [sys.executable, "-m", "producer_parity._worker", scenario.name, str(out_path)],
        check=True,
        env=env,
        cwd=str(repo_root),
        timeout=600,
    )
    payload = json.loads(out_path.read_text())
    assert payload["check_b_diffs"] == [], f"worker Check B: {payload['check_b_diffs'][:10]}"
    remote_snapshot = load_snapshot(payload)
    diffs = check_a(local.snapshot, remote_snapshot)
    assert not diffs, f"cross-process control not empty:\n{_format(diffs)}"


def test_grad_fn_retention_premise(tmp_path: Path) -> None:
    """Check A's partition premise: the handle index holds STRONG refs in a
    plain dict for the run (Opus v4 note N4). A weak-ref index would allow
    id() recycling inside the comparison window and must fail here."""

    import weakref

    from torchlens.ir.capture_events import CaptureEvents

    events = CaptureEvents()
    assert type(events.grad_fn_handles_by_label_raw) is dict
    assert not isinstance(events.grad_fn_handles_by_label_raw, weakref.WeakValueDictionary)
