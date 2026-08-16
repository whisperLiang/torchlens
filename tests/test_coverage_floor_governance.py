"""Coverage-floor governance: the tripwire floor must be LOCKED, not a YAML literal.

R72 (b9-fable/opus, round 3+4): the nightly coverage job's ``--cov-fail-under``
floor existed ONLY as a workflow literal with a "never lower it (tripwire)"
comment — no test pinned it, so deleting the job or editing the floor down to 1
kept every gate green. These checks make the claim mechanical:

1. the nightly ``coverage`` job EXISTS and instruments the smoke tier;
2. its ``--cov-fail-under`` value never drops below the committed baseline
   (shrink-forbidden ratchet: raising it is welcome, lowering goes red here);
3. ``[tool.coverage.report] fail_under`` carries the same authority in
   pyproject so a local ``coverage report`` reaches the same verdict as CI
   (R72 sol: without it, local and CI coverage verdicts silently differed).

pyproject parsing is line/regex-based on purpose: ``tomllib`` only exists on
3.11+ and the suite still runs a 3.10 leg (the test_order_isolation_infra
precedent). The workflow side parses real YAML (PyYAML is a declared test
dep) so the assertions bind to the actual ``jobs.coverage`` command instead
of whole-file substrings (r7 R72).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Committed coverage-floor baseline (percent). NEVER lower this to make a
#: gate pass — that is the exact silent-disarm this file exists to prevent.
#: Raise it as measured coverage grows (measured 64% on 2026-08-15; the 55
#: floor's 9-point slack is a known, deliberately conservative first gate).
COVERAGE_FLOOR_BASELINE = 55


def _nightly_text() -> str:
    """Return the nightly workflow text, or skip when the tree ships none.

    Returns
    -------
    str
        Raw text of ``.github/workflows/nightly.yml``.
    """

    path = _PROJECT_ROOT / ".github" / "workflows" / "nightly.yml"
    if not path.exists():
        pytest.skip("no nightly workflow in this tree (sdist/test-only layout)")
    return path.read_text(encoding="utf-8")


def _coverage_job_command() -> str:
    """Return the nightly ``coverage`` job's pytest command, job-scoped.

    r7 R72 (sol MED): the old checks searched the WHOLE workflow text for
    ``--cov=torchlens`` / floor literals, so an emptied coverage job plus
    those strings anywhere else satisfied every predicate. Parse the actual
    job so the assertions bind to the command that runs.
    """

    import yaml

    workflow = yaml.safe_load(_nightly_text())
    job = workflow.get("jobs", {}).get("coverage")
    assert job is not None, (
        "the nightly workflow no longer declares the `coverage` job — the only "
        "measured coverage gate in the repo (R72/SF-19). Restore it; do not "
        "delete the sole coverage tripwire."
    )
    commands = [
        step.get("run", "")
        for step in job.get("steps", [])
        if "pytest" in step.get("run", "") and "--cov" in step.get("run", "")
    ]
    assert len(commands) == 1, (
        f"expected exactly ONE instrumented pytest command in jobs.coverage, got {len(commands)}"
    )
    return commands[0]


def test_nightly_coverage_job_exists_and_instruments_the_smoke_tier() -> None:
    """Deleting or hollowing the nightly coverage job must go red here."""

    command = _coverage_job_command()
    assert "-m smoke" in command, "the coverage job no longer selects the smoke tier"
    assert "--cov=torchlens" in command and "--cov-branch" in command, (
        "the nightly coverage job no longer instruments torchlens under branch "
        "coverage — the floor would be measuring nothing"
    )
    # The instrumented leg must not deselect budget tests by name: the
    # carve-out lives in the budget machinery itself (a renamed test id once
    # turned the deselect into a silent no-op while claiming the exemption).
    assert "--deselect" not in command, (
        "the coverage job deselects tests by name again — the instrumentation "
        "carve-out belongs in tests/conftest.py (cov_source check), where a "
        "rename cannot silently void it"
    )
    conftest_text = (_PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert 'getattr(item.config.option, "cov_source", None)' in conftest_text, (
        "tests/conftest.py lost the instrumented-session budget carve-out that "
        "replaced the coverage job's brittle --deselect"
    )


def test_nightly_coverage_floor_is_never_lowered() -> None:
    """The job's --cov-fail-under may rise but never drop below baseline."""

    command = _coverage_job_command()
    floors = [int(value) for value in re.findall(r"--cov-fail-under=(\d+)", command)]
    assert floors, (
        "the nightly coverage job lost its --cov-fail-under floor entirely; "
        f"restore at least --cov-fail-under={COVERAGE_FLOOR_BASELINE}"
    )
    lowered = [floor for floor in floors if floor < COVERAGE_FLOOR_BASELINE]
    assert not lowered, (
        f"nightly --cov-fail-under {lowered} sits below the committed baseline "
        f"{COVERAGE_FLOOR_BASELINE}. The floor is a tripwire: never lower it to "
        "pass — root-cause the coverage loss instead. (Raising the baseline "
        "constant in this file alongside a real coverage gain is the only "
        "sanctioned edit.)"
    )


def test_pyproject_carries_the_same_coverage_floor() -> None:
    """[tool.coverage.report] fail_under must match the committed baseline.

    Keeps local ``coverage report`` verdicts aligned with CI's flag-passed
    floor: the flag overrides config, so the two can only disagree when the
    config is silently missing — which is exactly how the floor stayed
    unlocked for four review passes.
    """

    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^fail_under\s*=\s*(\d+)", pyproject, flags=re.MULTILINE)
    assert match, (
        "[tool.coverage.report] lost its fail_under — local coverage runs no "
        f"longer enforce any floor. Restore fail_under = {COVERAGE_FLOOR_BASELINE}."
    )
    assert int(match.group(1)) >= COVERAGE_FLOOR_BASELINE, (
        f"pyproject fail_under = {match.group(1)} sits below the committed "
        f"baseline {COVERAGE_FLOOR_BASELINE} — the floor is a tripwire, never "
        "lower it to pass"
    )


def test_floor_lock_is_red_capable() -> None:
    """The floor extraction flags a lowered literal (red-capability self-test)."""

    planted = "run: |\n  pytest tests/ --cov=torchlens --cov-fail-under=12 --tb=short"
    floors = [int(value) for value in re.findall(r"--cov-fail-under=(\d+)", planted)]
    assert floors == [12]
    assert [floor for floor in floors if floor < COVERAGE_FLOOR_BASELINE] == [12]
