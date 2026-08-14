"""CI / packaging governance gates (grind-p3 T13 fix lane).

Each test here pins a CI-plumbing invariant that regressed silently at least
once: pre-commit and the lint gate disagreeing on ruff order, oracle legs
skipping every golden while staying green, the byte oracles never executing
on any CI leg, unscoped release credentials, and an ungoverned sdist. These
are parse/lint assertions over the checked-in config — the runtime halves
live in the workflows themselves.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOWS = _PROJECT_ROOT / ".github" / "workflows"


def _load_yaml(path: Path) -> dict:
    """Parse one YAML config file."""

    return yaml.safe_load(path.read_text())


def _smoke_job() -> dict:
    """Return the smoke job of the Tests workflow."""

    return _load_yaml(_WORKFLOWS / "tests.yml")["jobs"]["smoke"]


def test_exactly_one_smoke_row_enforces_the_byte_oracle_goldens() -> None:
    """One smoke row declares oracle enforcement and matches the ENV markers.

    The byte-oracle goldens enforce only where the environment fingerprint
    matches the committed ``ENV`` marker; every other CI leg legitimately
    skips them. Without a declared enforcing row, a matrix torch bump moves
    that one leg off-canonical and the golden families enforce on NO leg at
    all while every row stays green (T13.1). This pins the lockstep between
    the matrix row and the markers, so bumping either alone goes red.
    """

    rows = [
        row for row in _smoke_job()["strategy"]["matrix"]["include"] if row.get("scope") == "smoke"
    ]
    assert len(rows) >= 6, "the PR-blocking smoke matrix shrank unexpectedly"
    enforcing = [row for row in rows if str(row.get("oracle_enforce", "")) == "1"]
    assert len(enforcing) == 1, (
        "exactly one smoke row must declare oracle_enforce so the byte-oracle "
        "goldens are guaranteed to enforce on one CI leg"
    )
    row = enforcing[0]
    row_fingerprint = f"py{row['python']}-torch{str(row['torch']).split('+', 1)[0]}"
    for goldens_dir in ("surface_oracle", "godobject_oracle"):
        marker = (_PROJECT_ROOT / "tests" / goldens_dir / "goldens" / "ENV").read_text().strip()
        assert row_fingerprint == marker, (
            f"the enforcing smoke row ({row_fingerprint}) no longer matches the "
            f"committed {goldens_dir} ENV marker ({marker}); rebaseline the "
            "goldens deliberately or fix the matrix row — do not let them drift"
        )


def test_every_smoke_row_runs_the_executed_floor_attestation() -> None:
    """All smoke rows emit junit XML and enforce an executed-test floor.

    A leg whose suite hollows out (collection drift, conftest guards, mass
    importorskip or oracle skipping) exits 0 and stays green; the floor turns
    an under-executed run into a failure. Previously only the nightly preview
    legs carried this attestation (T13.1).
    """

    steps = _smoke_job()["steps"]
    run_step = next(step for step in steps if step.get("name", "").startswith("Run smoke tests"))
    assert "--junitxml" in run_step["run"], "smoke run must emit junit XML"
    assert run_step["env"]["TORCHLENS_ORACLE_ENFORCE"] == "${{ matrix.oracle_enforce }}", (
        "the enforce declaration must reach the test process"
    )
    floor_step = next(
        (step for step in steps if "check_ci_executed_tests.py" in step.get("run", "")),
        None,
    )
    assert floor_step is not None, "smoke rows lost the executed-floor attestation"
    assert floor_step.get("if") == run_step.get("if"), (
        "the floor check must run on every row the smoke suite runs on"
    )


def test_precommit_ruff_fix_runs_before_ruff_format() -> None:
    """pre-commit applies lint fixes BEFORE formatting, matching the CI gate.

    ``ruff check --fix`` can rewrite code into an unformatted shape; running it
    AFTER ``ruff-format`` therefore commits bytes the Lint workflow's
    format-then-lint check rejects. Ruff's own pre-commit guidance orders the
    ``ruff`` (fix) hook before ``ruff-format`` so the formatter has the last
    word locally, exactly like ``ruff format --check`` has the last word in CI.
    """

    config = _load_yaml(_PROJECT_ROOT / ".pre-commit-config.yaml")
    ruff_repo = next(repo for repo in config["repos"] if "ruff-pre-commit" in repo["repo"])
    hook_ids = [hook["id"] for hook in ruff_repo["hooks"]]
    assert hook_ids.index("ruff") < hook_ids.index("ruff-format"), (
        "pre-commit must run the ruff (--fix) hook BEFORE ruff-format; the "
        "reverse order lets a lint autofix produce unformatted code that the "
        "CI `ruff format --check` gate rejects"
    )
