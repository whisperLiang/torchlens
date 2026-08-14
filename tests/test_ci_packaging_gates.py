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

import pytest
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


def test_render_byte_oracle_executes_on_a_ci_leg() -> None:
    """A dedicated step actually RUNS the heavy render byte-oracle test.

    The byte half of the render-identity oracle is heavy-marked, so setting
    ``TORCHLENS_RENDER_BYTE_ORACLE`` on a row whose selection is ``-m smoke``
    was dead config: DOT-byte identity was enforced on no CI leg while the
    flag looked wired (T13.2). The dedicated step must select the heavy
    marker and attest execution through the junit floor.
    """

    smoke = _smoke_job()
    rows = [
        row
        for row in smoke["strategy"]["matrix"]["include"]
        if str(row.get("render_byte_oracle", "")) == "1"
    ]
    assert len(rows) == 1, "exactly one smoke row must carry the render byte oracle"
    step = next(
        (
            step
            for step in smoke["steps"]
            if "test_viz_render_identity_oracle.py" in step.get("run", "")
        ),
        None,
    )
    assert step is not None, "no smoke step executes the render byte-oracle test"
    assert step.get("if") == "matrix.render_byte_oracle == '1'"
    assert step["env"]["TORCHLENS_RENDER_BYTE_ORACLE"] == "1"
    assert "-m heavy" in step["run"], (
        "the byte-oracle consumer is heavy-marked; without selecting the "
        "heavy marker the step executes nothing"
    )
    assert "check_ci_executed_tests.py" in step["run"], (
        "the byte-oracle step must attest the test EXECUTED rather than skipped"
    )


def test_capture_oracle_matrix_enforces_on_a_nightly_leg() -> None:
    """Nightly runs the slow capture-characterization matrix with a floor."""

    nightly = _load_yaml(_WORKFLOWS / "nightly.yml")["jobs"]
    job = nightly.get("capture-byte-oracle")
    assert job is not None, "nightly lost the capture-byte-oracle job (T13.2)"
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "tests/capture_oracle/" in runs and "-m slow" in runs
    assert "check_ci_executed_tests.py" in runs, (
        "the capture-oracle leg must attest executed tests: the version gate "
        "skips the whole matrix on any non-recording torch, which is exactly "
        "the silent-green this leg exists to prevent"
    )


def test_capture_oracle_version_gate_strips_the_build_tag() -> None:
    """The golden version gate compares torch SOURCE versions, not build tags.

    A ``+cu130``-recorded golden must enforce on a ``+cpu`` CI runtime of the
    same torch version; comparing full build strings made every CI leg skip
    the capture-characterization matrix forever (T13.2).
    """

    from capture_oracle.test_capture_oracle import _recording_torch_matches

    assert _recording_torch_matches("2.13.0+cu130", "2.13.0+cpu")
    assert _recording_torch_matches("2.13.0", "2.13.0+cpu")
    assert not _recording_torch_matches("2.12.0+cpu", "2.13.0+cpu")
    assert not _recording_torch_matches(None, "2.13.0+cpu")


def test_release_app_token_is_permission_scoped() -> None:
    """The minted GitHub App token carries an explicit minimal permission set.

    Without a ``permission-*`` input the token inherits the App
    installation's FULL permissions and checkout persists it to disk for the
    whole release job, including third-party pip installs (zizmor
    ``github-app`` HIGH, T13.3). Contents write is everything the job needs.
    """

    release = _load_yaml(_WORKFLOWS / "release.yml")["jobs"]["release"]
    token_step = next(
        step for step in release["steps"] if "create-github-app-token" in step.get("uses", "")
    )
    scoped = [key for key in token_step["with"] if key.startswith("permission-")]
    assert scoped == ["permission-contents"], (
        "the release App token must be minted with exactly the minimal permission-contents scope"
    )
    assert token_step["with"]["permission-contents"] == "write"


def test_non_release_checkouts_do_not_persist_credentials() -> None:
    """Every checkout that never pushes sets ``persist-credentials: false``.

    checkout persists its token into ``.git/config`` for ALL later steps by
    default (zizmor ``artipacked``). Only the release job's checkout may
    persist — semantic-release pushes the version commit and tag through it.
    """

    offenders = []
    for workflow in sorted(_WORKFLOWS.glob("*.yml")):
        for job_name, job in _load_yaml(workflow)["jobs"].items():
            for step in job.get("steps", ()):
                if "actions/checkout@" not in step.get("uses", ""):
                    continue
                with_block = step.get("with", {})
                if workflow.name == "release.yml" and job_name == "release":
                    assert "token" in with_block, (
                        "the release checkout must authenticate as the App "
                        "(its push path relies on the persisted scoped token)"
                    )
                    continue
                if with_block.get("persist-credentials") is not False:
                    offenders.append(f"{workflow.name}:{job_name}")
    assert not offenders, (
        f"checkout steps persisting credentials without needing to push: {offenders}"
    )


def test_a_smoke_row_covers_the_sys_monitoring_python() -> None:
    """At least one PR-blocking row runs Python >= 3.12.

    Advertised 3.12/3.13 support selects the ``sys.monitoring``
    escape-detection path, which 3.10/3.11 rows can never execute: without a
    >=3.12 row that whole path ships untested (its only >=3.12-gated unit
    test had already rotted unnoticed, T13.4).
    """

    rows = [
        row for row in _smoke_job()["strategy"]["matrix"]["include"] if row.get("scope") == "smoke"
    ]
    versions = [tuple(int(part) for part in str(row["python"]).split(".")) for row in rows]
    assert any(version >= (3, 12) for version in versions), (
        "no smoke row runs Python >= 3.12; the sys.monitoring escape-detection "
        "path has no CI coverage"
    )


def test_packaging_tripwires_run_on_the_nightly_wheel_leg() -> None:
    """Nightly builds run the wheel-diet and sdist manifest tripwires.

    The nightly wheel job used to build and smoke-install a wheel WITHOUT
    running the diet manifest test (slow-marked, weekly-only), so a packaging
    regression could ship for up to a week before the tripwire fired (T13.5).
    """

    wheel_job = _load_yaml(_WORKFLOWS / "nightly.yml")["jobs"]["wheel"]
    runs = "\n".join(step.get("run", "") for step in wheel_job["steps"])
    assert "test_built_wheel_manifest_is_diet" in runs, (
        "the nightly wheel job must run the wheel-diet tripwire, not just build"
    )
    assert "test_built_sdist_manifest_is_governed" in runs, (
        "the nightly wheel job must run the sdist manifest tripwire"
    )


@pytest.mark.slow
def test_built_sdist_manifest_is_governed(tmp_path: Path) -> None:
    """Assert the built sdist's manifest: package + metadata in, half-suites OUT.

    With no MANIFEST.in the default manifest shipped every ``tests/*.py``
    file WITHOUT its goldens, ENV markers, or data — an unrunnable half-suite
    — and nothing pinned the sdist at all (T13.5).
    """

    import subprocess
    import sys
    import tarfile

    sdist_dir = tmp_path / "sdist"
    sdist_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(sdist_dir)],
        cwd=_PROJECT_ROOT,
        check=True,
    )
    archives = sorted(sdist_dir.glob("torchlens-*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        members = [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]

    for required in (
        "LICENSE",
        "NOTICE",
        "README.md",
        "pyproject.toml",
        "torchlens/__init__.py",
        "torchlens/py.typed",
    ):
        assert required in members, f"sdist lost required member {required}"
    assert any(m.startswith("torchlens/schemas/") and m.endswith(".json") for m in members), (
        "sdist must ship the torchlens schema data files"
    )
    for banned_prefix in ("tests/", "menagerie/", "docs/", "examples/", "notebooks/"):
        offenders = [m for m in members if m.startswith(banned_prefix)]
        assert not offenders, (
            f"sdist ships {len(offenders)} member(s) under {banned_prefix} — the sdist "
            "is the wheel's source, not a repo snapshot (half-shipped suites are "
            "unrunnable; use a checkout)"
        )
