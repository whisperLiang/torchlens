"""Public-CLI acceptance uses the canonical authenticated real-environment composition."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.test_execution_dry_run_composition import (
    test_documented_dry_run_and_resume_use_real_environment as _assert_real_dry_run,
)

from .support import repository_root


# Real isolated workers execute both modes for all ten models; this is a
# release-environment composition, not an under-five-second unit test.
@pytest.mark.slow
@pytest.mark.optional
def test_cli_dry_run_real_forward_checkpoint_resume_and_milestone(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Exercise the release composition, including receipts, immutable ledgers and notices.

    Parameters
    ----------
    tmp_path:
        Disposable campaign parent; no live crawler state is used.
    real_environment_fixture:
        Strictly validated lock-built prefix, required rather than a fabricated worker.
    """

    _assert_real_dry_run(tmp_path, real_environment_fixture)


@pytest.mark.parametrize(
    ("release_gate", "diagnostic"),
    [
        ("0", "dry-run requires --dry-run-environment-prefix"),
        ("1", "unmet-release-gate: --dry-run-environment-prefix is required"),
    ],
)
def test_cli_dry_run_refuses_missing_environment_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_gate: str, diagnostic: str
) -> None:
    """Dry-run admission must fail before any campaign when no real prefix is selected.

    Parameters
    ----------
    tmp_path:
        Disposable path at which a refused invocation must not create a campaign.
    monkeypatch, release_gate, diagnostic:
        Select and verify the exact local or release-gate refusal contract.
    """

    monkeypatch.setenv("MENAGERIE_RELEASE_GATE", release_gate)
    campaign_root = tmp_path / "campaign"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "menagerie.crawler",
            "--repo-root",
            str(repository_root()),
            "run",
            "--dry-run",
            "--dry-run-root",
            str(campaign_root),
        ],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not campaign_root.exists()
