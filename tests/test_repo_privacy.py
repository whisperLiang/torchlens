"""Repo-privacy gate: no private working-notes path may ever be tracked.

torchlens is a PUBLIC repository. Internal planning notes live in gitignored
territory (``.research/``, ``.project-context/`` minus the two curated docs,
and repo-root sprint ledgers). The primary defense is the ``no-internal-notes``
pre-commit hook, but hooks are one ``--no-verify`` (or a global
``core.hooksPath`` override) away from silence and have zero CI layers.

This test is the hook-independent layer: it asserts against ``git ls-files``
directly, so a private path that has already been committed -- however it got
past the hook -- turns the smoke tier red on every machine and CI leg.

The path matcher below MUST stay in lockstep with the ``files:`` regex of the
``no-internal-notes`` hook in ``.pre-commit-config.yaml``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent

# Lockstep mirror of .pre-commit-config.yaml no-internal-notes `files:` regex.
PRIVATE_PATH_PATTERN = re.compile(
    r"^\.research/"
    r"|^\.project-context/(?!(architecture|state_of_torchlens)\.md$)"
    r"|^(FORKS|PROGRESS)\.md$"
    r"|^[^/]*_RESULTS\.md$"
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout (sdist/wheel install); privacy gate not applicable")
    return [p for p in result.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def find_private_violations(paths: list[str]) -> list[str]:
    return [p for p in paths if PRIVATE_PATH_PATTERN.search(p)]


def test_no_private_paths_are_tracked() -> None:
    violations = find_private_violations(_tracked_files())
    assert violations == [], (
        "PRIVATE paths are tracked in this PUBLIC repo -- untrack with "
        f"`git rm --cached` and never `git add -f` them: {violations}"
    )


def test_matcher_is_red_capable() -> None:
    """Non-vacuity: the matcher fires on each private class and spares the whitelist."""

    hits = find_private_violations(
        [
            ".research/notes.md",
            ".project-context/todos.md",
            ".project-context/torchlens_glossary.md",
            "FORKS.md",
            "PROGRESS.md",
            "sprint_RESULTS.md",
        ]
    )
    assert len(hits) == 6
    assert (
        find_private_violations(
            [
                ".project-context/architecture.md",
                ".project-context/state_of_torchlens.md",
                "RESULTS.md",
                "benchmarks/intervention_overhead_results.md",
                "torchlens/__init__.py",
            ]
        )
        == []
    )


def test_matcher_mirrors_precommit_hook() -> None:
    """The hook config's regex and this test's matcher may never drift apart."""

    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(r"id: no-internal-notes.*?files: '([^']+)'", config, flags=re.DOTALL)
    assert match is not None, "no-internal-notes hook (or its files: regex) is gone"
    hook_regex = re.compile(match.group(1))
    probes = [
        ".research/x.md",
        ".project-context/todos.md",
        ".project-context/architecture.md",
        ".project-context/state_of_torchlens.md",
        "FORKS.md",
        "PROGRESS.md",
        "lane_RESULTS.md",
        "RESULTS.md",
        "docs/guide.md",
    ]
    for probe in probes:
        assert bool(hook_regex.search(probe)) == bool(PRIVATE_PATH_PATTERN.search(probe)), (
            f"hook regex and test matcher disagree on {probe!r}"
        )
