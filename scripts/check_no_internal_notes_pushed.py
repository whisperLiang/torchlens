#!/usr/bin/env python3
"""Refuse a push whose outgoing commits ADD an internal-notes path.

This repo is PUBLIC. The ``no-internal-notes`` pre-commit hook blocks ``git add``
of a private path, but it is deliberately scoped to the commit stage: at pre-push
the same path matcher would also fire on DELETIONS of those paths (for example the
commit that untracks them), so it cannot simply be re-registered for pushing.

That leaves a real gap. A commit created with ``--no-verify``, from a worktree with
hooks uninstalled, or by an agent lane that bypassed the commit hook, can still be
pushed -- and once an internal note is in remote history it is public even if a
later commit deletes it. That is exactly how ``FORKS.md`` / ``PROGRESS.md`` reached
an unpushed commit on 2026-08-11 (curated out of local history 2026-08-18, before
any push).

This guard closes it by inspecting ADDITIONS ONLY (``--diff-filter=A``) across the
outgoing range, so deletions never trip it.

Exit codes: 0 clean, 1 an outgoing commit adds a private path.
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Lockstep mirror of the ``no-internal-notes`` pre-commit ``files:`` regex and its
#: CI twin in ``tests/test_repo_privacy.py::PRIVATE_PATH_PATTERN``. Keep all three
#: in sync -- a path class that is private at commit time is private at push time.
PRIVATE_PATHSPECS = (
    ".research/**",
    ".project-context/**",
    "FORKS.md",
    "PROGRESS.md",
    "*_RESULTS.md",
    "*_REPORT*.md",
    "*_AUDIT*.md",
    "*_SUMMARY*.md",
    "*_LEDGER*.md",
    "*_BATON*.md",
    "*_FINDINGS*.md",
    "*_NOTES*.md",
    "*_STATE*.md",
    "*_PLAN*.md",
    "HUNT_*.md",
    "SPRINT_*.md",
    "ROUND_*.md",
)

#: The only ``.project-context/`` files this repo ships on purpose.
WHITELISTED = frozenset(
    {
        ".project-context/architecture.md",
        ".project-context/state_of_torchlens.md",
    }
)


def _outgoing_range() -> str | None:
    """Return the ``A..B`` range being pushed, or ``None`` when undeterminable.

    Returns
    -------
    str | None
        Commit range for the push, preferring pre-commit's pre-push env vars and
        falling back to ``origin/main..HEAD``.
    """

    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")
    if from_ref and to_ref:
        return f"{from_ref}..{to_ref}"
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "origin/main"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "origin/main..HEAD" if probe.returncode == 0 else None


def main() -> int:
    """Scan the outgoing range for private-path additions.

    Returns
    -------
    int
        ``0`` when no outgoing commit adds a private path, ``1`` otherwise.
    """

    commit_range = _outgoing_range()
    if commit_range is None:
        return 0  # no remote to compare against; nothing is being published
    result = subprocess.run(
        [
            "git",
            "log",
            "--diff-filter=A",
            "--name-only",
            "--format=%H",
            commit_range,
            "--",
            *PRIVATE_PATHSPECS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    offenders: dict[str, list[str]] = {}
    current = ""
    for line in result.stdout.splitlines():
        if len(line) == 40 and all(char in "0123456789abcdef" for char in line):
            current = line
            continue
        if line and line not in WHITELISTED:
            offenders.setdefault(current, []).append(line)
    if not offenders:
        return 0
    print(
        "BLOCKED: this push would publish internal notes to a PUBLIC repo.\n"
        "Once a path is in remote history it stays public even if a later commit\n"
        "deletes it, so the fix is to curate history BEFORE pushing, not after.\n",
        file=sys.stderr,
    )
    for sha, paths in offenders.items():
        print(f"  {sha[:12]} adds:", file=sys.stderr)
        for path in sorted(set(paths)):
            print(f"    {path}", file=sys.stderr)
    print(
        "\nRemedy: drop the paths from the outgoing commits, e.g.\n"
        "  git filter-branch -f --index-filter "
        "'git rm --cached --ignore-unmatch <paths>' --prune-empty -- origin/main..HEAD\n"
        "Take a backup ref first, and verify `git rev-parse HEAD^{tree}` is unchanged\n"
        "afterwards -- that proves you rewrote history without touching the code.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
