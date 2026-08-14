"""Guard: the repo's git hooks must actually be reachable (b9 R70-1).

A global ``core.hooksPath`` redirect silently disarms EVERY repo hook --
including the LOCKED ``no-internal-notes`` privacy gate on this PUBLIC repo
and the format/lint pre-commit gates. That is precisely what happened on
2026-08-11 (see the recorded lesson): the redirect left ``.git/hooks``
inert and two CI-gate reds accumulated invisibly. This test turns the
disarm into a red test instead of an invisible condition.

The guard passes when any ONE of these holds:

* ``core.hooksPath`` is unset, so ``.git/hooks`` (the pre-commit framework
  install target) is consulted;
* the effective ``core.hooksPath`` resolves inside this repository;
* the redirected hooks directory's ``pre-commit`` script CHAINS to the
  repo-local hooks or the pre-commit framework (so both the machine-wide
  hook and the repo gates run).

Deliberately NOT smoke-tier: on a machine with the bare redirect this is a
true red that needs a machine-config fix (chain or scope the redirect), and
it must not poison every lane's commit-level gate mid-wave; the mid backstop
and CI (which has no global redirect) still see it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    """Run one git command in the repo and return stripped stdout.

    Parameters
    ----------
    args:
        Git subcommand and arguments.

    Returns
    -------
    str
        Stdout with surrounding whitespace removed (empty on nonzero exit).
    """

    completed = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def test_repo_hooks_are_reachable() -> None:
    """A hooksPath redirect must not silently disarm the repo's own gates."""

    hooks_path = _git("config", "core.hooksPath")
    if not hooks_path:
        return  # default: .git/hooks is consulted, repo hooks armed

    resolved = Path(hooks_path).expanduser()
    if not resolved.is_absolute():
        resolved = (_REPO_ROOT / resolved).resolve()

    common_dir = Path(_git("rev-parse", "--git-common-dir"))
    repo_targets = {
        _REPO_ROOT.resolve(),
        common_dir.resolve() if common_dir.is_absolute() else (_REPO_ROOT / common_dir).resolve(),
    }
    if any(target in resolved.parents or target == resolved for target in repo_targets):
        return  # repo-local redirect: repo hooks are the effective hooks

    redirected_pre_commit = resolved / "pre-commit"
    chained = False
    if redirected_pre_commit.exists():
        text = redirected_pre_commit.read_text(encoding="utf-8", errors="replace")
        chained = "pre-commit" in text.replace("#!", "").splitlines()[0] or any(
            marker in text
            for marker in (".git/hooks/pre-commit", "pre-commit run", "pre-commit hook-impl")
        )
    assert chained, (
        f"core.hooksPath is redirected to {resolved}, and its pre-commit does not "
        "chain to this repo's hooks: the no-internal-notes privacy gate and every "
        "other repo hook are DISARMED on this machine. Fix the machine config: "
        "either scope the redirect away from this repo, or make the global "
        "pre-commit script chain to `.git/hooks/pre-commit` / `pre-commit run` "
        "after its own checks. Do NOT delete this guard; it exists because this "
        "exact disarm shipped two invisible CI-gate reds (b9 R70-1)."
    )
