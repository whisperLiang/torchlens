"""Guard: the repo's git hooks must actually be reachable AND functional.

A global ``core.hooksPath`` redirect silently disarms EVERY repo hook --
including the LOCKED ``no-internal-notes`` privacy gate on this PUBLIC repo
and the never-ship-a-major commit-msg/pre-push layers. That is precisely what
happened on 2026-08-11 (see the recorded lesson): the redirect left
``.git/hooks`` inert and two CI-gate reds accumulated invisibly.

Round-4 rewrite (b10 R86-1 / b9 R70r3-F1): the previous guard was a COMMENT
DETECTOR — it substring-matched the whole global script for markers like
``.git/hooks/pre-commit`` and passed on a comment line while the actual chain
(``"$common_dir/hooks/pre-commit"``) matched nothing; it never checked that
the chained-to hook EXISTS; and it audited only ``pre-commit``, leaving the
``commit-msg``/``pre-push`` major-bump layers unexamined. Now:

* chain detection matches INVOCATIONS on non-comment lines only, including
  the ``$common_dir``/``$(git rev-parse --git-common-dir)`` spellings;
* every hook type declared in ``.pre-commit-config.yaml``'s
  ``default_install_hook_types`` must be INSTALLED (exists + executable) in
  the resolved git common dir on developer machines;
* both properties are checked per hook type, with a red-capable unit battery
  proving a comment-only marker no longer satisfies the guard.

Under ``CI`` the installed-ness check is skipped: ephemeral checkouts never
run ``pre-commit install`` — CI enforcement is the dedicated ``pre-commit
run --all-files`` job, which other gates pin.

Deliberately NOT smoke-tier: on a machine with a bare redirect this is a
true red that needs a machine-config fix (chain or scope the redirect), and
it must not poison every lane's commit-level gate mid-wave; the mid backstop
still sees it.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Invocation shapes that constitute a REAL chain to the repo hook of a given
#: type (formatted with the hook type). Matched on non-comment lines only.
_CHAIN_INVOCATION_PATTERNS = (
    r"\.git/hooks/{hook}",  # literal repo-hook path
    r"hooks/{hook}",  # $common_dir/hooks/<type> and equivalents
    r"pre-commit\s+run",  # framework direct invocation
    r"pre-commit\s+hook-impl",  # framework hook-impl invocation
)


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


def _declared_hook_types() -> tuple[str, ...]:
    """Return ``default_install_hook_types`` from .pre-commit-config.yaml."""

    config = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(r"^default_install_hook_types:\s*\[([^\]]+)\]", config, re.MULTILINE)
    assert match, (
        ".pre-commit-config.yaml lost default_install_hook_types — the "
        "commit-msg/pre-push major-bump layers install through it"
    )
    return tuple(part.strip() for part in match.group(1).split(","))


def _script_chains_to_hook(script_text: str, hook_type: str) -> bool:
    """Return whether a hooksPath script INVOKES the repo hook of ``hook_type``.

    Comment-only mentions do not count (the round-3 guard passed on a comment
    while the real chain line matched none of its markers). A line's comment
    tail is stripped before matching, so trailing annotations stay harmless.

    Parameters
    ----------
    script_text:
        Full text of the redirected hook script.
    hook_type:
        Hook type being audited (``pre-commit``, ``commit-msg``, ...).

    Returns
    -------
    bool
        Whether a non-comment line invokes the repo hook or the framework.
    """

    patterns = [pattern.format(hook=re.escape(hook_type)) for pattern in _CHAIN_INVOCATION_PATTERNS]
    for raw_line in script_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if any(re.search(pattern, line) for pattern in patterns):
            return True
    return False


def _resolved_common_hooks_dir() -> Path:
    """Return the resolved ``<git common dir>/hooks`` for this checkout."""

    common_dir = Path(_git("rev-parse", "--git-common-dir"))
    if not common_dir.is_absolute():
        common_dir = (_REPO_ROOT / common_dir).resolve()
    return common_dir / "hooks"


def test_declared_repo_hooks_are_installed_and_executable() -> None:
    """Every declared hook type must be installed in the git common dir.

    The chained-to hook EXISTING is half the guard: a redirect that chains to
    ``$common_dir/hooks/pre-commit`` in a repo where ``pre-commit install``
    never ran passes any text check while enforcing nothing (b10 R86-1).
    """

    if os.environ.get("CI"):
        pytest.skip(
            "ephemeral CI checkouts do not install git hooks; the "
            "pre-commit run --all-files job is CI's enforcement layer"
        )
    hooks_dir = _resolved_common_hooks_dir()
    problems = []
    for hook_type in _declared_hook_types():
        hook = hooks_dir / hook_type
        if not hook.exists():
            problems.append(f"{hook} missing")
        elif not os.access(hook, os.X_OK):
            problems.append(f"{hook} not executable")
    assert not problems, (
        "declared git hooks are not installed — the no-internal-notes privacy "
        "gate and/or the never-ship-a-major commit-msg/pre-push layers are "
        "DISARMED on this machine. Run `pre-commit install` in the repo. "
        "Problems:\n  " + "\n  ".join(problems)
    )


def test_repo_hooks_are_reachable() -> None:
    """A hooksPath redirect must chain BY INVOCATION for every declared type."""

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

    unchained = []
    for hook_type in _declared_hook_types():
        redirected = resolved / hook_type
        if not redirected.exists():
            # git falls back to NO hook for this type under a hooksPath
            # redirect — the repo layer for this type never runs.
            unchained.append(f"{hook_type}: {redirected} does not exist")
            continue
        text = redirected.read_text(encoding="utf-8", errors="replace")
        if not _script_chains_to_hook(text, hook_type):
            unchained.append(
                f"{hook_type}: {redirected} never INVOKES the repo hook "
                "(comment mentions do not count)"
            )
    assert not unchained, (
        f"core.hooksPath is redirected to {resolved} and these hook types do "
        "not chain to this repo's hooks: the no-internal-notes privacy gate "
        "and/or the never-ship-a-major layers are DISARMED on this machine. "
        "Fix the machine config: scope the redirect away from this repo, or "
        "make each global script invoke `$(git rev-parse --git-common-dir)"
        "/hooks/<type>` after its own checks. Do NOT delete this guard "
        "(b9 R70-1 / b10 R86-1).\n  " + "\n  ".join(unchained)
    )


# ---------------------------------------------------------------------------
# Red-capability: the round-3 false-green shapes must now be caught
# ---------------------------------------------------------------------------

_COMMENT_ONLY_SCRIPT = """#!/bin/bash
# Chain to the repository's own .git/hooks/pre-commit so this global
# hook does not disarm repo gates.
gitleaks protect --staged
exit 0
"""

_REAL_CHAIN_SCRIPT = """#!/bin/bash
gitleaks protect --staged || exit 1
common_dir=$(git rev-parse --git-common-dir)
if [ -x "$common_dir/hooks/pre-commit" ]; then
  "$common_dir/hooks/pre-commit" "$@"  # chain to the repo's own hook
fi
"""

_FRAMEWORK_CHAIN_SCRIPT = """#!/bin/sh
pre-commit run --hook-stage pre-commit
"""

_TRAILING_COMMENT_ONLY = """#!/bin/sh
true  # we deliberately do NOT chain to .git/hooks/pre-commit
"""


@pytest.mark.parametrize(
    ("script", "hook_type", "expected"),
    [
        pytest.param(_COMMENT_ONLY_SCRIPT, "pre-commit", False, id="comment-only-false-green"),
        pytest.param(_TRAILING_COMMENT_ONLY, "pre-commit", False, id="trailing-comment"),
        pytest.param(_REAL_CHAIN_SCRIPT, "pre-commit", True, id="common-dir-invocation"),
        pytest.param(_FRAMEWORK_CHAIN_SCRIPT, "pre-commit", True, id="framework-run"),
        pytest.param(_REAL_CHAIN_SCRIPT, "commit-msg", False, id="wrong-type-not-satisfied"),
    ],
)
def test_chain_detector_is_red_capable(script: str, hook_type: str, expected: bool) -> None:
    """Comment-only markers fail; real invocations pass; types are distinct."""

    assert _script_chains_to_hook(script, hook_type) is expected
