"""Prepare and attest the parent and fixed-anchor worktrees for the nightly A/B gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    """Run a required git operation, refusing missing history or broken objects."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _verified_commit(repo: Path, ref: str) -> str:
    """Resolve a commit and verify its checkout content before creating any worktree."""
    commit = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    return commit


def prepare_baselines(repo: Path, workdir: Path, anchor_file: Path) -> dict[str, str]:
    """Create both A/B baseline trees and record the exact commits and anchor decision.

    Parameters
    ----------
    repo:
        Current checkout with full history and tags fetched.
    workdir:
        Destination for ``perf-base``, ``perf-anchor`` and their JSON attestation.
    anchor_file:
        Reviewed immutable initial anchor used when no release tag is reachable.

    Returns
    -------
    dict[str, str]
        Commit identities and the reason for selecting the fixed anchor.
    """
    tags = _git(repo, "tag", "--merged", "HEAD", "--list", "v[0-9]*")
    if tags:
        anchor = _git(repo, "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", "HEAD")
        source = "release_tag"
        reason = "Latest reachable release tag; fixed until the next release."
    else:
        initial = json.loads(anchor_file.read_text())
        anchor = initial["commit"]
        if not isinstance(anchor, str) or re.fullmatch(r"[0-9a-f]{40}", anchor) is None:
            raise ValueError("The initial performance anchor must be a full immutable commit SHA")
        if not isinstance(initial["reason"], str) or not initial["reason"].strip():
            raise ValueError("The initial performance anchor requires a reviewed reason")
        source = "initial_commit"
        reason = f"No reachable release tag. {initial['reason']}"

    evidence = {
        "current_commit": _verified_commit(repo, "HEAD"),
        "parent_commit": _verified_commit(repo, "HEAD^"),
        "anchor_commit": _verified_commit(repo, anchor),
        "anchor_ref": anchor,
        "anchor_source": source,
        "anchor_reason": reason,
    }
    workdir.mkdir(parents=True, exist_ok=True)
    for name, commit in (
        ("perf-base", evidence["parent_commit"]),
        ("perf-anchor", evidence["anchor_commit"]),
    ):
        _git(repo, "worktree", "add", "--detach", str(workdir / name), commit)
    (workdir / "perf-baselines.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main() -> None:
    """Prepare both baseline worktrees for the current checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    evidence = prepare_baselines(repo, args.workdir.resolve(), repo / "benchmarks/perf_anchor.json")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
