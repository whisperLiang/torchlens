"""Small shared repository-inspection helpers for crawler acceptance tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from menagerie.crawler.tests.test_slice_c_envs_lifecycle import _release_lock_provenance_errors


def repository_root() -> Path:
    """Return the checked-out repository root containing this acceptance package."""

    return Path(__file__).resolve().parents[2]


def tracked_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return every Git-tracked path without consulting ignored runtime files."""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return tuple(Path(value.decode("utf-8")) for value in completed.stdout.split(b"\0") if value)


def crawler_lock_provenance_errors(repo_root: Path) -> tuple[str, ...]:
    """Validate release provenance against tracked files, excluding local solve outputs.

    Parameters
    ----------
    repo_root:
        Checked-out repository whose committed lock families should be inspected.

    Returns
    -------
    tuple[str, ...]
        Strict lifecycle-validator diagnostics, empty for fully attested release locks.
    """

    env_relative = Path("menagerie/crawler/envs")
    env_paths = tuple(
        path for path in tracked_paths(repo_root) if path.is_relative_to(env_relative)
    )
    missing = tuple(
        f"{repo_root / path}:missing-tracked-file"
        for path in env_paths
        if not (repo_root / path).is_file()
    )
    if missing:
        return missing
    # Untracked companions cannot attest a committed lock. Conversely, ignored
    # target-local solve outputs must not become apparent release artifacts.
    with TemporaryDirectory(prefix="crawler-tracked-locks-") as snapshot:
        snapshot_root = Path(snapshot)
        for path in env_paths:
            target = snapshot_root / path.relative_to(env_relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo_root / path, target)
        return tuple(
            error.replace(str(snapshot_root), str(repo_root / env_relative), 1)
            for error in _release_lock_provenance_errors(snapshot_root)
        )
