"""Filesystem capability helpers for tests that require real symbolic links."""

from pathlib import Path

import pytest


def symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    """Create a real symbolic link, disclosing a missing Windows privilege.

    Parameters
    ----------
    link
        Symbolic link to create in the test's private directory.
    target
        File or directory the link should reference.
    target_is_directory
        Whether the target is a directory, as required by Windows.
    """
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symbolic-link creation privilege is unavailable")
        raise
