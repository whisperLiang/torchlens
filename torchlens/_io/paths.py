"""Path validation helpers for portable TorchLens bundles."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from . import TorchLensIOError

ExceptionType: TypeAlias = type[Exception]


def reject_symlink_path(
    path: Path,
    *,
    context: str,
    exc_type: ExceptionType = TorchLensIOError,
    message_prefix: str = "Refusing symlinked",
    trailing_period: bool = True,
) -> None:
    """Reject symlink paths with a caller-chosen exception policy.

    Parameters
    ----------
    path:
        Path to validate.
    context:
        Human-readable path role included in the exception message.
    exc_type:
        Exception type raised when ``path`` is a symlink.
    message_prefix:
        Prefix text used before the contextual path description.
    trailing_period:
        Whether to end the message with ``"."``.

    Returns
    -------
    None
        Returns when ``path`` is not a symlink.

    Raises
    ------
    Exception
        Raised as ``exc_type`` when ``path`` is a symlink.
    """

    if path.is_symlink():
        suffix = "." if trailing_period else ""
        raise exc_type(f"{message_prefix} {context}: {path}{suffix}")


def resolve_bundle_blob_path(bundle_root: Path, relative_path: str) -> Path:
    """Resolve one manifest-supplied blob path under ``<bundle>/blobs``.

    Parameters
    ----------
    bundle_root:
        Root directory of the portable bundle.
    relative_path:
        Manifest-provided blob path relative to the bundle root.

    Returns
    -------
    Path
        Absolute resolved blob path.

    Raises
    ------
    TorchLensIOError
        If the path is absolute, contains ``".."``, resolves outside the
        bundle's ``blobs/`` directory, or the ``blobs/`` directory itself is a
        symlink.
    """

    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise TorchLensIOError(f"Bundle rejected absolute relative_path {relative_path!r}.")
    if ".." in candidate_path.parts:
        raise TorchLensIOError(
            f"Bundle rejected parent traversal in relative_path {relative_path!r}."
        )

    blobs_dir = bundle_root / "blobs"
    if blobs_dir.is_symlink():
        # Per-file symlink checks are performed on the resolved blob path, so a
        # symlinked blobs/ DIRECTORY would otherwise silently redirect every
        # "real file" containment check into an attacker-chosen tree.
        raise TorchLensIOError(f"Refusing symlinked blobs directory: {blobs_dir}.")
    candidate = (bundle_root / candidate_path).resolve()
    allowed_root = blobs_dir.resolve()
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise TorchLensIOError(
            f"Bundle rejected path traversal outside blobs/: {relative_path!r}."
        ) from exc
    return candidate
