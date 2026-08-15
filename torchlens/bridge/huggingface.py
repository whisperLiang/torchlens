"""Hugging Face Hub publishing helpers."""

from __future__ import annotations

import io
import pickle
import tarfile
import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_PATH_IN_REPO = "torchlens_artifact.pkl"
_DEFAULT_TARBALL_PATH_IN_REPO = "torchlens_artifact.tar.gz"


def push_to_hub(
    log_or_bundle_or_spec: Any,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool | None = None,
    path_in_repo: str = _DEFAULT_PATH_IN_REPO,
    commit_message: str = "Add TorchLens artifact",
    create_repo: bool = True,
    dry_run: bool = False,
    api: Any | None = None,
    save_level: str = "portable",
) -> dict[str, Any]:
    """Upload a real TorchLens artifact to the Hugging Face Hub.

    Parameters
    ----------
    log_or_bundle_or_spec:
        ``Trace``, ``Bundle``, or ``InterventionSpec``-like object to publish.
    repo_id:
        Target Hugging Face repository ID.
    token:
        Optional Hub token.
    private:
        Optional repository privacy flag used when creating the repo.
    path_in_repo:
        Destination filename inside the repository.
    commit_message:
        Commit message for the upload.
    create_repo:
        Whether to create the repo before upload.
    dry_run:
        If True, serialize locally and return planned upload metadata without
        contacting the Hub.
    api:
        Optional ``HfApi``-compatible object for tests or advanced callers.
    save_level:
        Public ``.tlspec`` save level used when the artifact must be
        serialized through the portable-bundle scrub path (``"audit"``,
        ``"executable_with_callables"``, or ``"portable"``). Only consulted
        when ``log_or_bundle_or_spec`` is a ``Trace``/``Bundle`` that cannot
        be pickled directly (see ``_artifact_bytes``).

    Returns
    -------
    dict[str, Any]
        Upload metadata including ``repo_id``, ``path_in_repo``, and ``format``
        (``"pickle"`` or ``"tar.gz"``). When the artifact must be packed as a
        gzipped tar bundle and the caller left ``path_in_repo`` at its default,
        the destination name is switched to a ``.tar.gz`` extension so a bundle
        is never uploaded under a ``.pkl`` name.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is unavailable.
    TorchLensIOError
        If the artifact cannot be serialized at all. ``push_to_hub`` never
        silently substitutes a metadata-only stub for genuine artifact
        content -- either the real artifact is uploaded, or this raises.
    """

    if api is None and not dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "Hugging Face publishing requires the `hf` extra: install torchlens[hf]."
            ) from exc
        api = HfApi(token=token)

    payload, artifact_format = _artifact_bytes(log_or_bundle_or_spec, save_level=save_level)
    if artifact_format == "tar.gz" and path_in_repo == _DEFAULT_PATH_IN_REPO:
        path_in_repo = _DEFAULT_TARBALL_PATH_IN_REPO

    if dry_run:
        return {
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "size_bytes": len(payload),
            "format": artifact_format,
            "dry_run": True,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / Path(path_in_repo).name
        artifact_path.write_bytes(payload)
        size_bytes = artifact_path.stat().st_size
        if api is None:
            raise RuntimeError("A Hugging Face API object is required when dry_run=False.")
        if create_repo:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        upload_result = api.upload_file(
            path_or_fileobj=str(artifact_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )

    return {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "size_bytes": size_bytes,
        "format": artifact_format,
        "dry_run": False,
        "upload_result": upload_result,
    }


def _artifact_bytes(
    log_or_bundle_or_spec: Any, *, save_level: str = "portable"
) -> tuple[bytes, str]:
    """Serialize an artifact for Hub upload and report its serialized format.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Artifact object.
    save_level:
        Public ``.tlspec`` save level used for the portable-bundle fallback
        path (see below).

    Returns
    -------
    tuple[bytes, str]
        The serialized bytes and a format tag. A ``Trace``/``Bundle`` -- which
        raw ``pickle.dumps`` would embed ``$HOME``, the username, and absolute
        source/bundle/visualizer paths into (R62-1: a privacy leak to the one
        PUBLIC sharing surface) -- is ALWAYS serialized through the same real,
        privacy-scrubbed ``.tlspec`` portable bundle path used by
        :func:`torchlens.save`/``Bundle.save`` and returned as a gzipped tar
        archive with the tag ``"tar.gz"``. The scrub is real, not a metadata
        stand-in. Only an object with NO portable-bundle save path (e.g. a plain
        dict of user data the caller chose to push) is pickled directly and
        tagged ``"pickle"``.

    Raises
    ------
    TorchLensIOError
        If the artifact cannot be serialized at all (the portable-bundle path
        itself fails, or a no-bundle object fails to pickle).
    """

    from .._io import TorchLensIOError

    # Privacy-first (R62-1): any artifact with a portable-bundle save path
    # (Trace/Bundle) is serialized through the SCRUBBED bundle path, never raw
    # ``pickle.dumps`` -- raw pickle leaks $HOME, the username, and absolute
    # source paths to a public hub. Raw pickle is reserved for objects that have
    # no bundle save path and thus carry no scrubbable TorchLens internals.
    saver = _resolve_bundle_saver(log_or_bundle_or_spec)
    if saver is not None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                saver(bundle_dir, level=save_level, overwrite=True)
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                    tar.add(bundle_dir, arcname=bundle_dir.name)
                return buffer.getvalue(), "tar.gz"
        except Exception as bundle_error:
            raise TorchLensIOError(
                "push_to_hub could not serialize this "
                f"{type(log_or_bundle_or_spec).__name__} artifact for upload through "
                f"the privacy-scrubbed portable `.tlspec` bundle path ({bundle_error!r}). "
                "Refusing to fall back to raw pickle (which would leak local paths) or "
                "to silently upload a metadata-only stub instead of the real artifact."
            ) from bundle_error

    try:
        return pickle.dumps(log_or_bundle_or_spec), "pickle"
    except Exception as direct_pickle_error:
        raise TorchLensIOError(
            "push_to_hub could not serialize this "
            f"{type(log_or_bundle_or_spec).__name__} artifact for upload "
            f"({direct_pickle_error!r}) and it has no portable `.tlspec` bundle "
            "save path to fall back to. Refusing to silently upload a metadata-only "
            "stub instead of the real artifact."
        ) from direct_pickle_error


def _resolve_bundle_saver(log_or_bundle_or_spec: Any) -> Any | None:
    """Return a ``save(path, *, level, overwrite)`` callable for real objects.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Candidate artifact object.

    Returns
    -------
    Any | None
        A bound ``save`` callable matching the ``.tlspec`` bundle contract
        (``Trace``/``Bundle`` both expose one), or ``None`` if the object has
        no known portable-bundle save path.
    """

    from ..data_classes.trace import Trace

    if isinstance(log_or_bundle_or_spec, Trace):
        from .._io.bundle import save as _save_trace_bundle

        def _save_trace(path: Path, *, level: str, overwrite: bool) -> None:
            """Save the captured ``Trace`` as a portable ``.tlspec`` bundle.

            Parameters
            ----------
            path:
                Destination bundle directory.
            level:
                Public ``.tlspec`` save level.
            overwrite:
                Whether to overwrite an existing bundle at ``path``.
            """

            _save_trace_bundle(log_or_bundle_or_spec, path, level=level, overwrite=overwrite)

        return _save_trace

    from ..bundle import Bundle

    if isinstance(log_or_bundle_or_spec, Bundle):
        return log_or_bundle_or_spec.save

    return None


__all__ = ["push_to_hub"]
