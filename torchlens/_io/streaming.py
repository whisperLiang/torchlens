"""Streaming bundle writer used during forward-pass out capture.

This module implements the strict streaming writer behind
``trace(save_outs_to=...)``. It writes one safetensors blob
per saved out into a temporary bundle during the forward pass, then
finalizes ``manifest.json`` and ``metadata.pkl`` at postprocess time so the
returned log can stay memory-backed or disk-backed with the same on-disk
bundle.
"""

from __future__ import annotations

import os
import pickle
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .. import __version__ as TORCHLENS_VERSION
from .._state import pause_logging
from . import TLSPEC_VERSION, TorchLensIOError
from ._durability import fsync_dir, fsync_tree
from .manifest import Manifest, TensorEntry, sha256_of_file
from .scrub import BlobSpec, dump_canonical_metadata
from .tensor_policy import FailReason, Ok, SkipReason, is_supported_for_save
from .tlspec import _TlSpecWriter

PARTIAL_SENTINEL = "PARTIAL"
REASON_SENTINEL = "REASON.txt"
_BLOB_TENSOR_KEY = "data"


def _restrict_mode(path: Path, mode: int) -> None:
    """Best-effort tighten a streamed bundle path's permissions (POSIX only).

    Twin of ``_io/bundle.py::_restrict_mode``: ``mkdir``/``open`` honor the
    ambient umask, so under umask 002 the streaming bundle dir and its metadata
    sidecars were left group-writable while the core bundle writer tightens
    them. Best-effort: a filesystem that ignores mode bits is not a save
    failure.

    Parameters
    ----------
    path:
        Bundle directory or file to tighten.
    mode:
        Target permission bits (``0o700`` for directories, ``0o600`` for files).
    """

    if os.name != "posix":
        return
    try:
        path.chmod(mode)
    except OSError:
        pass


def next_blob_id(blob_index: int) -> str:
    """Return the canonical zero-padded blob id for one monotonic counter.

    Parameters
    ----------
    blob_index:
        One-based blob counter.

    Returns
    -------
    str
        Zero-padded blob id.
    """

    return f"{blob_index:010d}"


class BundleStreamWriter:
    """Persist out blobs incrementally into a temp TorchLens bundle.

    Parameters
    ----------
    path:
        Final bundle directory path.
    strict:
        Streaming bundles are always strict. Passing ``False`` is rejected.
    """

    def __init__(self, path: str | Path, *, strict: bool = True) -> None:
        """Create the temp bundle directory used for streaming writes.

        Parameters
        ----------
        path:
            Final bundle directory path.
        strict:
            Streaming bundles are always strict. Passing ``False`` is rejected.

        Raises
        ------
        TorchLensIOError
            If the target path is invalid or the temp directory cannot be created.
        """

        if not strict:
            raise TorchLensIOError("Streaming out save is always strict.")

        self.final_path = Path(path)
        if self.final_path.is_symlink():
            raise TorchLensIOError(f"Refusing symlinked save target: {self.final_path}.")
        if self.final_path.exists():
            raise TorchLensIOError(f"Bundle path already exists: {self.final_path}")

        self.tmp_path = self.final_path.parent / f"{self.final_path.name}.tmp.{uuid.uuid4().hex}"
        self.blobs_path = self.tmp_path / "blobs"
        self._blob_counter = 0
        self._closed = False
        self._finalized = False
        self._tensor_entries: list[TensorEntry] = []
        self._entries_by_blob_id: dict[str, TensorEntry] = {}

        try:
            self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
            self.tmp_path.mkdir()
            self.blobs_path.mkdir()
            # Permission parity with tl.save (B8-10): mkdir honors the ambient
            # umask, so under umask 002 the streaming bundle dir and its blobs/
            # dir were left group-writable/readable while the core bundle
            # writer tightens them to 0700. The rename at finalize preserves
            # tmp_path's mode, so tightening here also tightens the published
            # bundle directory.
            _restrict_mode(self.tmp_path, 0o700)
            _restrict_mode(self.blobs_path, 0o700)
        except OSError as exc:
            raise TorchLensIOError(
                f"Failed to create streaming temp bundle at {self.tmp_path}."
            ) from exc

    def next_blob_id(self) -> str:
        """Return the next monotonic blob id for this writer.

        Returns
        -------
        str
            Zero-padded blob id.
        """

        self._blob_counter += 1
        return next_blob_id(self._blob_counter)

    def write_blob(
        self,
        blob_id: str,
        tensor: torch.Tensor,
        *,
        kind: str,
        label: str,
    ) -> TensorEntry:
        """Write one tensor blob into ``blobs/`` and record its manifest entry.

        Parameters
        ----------
        blob_id:
            Opaque zero-padded blob identifier.
        tensor:
            Tensor payload to persist.
        kind:
            Logical tensor kind.
        label:
            Human-readable or provisional label for the tensor owner.

        Returns
        -------
        TensorEntry
            Recorded manifest entry.

        Raises
        ------
        TorchLensIOError
            If the tensor is unsupported or writing fails.
        """

        self._ensure_writable()
        if not isinstance(tensor, torch.Tensor):
            transform_name = "grad_transform" if kind == "grad" else "activation_transform"
            reason = (
                f"Streaming {kind} save requires {transform_name} outputs to be torch.Tensor "
                f"instances, but blob_id={blob_id} ({label}) received {type(tensor).__name__}."
            )
            self.abort(reason)
            raise TorchLensIOError(reason)

        decision = is_supported_for_save(tensor, strict=True)
        if not isinstance(decision, Ok):
            if isinstance(decision, (SkipReason, FailReason)):
                reason_text = decision.text
            else:
                reason_text = "unsupported tensor"
            reason = (
                f"Unsupported tensor for streaming {kind} save at {label} "
                f"(blob_id={blob_id}, kind={kind}): {reason_text}"
            )
            self.abort(reason)
            raise TorchLensIOError(reason)
        if blob_id in self._entries_by_blob_id:
            reason = f"Duplicate streaming blob_id={blob_id} for {label}."
            self.abort(reason)
            raise TorchLensIOError(reason)

        try:
            entry = self._write_tensor_blob(blob_id=blob_id, tensor=tensor, kind=kind, label=label)
        except OSError as exc:
            reason = f"Failed to write streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc
        except BaseException as exc:
            # Safety-net catch-all mirroring bundle.py's ``save()`` handler
            # (round-8 F3): a hand-enumerated except clause can always miss
            # the next not-yet-seen failure shape (e.g. a bare ``KeyError``
            # from ``safetensors.torch.save_file()`` for an allow-listed-but-
            # actually-unwritable dtype, cert round 8 BLOCKER) or a
            # KeyboardInterrupt/SystemExit/GeneratorExit unwinding mid-write.
            # This guarantees the ``.tmp`` dir is always marked PARTIAL --
            # and thus sweepable by ``cleanup_tmp()`` -- for any failure,
            # known or not, while re-raising non-``Exception``
            # ``BaseException``s unwrapped so control-flow semantics are
            # preserved.
            reason = f"Failed to write streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise TorchLensIOError(reason) from exc
            raise

        self._tensor_entries.append(entry)
        self._entries_by_blob_id[blob_id] = entry
        return entry

    def finalize(
        self,
        scrubbed_state: dict[str, Any],
        blob_specs: list[BlobSpec],
        unsupported: list[dict[str, str]],
        *,
        trace: Any,
    ) -> Path:
        """Finish the bundle by writing remaining blobs, manifest, and metadata.

        Parameters
        ----------
        scrubbed_state:
            Portable scrubbed metadata state.
        blob_specs:
            Remaining blob specs that were not already streamed during the pass.
        unsupported:
            Unsupported tensor records for the manifest.
        trace:
            Source ``Trace`` being streamed to disk. Used to write the same
            unified ``.tlspec`` manifest fields (``kind``, ``model_signature``,
            ``sites``, ``body_index``, ...) that ``Trace.save()``/``tl.save()``
            write, so streaming bundles are detected as ``"v2.0_unified"`` and
            go through the same ``validate_tlspec()`` schema validation.

        Returns
        -------
        Path
            Final bundle directory path.

        Raises
        ------
        TorchLensIOError
            If finalization fails.
        """

        self._ensure_writable()
        try:
            for blob_id, tensor, kind, label in blob_specs:
                if blob_id in self._entries_by_blob_id:
                    continue
                self.write_blob(blob_id, tensor, kind=kind, label=label)

            legacy_manifest = self._build_manifest(
                scrubbed_state=scrubbed_state, unsupported=unsupported, trace=trace
            )
            _TlSpecWriter.write_trace_manifest(
                path=self.tmp_path / "manifest.json",
                trace=trace,
                legacy_manifest=legacy_manifest,
                save_level="portable",
            )
            _restrict_mode(self.tmp_path / "manifest.json", 0o600)
            with (self.tmp_path / "metadata.pkl").open("wb") as handle:
                # B3R4-R21-2: canonical container bytes (set/frozenset members
                # sorted); persisted metadata must not vary with PYTHONHASHSEED.
                dump_canonical_metadata(scrubbed_state, handle)
            _restrict_mode(self.tmp_path / "metadata.pkl", 0o600)
        except TorchLensIOError:
            raise
        except (OSError, TypeError, ValueError, pickle.PickleError) as exc:
            # See torchlens/_io/bundle.py's ``save()`` handler: ``TypeError``
            # is included alongside ``pickle.PickleError`` because
            # ``pickle.dump()`` raises a bare ``TypeError`` (not the
            # ``PickleError`` subclass) for many live-resource objects.
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc
        except BaseException as exc:
            # Safety-net catch-all closing the same bug class as
            # bundle.py's ``save()`` (round-8 F3): a hand-enumerated except
            # tuple can always miss the next not-yet-discovered exception
            # shape, or a KeyboardInterrupt/SystemExit/GeneratorExit
            # unwinding mid-finalize (e.g. during ``pickle.dump()``).
            # Guarantees the ``.tmp`` dir is always marked PARTIAL -- and
            # thus sweepable by ``cleanup_tmp()`` -- for any failure, while
            # re-raising non-``Exception`` ``BaseException``s unwrapped so
            # KeyboardInterrupt/SystemExit/GeneratorExit control flow is
            # preserved.
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise TorchLensIOError(reason) from exc
            raise

        # Crash-durability before publish: fsync every written blob/sidecar and
        # the staged directories so a power/OS crash after the rename below
        # cannot publish a final-named bundle holding zero-length/partial files
        # with no PARTIAL sentinel (cleanup_tmp would never sweep it). Mirrors
        # the tl.save writer (_io/bundle.py:568-586).
        try:
            fsync_tree(self.tmp_path)
        except OSError as exc:
            reason = f"Failed to flush streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc

        # Re-check target absence at finalize, not just at __init__ (R59 TOCTOU):
        # a streaming writer never overwrites, but a concurrent writer could have
        # created ``final_path`` after the start-of-stream check. A bare rename
        # would then replace an empty concurrent target or surface a confusing
        # ENOTEMPTY; refuse it cleanly instead. (The narrow residual window
        # between this check and the rename cannot be closed without an atomic
        # exclusive-directory create, matching the other writers.)
        if self.final_path.exists():
            reason = f"Bundle path already exists: {self.final_path}"
            self.abort(reason)
            raise TorchLensIOError(reason)
        try:
            self.tmp_path.rename(self.final_path)
        except OSError as exc:
            reason = f"Failed to atomically rename {self.tmp_path} to {self.final_path}."
            self.abort(reason)
            raise TorchLensIOError(reason) from exc
        # Make the rename itself durable before declaring the save complete.
        fsync_dir(self.final_path.parent)

        self._closed = True
        self._finalized = True
        return self.final_path

    def abort(self, reason: str) -> None:
        """Mark the temp bundle as partial and stop accepting writes.

        Parameters
        ----------
        reason:
            Human-readable failure reason written to ``REASON.txt``.
        """

        if self._finalized:
            return
        self._closed = True
        self._mark_partial(reason)

    def relabel_blob(self, blob_id: str, label: str) -> None:
        """Update the manifest label for an already-written blob.

        Parameters
        ----------
        blob_id:
            Blob identifier to relabel.
        label:
            Final human-readable label.
        """

        entry = self._entries_by_blob_id.get(blob_id)
        if entry is None:
            return
        updated_entry = TensorEntry(
            blob_id=entry.blob_id,
            kind=entry.kind,
            label=label,
            relative_path=entry.relative_path,
            backend=entry.backend,
            shape=entry.shape,
            dtype=entry.dtype,
            device_at_save=entry.device_at_save,
            layout=entry.layout,
            bytes=entry.bytes,
            sha256=entry.sha256,
            requires_grad=entry.requires_grad,
            logical_backend=entry.logical_backend,
            codec=entry.codec,
            logical_dtype=entry.logical_dtype,
            logical_device=entry.logical_device,
            transport_backend=entry.transport_backend,
            transport_dtype=entry.transport_dtype,
            codec_metadata=entry.codec_metadata,
        )
        self._entries_by_blob_id[blob_id] = updated_entry
        for index, existing_entry in enumerate(self._tensor_entries):
            if existing_entry.blob_id == blob_id:
                self._tensor_entries[index] = updated_entry
                break

    def get_entry(self, blob_id: str) -> TensorEntry:
        """Return the manifest entry recorded for one blob id.

        Parameters
        ----------
        blob_id:
            Blob identifier to look up.

        Returns
        -------
        TensorEntry
            Recorded manifest entry.

        Raises
        ------
        TorchLensIOError
            If the blob id is unknown.
        """

        if blob_id not in self._entries_by_blob_id:
            raise TorchLensIOError(f"Streaming bundle is missing blob_id={blob_id}.")
        return self._entries_by_blob_id[blob_id]

    def _write_tensor_blob(
        self,
        *,
        blob_id: str,
        tensor: torch.Tensor,
        kind: str,
        label: str,
    ) -> TensorEntry:
        """Write one supported tensor blob and return its manifest entry."""

        with pause_logging():
            contiguous_tensor = tensor.resolve_conj().resolve_neg().contiguous()
        relative_path = Path("blobs") / f"{blob_id}.safetensors"
        blob_path = self.tmp_path / relative_path
        save_file({_BLOB_TENSOR_KEY: contiguous_tensor}, str(blob_path))
        return TensorEntry(
            blob_id=blob_id,
            kind=kind,
            label=label,
            relative_path=relative_path.as_posix(),
            backend="safetensors",
            shape=[int(dim) for dim in contiguous_tensor.shape],
            dtype=str(contiguous_tensor.dtype).replace("torch.", ""),
            device_at_save=str(tensor.device),
            layout=str(contiguous_tensor.layout).replace("torch.", ""),
            bytes=int(contiguous_tensor.numel() * contiguous_tensor.element_size()),
            sha256=sha256_of_file(blob_path),
            requires_grad=bool(tensor.requires_grad),
        )

    def _build_manifest(
        self,
        *,
        scrubbed_state: dict[str, Any],
        unsupported: list[dict[str, str]],
        trace: Any,
    ) -> Manifest:
        """Build the final manifest for the streamed bundle."""

        tensor_entries = list(self._tensor_entries)
        n_out_blobs = sum(1 for entry in tensor_entries if entry.kind == "out")
        n_grad_blobs = sum(1 for entry in tensor_entries if entry.kind == "grad")
        n_auxiliary_blobs = len(tensor_entries) - n_out_blobs - n_grad_blobs
        layer_list = scrubbed_state.get("layer_list", [])
        n_layers = len(layer_list) if isinstance(layer_list, list) else 0
        # Disclose the harvested module-attribute channel (R62): the documented
        # invariant is that EVERY save writes a custom_attributes_disclosure
        # entry, but the streaming writer shipped the channel with none. The
        # streaming path persists custom_attributes with the same default as
        # tl.save (include_custom_attributes=True), so it is reported included.
        from .bundle import _custom_attributes_disclosure

        return Manifest(
            tlspec_version=TLSPEC_VERSION,
            torchlens_version=TORCHLENS_VERSION,
            torch_version=torch.__version__,
            python_version=(
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            platform=f"{platform.system().lower()}-{platform.machine().lower()}",
            created_at=datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            bundle_format="directory",
            n_layers=n_layers,
            n_out_blobs=n_out_blobs,
            n_grad_blobs=n_grad_blobs,
            n_auxiliary_blobs=n_auxiliary_blobs,
            tensors=tensor_entries,
            unsupported_tensors=unsupported,
            custom_attributes_disclosure=_custom_attributes_disclosure(trace, included=True),
        )

    def _ensure_writable(self) -> None:
        """Raise if the writer has already been closed."""

        if self._closed:
            raise TorchLensIOError("Streaming bundle writer is already closed.")

    def _mark_partial(self, reason: str) -> None:
        """Best-effort write the partial sentinel and human-readable reason."""

        try:
            if self.tmp_path.exists():
                (self.tmp_path / PARTIAL_SENTINEL).write_text("", encoding="utf-8")
                (self.tmp_path / REASON_SENTINEL).write_text(reason, encoding="utf-8")
        except OSError:
            return
