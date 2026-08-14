"""Load and recover fastlog directory bundles."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors

from .._io import TorchLensIOError
from .._io._json import loads_bounded, read_bounded, read_bytes_bounded
from .._io.manifest import Manifest, enforce_version_policy
from .._io.paths import resolve_bundle_blob_path
from .exceptions import BundleNotFinalizedError, RecoveryError
from .storage_disk import record_from_json
from .storage_ram import RamStorageBackend
from .types import ActivationRecord, Recording


def load(path: str | Path) -> Recording:
    """Load a finalized fastlog bundle.

    Parameters
    ----------
    path:
        Fastlog bundle directory.

    Returns
    -------
    Recording
        Loaded recording with ``recovered=False``.

    Raises
    ------
    BundleNotFinalizedError
        If the bundle has no valid manifest.
    TorchLensIOError
        If the finalized bundle is malformed.
    """

    bundle_path = Path(path)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise BundleNotFinalizedError("fastlog bundle is partial; use tl.fastlog.recover()")
    try:
        manifest = Manifest.read(manifest_path)
    except TorchLensIOError as exc:
        raise BundleNotFinalizedError(
            "fastlog bundle manifest is invalid; use tl.fastlog.recover()"
        ) from exc
    _validate_fastlog_layout(bundle_path, manifest)
    return _load_from_index(
        bundle_path,
        recovered=False,
        recovery_warnings=[],
        strict_integrity=True,
    )


def recover(path: str | Path) -> Recording:
    """Recover a finalized or partial fastlog bundle.

    Parameters
    ----------
    path:
        Fastlog bundle directory or streaming temp directory.

    Returns
    -------
    Recording
        Loaded or recovered recording.

    Raises
    ------
    RecoveryError
        If no recoverable index exists.
    """

    bundle_path = Path(path)
    manifest_path = bundle_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = Manifest.read(manifest_path)
            _validate_fastlog_layout(bundle_path, manifest)
        except TorchLensIOError:
            pass
        else:
            return _load_from_index(
                bundle_path,
                recovered=False,
                recovery_warnings=[],
                strict_integrity=False,
            )

    index_path = bundle_path / "fastlog_index.jsonl"
    if not index_path.exists():
        raise RecoveryError("no recoverable index")
    return _load_from_index(
        bundle_path,
        recovered=True,
        recovery_warnings=[],
        strict_integrity=False,
    )


def _load_from_index(
    bundle_path: Path,
    *,
    recovered: bool,
    recovery_warnings: list[str],
    strict_integrity: bool,
) -> Recording:
    """Load records by scanning ``fastlog_index.jsonl``."""

    metadata = _read_metadata(bundle_path / "metadata.json")
    records: list[ActivationRecord] = []
    collector = _RecoveryWarningCollector(recovery_warnings)
    lines = _read_index_lines(bundle_path / "fastlog_index.jsonl")
    for line_number, raw_line in enumerate(lines, start=1):
        try:
            # Bounded parse (B8-13): a depth-bomb or oversized index line must
            # refuse as a JSONDecodeError handled below, never escape recovery
            # as a raw RecursionError from the stdlib decoder.
            data = loads_bounded(raw_line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                collector.add("truncated tail", "truncated tail")
            else:
                collector.add("malformed line", f"malformed line {line_number}")
            continue
        if not isinstance(data, dict):
            collector.add("malformed line", f"malformed line {line_number}")
            continue
        record = record_from_json(data)
        blob_recoverable, validated_payloads = _blob_is_recoverable(
            bundle_path,
            record,
            collector,
        )
        if not blob_recoverable:
            continue
        rehydrated_record = _rehydrate_record_payloads(
            record,
            validated_payloads,
        )
        if rehydrated_record is None:
            continue
        records.append(rehydrated_record)
    warnings_out = collector.messages()
    if strict_integrity and warnings_out:
        raise TorchLensIOError(_format_strict_integrity_error(warnings_out))
    recording = _recording_from_records(
        records,
        bundle_path=bundle_path,
        metadata=metadata,
        recovered=recovered or bool(warnings_out),
        recovery_warnings=warnings_out,
    )
    RamStorageBackend(recording).finalize()
    return recording


class _RecoveryWarningCollector:
    """Deduplicating, bounded collector for recovery warnings.

    A corrupt index can carry hundreds of thousands of bad lines; retaining one
    string per problem made ``Recording.recovery_warnings`` grow without bound
    (B8-41: a 200k-line corruption held 15+ MB of warnings whose actionable
    content was one sentence). At most ``_MAX_PER_KIND`` messages are retained
    per warning kind; every further problem is counted and summarized by one
    final ``(+N more suppressed)`` entry so the total stays accurate.
    """

    _MAX_PER_KIND = 20

    def __init__(self, seed: list[str] | None = None) -> None:
        """Initialize the collector.

        Parameters
        ----------
        seed:
            Pre-existing warning messages to retain verbatim (uncounted).
        """

        self._messages: list[str] = list(seed) if seed else []
        self._counts: dict[str, int] = {}
        self._suppressed_total = 0

    def add(self, kind: str, message: str) -> None:
        """Record one recovery problem, retaining at most ``_MAX_PER_KIND`` per kind.

        Parameters
        ----------
        kind:
            Stable warning kind used for dedup bucketing (for example
            ``"malformed line"`` or ``"hash mismatch"``).
        message:
            Full human-readable warning message.
        """

        count = self._counts.get(kind, 0) + 1
        self._counts[kind] = count
        if count <= self._MAX_PER_KIND:
            self._messages.append(message)
        else:
            self._suppressed_total += 1

    def messages(self) -> list[str]:
        """Return the bounded warning list, with one suppression-summary tail.

        Returns
        -------
        list[str]
            Retained warning messages, plus a final ``(+N more suppressed)``
            entry when any problems exceeded the per-kind retention cap.
        """

        if self._suppressed_total:
            return [*self._messages, f"(+{self._suppressed_total} more suppressed)"]
        return list(self._messages)


def _format_strict_integrity_error(recovery_warnings: list[str]) -> str:
    """Return a finalized-load integrity error message.

    Parameters
    ----------
    recovery_warnings:
        Recovery diagnostics accumulated while scanning the JSONL index.

    Returns
    -------
    str
        Human-readable error explaining why a finalized bundle load refused to
        continue.
    """

    if not recovery_warnings:
        return "Finalized fastlog bundle failed integrity validation."
    primary = recovery_warnings[0]
    if len(recovery_warnings) == 1:
        return (
            "Finalized fastlog bundle failed integrity validation: "
            f"{primary}. Use tl.fastlog.recover() to inspect salvageable records."
        )
    return (
        "Finalized fastlog bundle failed integrity validation: "
        f"{primary} (and {len(recovery_warnings) - 1} more issue(s)). "
        "Use tl.fastlog.recover() to inspect salvageable records."
    )


def _read_index_lines(path: Path) -> list[str]:
    """Read index lines from a recoverable fastlog index file.

    The raw read is size-bounded (B8-13): ``Path.read_text()`` materialized the
    whole attacker-sized index (plus a second full copy from ``splitlines()``)
    before any ceiling could apply. ``read_bytes_bounded`` refuses an oversized
    index before allocating it, and the refusal is converted into the module's
    typed ``RecoveryError`` channel rather than escaping raw.
    """

    try:
        raw = read_bytes_bounded(path)
    except json.JSONDecodeError as exc:
        raise RecoveryError("fastlog index exceeds the maximum recoverable size") from exc
    except OSError as exc:
        raise RecoveryError("no recoverable index") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryError("fastlog index is not valid UTF-8") from exc
    del raw
    return text.splitlines()


def _blob_is_recoverable(
    bundle_path: Path,
    record: ActivationRecord,
    recovery_warnings: _RecoveryWarningCollector,
) -> tuple[bool, dict[str, Any]]:
    """Return whether a record's blob(s) are present and hash-valid.

    Both the raw out blob and the transformed out blob are
    validated when their metadata is present. A missing or hash-mismatched
    blob disqualifies the record. Successful validation also returns the
    materialized payloads so the caller does not have to read the file again.
    """

    raw_recoverable, raw_payload = _validate_blob_metadata(
        bundle_path,
        record.metadata.get("blob_id"),
        record.metadata.get("relative_path"),
        record.metadata.get("sha256"),
        recovery_warnings,
    )
    if not raw_recoverable:
        return False, {}
    transformed_recoverable, transformed_payload = _validate_blob_metadata(
        bundle_path,
        record.metadata.get("transformed_out_blob_id"),
        record.metadata.get("transformed_out_relative_path"),
        record.metadata.get("transformed_out_sha256"),
        recovery_warnings,
    )
    if not transformed_recoverable:
        return False, {}
    return True, {
        "disk_payload": raw_payload,
        "transformed_disk_payload": transformed_payload,
    }


def _validate_blob_metadata(
    bundle_path: Path,
    blob_id: Any,
    relative_path: Any,
    expected_sha256: Any,
    recovery_warnings: _RecoveryWarningCollector,
) -> tuple[bool, Any | None]:
    """Validate a single blob entry from record metadata."""

    if blob_id is None and relative_path is None and expected_sha256 is None:
        return True, None
    if blob_id is None or relative_path is None or expected_sha256 is None:
        warning_id = "unknown"
        if blob_id is not None:
            warning_id = str(blob_id)
        elif relative_path is not None:
            warning_id = str(relative_path)
        recovery_warnings.add("incomplete blob metadata", f"incomplete blob metadata {warning_id}")
        return False, None
    try:
        blob_path = resolve_bundle_blob_path(bundle_path, str(relative_path))
    except TorchLensIOError:
        recovery_warnings.add("malformed blob path", f"malformed blob path {blob_id}")
        return False, None
    if not blob_path.exists():
        recovery_warnings.add("missing blob", f"missing blob {blob_id}")
        return False, None
    try:
        payload = _load_verified_blob_tensor(blob_path, str(expected_sha256))
    except TorchLensIOError:
        recovery_warnings.add("hash mismatch", f"hash mismatch {blob_id}")
        return False, None
    return True, payload


def _load_verified_blob_tensor(blob_path: Path, expected_sha256: str) -> Any:
    """Read, verify, and materialize one fastlog blob in a single pass."""

    try:
        payload = blob_path.read_bytes()
    except OSError as exc:
        raise TorchLensIOError(f"Failed to read fastlog blob at {blob_path}.") from exc
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise TorchLensIOError(f"Checksum mismatch for fastlog blob at {blob_path}.")
    return _load_blob_tensor_from_bytes(payload, blob_path)


def _load_blob_tensor_from_bytes(payload: bytes, blob_path: Path) -> Any:
    """Load the single tensor stored in one fastlog safetensors blob.

    Fastlog blobs are always written with exactly one tensor per file (see
    ``BundleStreamWriter._write_tensor_blob``), so the blob's sole value is the
    materialized payload; the storage key itself is not part of the public
    contract.

    Raises
    ------
    TorchLensIOError
        If the blob is missing, unreadable, or does not contain exactly one
        tensor.
    """

    try:
        tensor_map = load_safetensors(payload)
    except ImportError as exc:
        raise TorchLensIOError(
            "Fastlog bundle payload materialization requires the safetensors "
            "backend. Install safetensors>=0.4."
        ) from exc
    except (OSError, SafetensorError, ValueError) as exc:
        raise TorchLensIOError(f"Failed to materialize fastlog blob at {blob_path}.") from exc
    if len(tensor_map) != 1:
        raise TorchLensIOError(f"Expected a single tensor in fastlog blob file {blob_path}.")
    return next(iter(tensor_map.values()))


def _rehydrate_record_payloads(
    record: ActivationRecord,
    validated_payloads: dict[str, Any],
) -> ActivationRecord | None:
    """Rehydrate a reloaded record's disk-persisted tensor payloads.

    ``_blob_is_recoverable`` already confirmed both the raw and transformed
    blobs (when present) exist and are hash-valid, so a subsequent failure to
    read them back here reflects a genuine I/O problem (e.g. a race with
    concurrent bundle mutation) rather than a corruption this function should
    silently mask. Such a record is skipped with a recovery warning, matching
    the existing missing-blob/hash-mismatch skip-and-warn behavior in this
    module, instead of raising out of ``load()``/``recover()``.

    Returns
    -------
    ActivationRecord | None
        A record with ``disk_payload``/``transformed_disk_payload`` populated
        from their persisted blobs, or ``None`` if materialization failed for
        a blob that ``_blob_is_recoverable`` had already validated.
    """

    disk_payload = validated_payloads.get("disk_payload")
    transformed_disk_payload = validated_payloads.get("transformed_disk_payload")

    if disk_payload is None and transformed_disk_payload is None:
        return record
    return dataclasses.replace(
        record,
        disk_payload=disk_payload,
        transformed_disk_payload=transformed_disk_payload,
    )


def _recording_from_records(
    records: list[ActivationRecord],
    *,
    bundle_path: Path,
    metadata: dict[str, Any],
    recovered: bool,
    recovery_warnings: list[str],
) -> Recording:
    """Build a Recording around loaded records."""

    halted = bool(metadata.get("halted", False))
    status: Literal["complete", "halted", "partial_error", "recovered"] = (
        "recovered" if recovered else "halted" if halted else "complete"
    )
    return Recording(
        records=records,
        by_pass={},
        by_label={},
        by_address={},
        orphan_records=list(metadata.get("orphan_records", [])),
        bundle_path=bundle_path,
        n_ops=int(metadata.get("n_passes", metadata.get("n_ops", 1))),
        start_times=list(metadata.get("start_times", [])),
        end_times=list(metadata.get("end_times", [])),
        predicate_failures=[],
        predicate_failure_overflow_count=int(metadata.get("predicate_failure_overflow_count", 0)),
        halted=halted,
        halt_reason=metadata.get("halt_reason"),
        halts_by_pass={
            int(pass_index): str(reason)
            for pass_index, reason in dict(metadata.get("halts_by_pass", {})).items()
        },
        keep_op_repr=metadata.get("keep_op_repr"),
        history_size=int(metadata.get("history_size", 0)),
        _activation_transform_repr=metadata.get("_activation_transform_repr"),
        recovered=recovered,
        status=status,
        recovery_warnings=recovery_warnings,
    )


def _read_metadata(path: Path) -> dict[str, Any]:
    """Read optional fastlog metadata JSON.

    Bounded read/parse (B8-13): an oversized or depth-bomb ``metadata.json`` is
    corruption and degrades to the same graceful empty-metadata outcome as any
    other unreadable metadata file, instead of exhausting memory or escaping as
    a raw ``RecursionError`` from the stdlib decoder.
    """

    try:
        data = read_bounded(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _validate_fastlog_layout(bundle_path: Path, manifest: Manifest) -> None:
    """Validate the finalized fastlog directory layout.

    Notes
    -----
    This finalized-layout tripwire currently validates the required fastlog
    sidecar files by presence only. ``load()`` reconstructs the effective
    lookup indexes from ``fastlog_index.jsonl`` rather than trusting the
    sidecar contents directly; the sidecar authority contract remains an owner
    decision.
    """

    enforce_version_policy(manifest)
    if manifest.bundle_format != "fastlog-directory":
        raise TorchLensIOError("Expected fastlog-directory bundle format.")
    required = (
        "manifest.json",
        "fastlog_index.jsonl",
        "pass_index.json",
        "label_index.json",
        "metadata.json",
        "blobs",
    )
    for name in required:
        candidate = bundle_path / name
        if not candidate.exists():
            raise TorchLensIOError(f"Fastlog bundle is missing {name}.")
    if not (bundle_path / "blobs").is_dir():
        raise TorchLensIOError("Fastlog bundle blobs path is not a directory.")
