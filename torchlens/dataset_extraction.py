"""Batched dataset extraction with a self-describing, resumable disk artifact.

This module owns :func:`torchlens.extract_dataset`'s implementation. In disk
mode (``output_dir=``) the artifact directory is SELF-DESCRIBING: alongside the
``batch_XXXXX.pt`` shards it carries a ``manifest.json`` recording site
identity (layer label plus structural site key where derivable), stimulus
ordering and provenance, axis semantics, dtypes, devices, the transform
disclosure, and the TorchLens version — so the artifact can be handed to a
collaborator who never saw the producing script.

The manifest doubles as the RESUME ledger: shard writes are atomic (temp file
plus ``os.replace``) and the manifest is atomically rewritten after every
shard with that shard's exact row count, so a run killed mid-extraction can be
resumed with ``resume=True`` from the last completed shard. Resume-from-shard
was chosen over content-addressed caching because one-shot stimulus iterables
cannot be hashed without being consumed, and the shard layout is already the
public on-disk contract.

Every spelling introduced here (``resume=``, ``stimulus_ids=``,
:func:`load_extraction`, :class:`LoadedExtraction`, the manifest schema) is
DOCUMENTED-UNSTABLE pending the naming/UI sprint.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ._errors import _actionable_message, _ActionableErrorMixin
from ._io import _json
from .errors._base import ConfigurationError

#: Manifest schema identifier written to and required from ``manifest.json``.
MANIFEST_SCHEMA = "tl_extract_manifest_v1"

#: Filename of the self-describing manifest inside an extraction directory.
MANIFEST_FILENAME = "manifest.json"

#: Maximum number of tensor elements sampled into the stimulus digest.
_DIGEST_SAMPLE_ELEMENTS = 4096


class DatasetExtractionResumeError(_ActionableErrorMixin, ConfigurationError, RuntimeError):
    """Raised when a resumable extraction artifact cannot be safely continued."""

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize an actionable extraction-resume refusal.

        Parameters
        ----------
        problem:
            Description of the artifact state and why it was rejected.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


def _move_nested_to_device(value: Any, device: torch.device | str | None) -> Any:
    """Move tensors in a nested value to a device.

    Parameters
    ----------
    value:
        Tensor or nested Python container.
    device:
        Target device, or ``None`` to leave values unchanged.

    Returns
    -------
    Any
        Value with tensors moved to ``device``.
    """

    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    return value


def _collate_batch(items: list[Any]) -> Any:
    """Collate a small list of stimuli into one model input.

    Parameters
    ----------
    items:
        Stimulus items accumulated for one batch.

    Returns
    -------
    Any
        Batched tensor or nested container.
    """

    if not items:
        raise ValueError("Cannot collate an empty batch.")
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items)
    if isinstance(first, tuple):
        return tuple(_collate_batch([item[index] for item in items]) for index in range(len(first)))
    if isinstance(first, list):
        return [_collate_batch([item[index] for item in items]) for index in range(len(first))]
    if isinstance(first, dict):
        return {key: _collate_batch([item[key] for item in items]) for key in first}
    return items


def _iter_batches(stimuli: Any, batch_size: int) -> Iterable[Any]:
    """Yield batched model inputs from tensors or iterables.

    Parameters
    ----------
    stimuli:
        Tensor with batch dimension or iterable stimulus set.
    batch_size:
        Number of items per batch.

    Yields
    ------
    Any
        One batch suitable for ``model.forward``.
    """

    if isinstance(stimuli, torch.Tensor):
        for start in range(0, stimuli.shape[0], batch_size):
            yield stimuli[start : start + batch_size]
        return

    batch: list[Any] = []
    for item in stimuli:
        batch.append(item)
        if len(batch) == batch_size:
            yield _collate_batch(batch)
            batch = []
    if batch:
        yield _collate_batch(batch)


def _merge_batch_outputs(
    accumulator: dict[str, list[torch.Tensor]],
    batch_outputs: dict[str, torch.Tensor],
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
) -> None:
    """Append one batch of extracted outs to an accumulator.

    Parameters
    ----------
    accumulator:
        Mutable mapping from layer label to per-batch tensors.
    batch_outputs:
        Extraction output from one batch.
    transform:
        Optional transform applied to each out before storage.
    """

    for layer_name, tensor in batch_outputs.items():
        stored = transform(tensor) if transform is not None else tensor
        accumulator.setdefault(layer_name, []).append(stored.detach().cpu())


def _shard_filename(index: int) -> str:
    """Return the canonical shard filename for a batch index.

    Parameters
    ----------
    index:
        Zero-based batch index.

    Returns
    -------
    str
        Filename of the form ``batch_00042.pt``.
    """

    return f"batch_{index:05d}.pt"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document atomically (temp file plus ``os.replace``).

    Parameters
    ----------
    path:
        Final destination path.
    payload:
        JSON-serializable document.
    """

    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Save a torch payload atomically so a partial file never bears the final name.

    Parameters
    ----------
    payload:
        Object passed to ``torch.save``.
    path:
        Final destination path.
    """

    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _tensor_digest(stimuli: torch.Tensor) -> str:
    """Return a cheap sampled content digest for tensor stimuli.

    The digest hashes the shape, dtype, and up to ``_DIGEST_SAMPLE_ELEMENTS``
    strided elements. It catches accidental stimulus swaps on resume; it is not
    an adversarial integrity check.

    Parameters
    ----------
    stimuli:
        Stimulus tensor with a leading batch dimension.

    Returns
    -------
    str
        ``sha256:...`` digest string.
    """

    hasher = hashlib.sha256()
    hasher.update(repr(tuple(stimuli.shape)).encode())
    hasher.update(str(stimuli.dtype).encode())
    flat = stimuli.detach().reshape(-1)
    stride = max(1, flat.numel() // _DIGEST_SAMPLE_ELEMENTS)
    sample = flat[::stride].cpu().contiguous()
    hasher.update(sample.numpy().tobytes())
    return f"sha256:{hasher.hexdigest()}"


def _transform_signature(
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
) -> dict[str, str] | None:
    """Describe a transform callable for the manifest signature.

    Callable identity cannot be verified across processes; the qualified name
    is a DISCLOSURE (and a resume compatibility check), not a proof.

    Parameters
    ----------
    transform:
        Optional tensor transform supplied by the caller.

    Returns
    -------
    dict[str, str] | None
        ``{"module": ..., "qualname": ...}`` or ``None`` when no transform.
    """

    if transform is None:
        return None
    return {
        "module": getattr(transform, "__module__", "") or "",
        "qualname": getattr(
            transform, "__qualname__", getattr(type(transform), "__qualname__", "")
        ),
    }


def _stimuli_signature(stimuli: Any) -> dict[str, Any]:
    """Describe the stimulus set for the manifest signature.

    Parameters
    ----------
    stimuli:
        Tensor with a leading batch dimension or an iterable stimulus set.

    Returns
    -------
    dict[str, Any]
        Signature block. Tensor stimuli carry shape, dtype, and a sampled
        digest; iterable identity is disclosed as unverifiable.
    """

    if isinstance(stimuli, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(stimuli.shape),
            "dtype": str(stimuli.dtype),
            "digest": _tensor_digest(stimuli),
        }
    return {
        "kind": "iterable",
        "note": (
            "iterable stimulus identity is not verifiable; resume assumes the "
            "same stimuli in the same iteration order"
        ),
    }


def _build_signature(
    layer_plan: dict[str, str],
    layers_kind: str,
    batch_size: int,
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
    stimuli: Any,
) -> dict[str, Any]:
    """Build the resume-compatibility signature block of the manifest.

    Parameters
    ----------
    layer_plan:
        Normalized ``output key -> layer lookup`` extraction plan.
    layers_kind:
        ``"mapping"`` or ``"sequence"``, preserving list-versus-dict semantics.
    batch_size:
        Number of stimuli per forward pass.
    transform:
        Optional tensor transform supplied by the caller.
    stimuli:
        Stimulus tensor or iterable.

    Returns
    -------
    dict[str, Any]
        JSON-serializable signature compared verbatim on resume.
    """

    return {
        "layer_plan": dict(layer_plan),
        "layers_kind": layers_kind,
        "batch_size": batch_size,
        "transform": _transform_signature(transform),
        "stimuli": _stimuli_signature(stimuli),
    }


def _layer_metadata(
    layer_views: dict[str, Any],
    processed: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Build the per-site self-description block from the first computed batch.

    Parameters
    ----------
    layer_views:
        Mapping from output key to the resolved ``Layer`` view of the first
        computed batch's trace.
    processed:
        The same batch's stored (post-transform, CPU) tensors keyed identically.

    Returns
    -------
    dict[str, dict[str, Any]]
        Site identity, axis semantics, dtype, and device per output key.
    """

    from .errors._base import TorchLensError

    metadata: dict[str, dict[str, Any]] = {}
    for key, layer in layer_views.items():
        captured = layer.out
        site_key: str | None
        site_key_unavailable: str | None
        try:
            site_key = str(layer.site_key)
            site_key_unavailable = None
        except TorchLensError as exc:
            site_key = None
            code = exc.fields.get("code") if isinstance(exc.fields, dict) else None
            site_key_unavailable = str(code or type(exc).__name__)
        stored = processed[key]
        metadata[key] = {
            "layer_label": str(layer.layer_label),
            "site_key": site_key,
            "site_key_unavailable": site_key_unavailable,
            "captured_dtype": str(captured.dtype),
            "captured_device": str(captured.device),
            "per_stimulus_shape": list(captured.shape[1:]),
            "stored_dtype": str(stored.dtype),
            "stored_per_stimulus_shape": list(stored.shape[1:]),
            "batch_axis": 0,
        }
    return metadata


def _base_manifest(signature: dict[str, Any], stimulus_ids: list[str] | None) -> dict[str, Any]:
    """Create a fresh in-progress manifest document.

    Parameters
    ----------
    signature:
        Resume-compatibility signature block.
    stimulus_ids:
        Optional caller-supplied per-stimulus identifiers, in iteration order.

    Returns
    -------
    dict[str, Any]
        Manifest with an empty batch ledger and no layer metadata yet.
    """

    from . import __version__

    return {
        "schema": MANIFEST_SCHEMA,
        "torchlens_version": __version__,
        "status": "in_progress",
        "signature": signature,
        "stimulus_provenance": {
            "order": (
                "row i of every concatenated activation tensor corresponds to "
                "stimulus i in iteration order of the stimuli argument; within "
                "shard k, global stimulus index = (sum of prior shards' "
                "n_stimuli) + row"
            ),
            "n_stimuli": None,
            "stimulus_ids": list(stimulus_ids) if stimulus_ids is not None else None,
        },
        "storage": {
            "shard_filename_format": "batch_{index:05d}.pt",
            "shard_payload": ("dict[output key -> torch.Tensor] with the stimulus axis leading"),
            "tensor_placement": "cpu",
            "writes": (
                "atomic (temp file + os.replace); a shard file bearing its final name is complete"
            ),
        },
        "layers": None,
        "batches": [],
    }


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and structurally validate an extraction manifest.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.json`` inside the extraction directory.

    Returns
    -------
    dict[str, Any]
        Parsed manifest document.

    Raises
    ------
    DatasetExtractionResumeError
        If the manifest is unparseable or not this module's schema.
    """

    try:
        # read_bounded, never json.loads(read_text()): the manifest is a
        # user-supplied artifact, so read_text() would materialize the whole
        # file before any ceiling could apply (an over-size manifest is an
        # allocation DoS). It raises json.JSONDecodeError, a ValueError, so the
        # handler below catches over-size and over-nested payloads unchanged.
        manifest = _json.read_bounded(manifest_path)
    except (OSError, ValueError) as exc:
        raise DatasetExtractionResumeError(
            f"Extraction manifest {str(manifest_path)!r} could not be parsed ({exc}).",
            code="extraction_manifest_invalid",
            remedy="delete the output directory and re-run the extraction from scratch",
            manifest_path=str(manifest_path),
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise DatasetExtractionResumeError(
            f"Extraction manifest {str(manifest_path)!r} does not carry schema "
            f"{MANIFEST_SCHEMA!r} (found {manifest.get('schema') if isinstance(manifest, dict) else type(manifest).__name__!r}).",
            code="extraction_manifest_invalid",
            remedy="delete the output directory and re-run the extraction from scratch",
            manifest_path=str(manifest_path),
        )
    return manifest


def _completed_prefix(manifest: dict[str, Any], container_path: Path) -> list[dict[str, Any]]:
    """Return the ledgered shard prefix whose files are all present on disk.

    The ledger is written contiguously from index 0; the first ledger row whose
    file is missing (user deletion, partial sync) truncates the trusted prefix,
    and everything after it is recomputed.

    Parameters
    ----------
    manifest:
        Parsed manifest document.
    container_path:
        Extraction directory containing the shards.

    Returns
    -------
    list[dict[str, Any]]
        Contiguous ledger rows (``index``, ``file``, ``n_stimuli``) verified
        present on disk.
    """

    prefix: list[dict[str, Any]] = []
    for row in manifest.get("batches") or []:
        expected_name = _shard_filename(int(row["index"]))
        if row.get("file") != expected_name or not (container_path / expected_name).exists():
            break
        prefix.append(row)
    return prefix


def _clean_orphan_tmp_files(container_path: Path) -> None:
    """Remove leftover atomic-write temp files from a crashed run.

    Parameters
    ----------
    container_path:
        Extraction directory to sweep.
    """

    for tmp_path in container_path.glob("*.tmp"):
        with contextlib.suppress(OSError):
            tmp_path.unlink()


def _consume_skipped_stimuli(stimuli: Any, n_skip: int) -> Any:
    """Advance past already-extracted stimuli and return the remaining stream.

    Parameters
    ----------
    stimuli:
        Stimulus tensor or iterable.
    n_skip:
        Exact number of stimuli covered by the trusted shard prefix.

    Returns
    -------
    Any
        Remaining stimuli: a tensor slice, or the advanced iterator.

    Raises
    ------
    DatasetExtractionResumeError
        If an iterable stimulus stream ends before covering the ledgered
        prefix (the stimuli cannot be the ones the artifact was built from).
    """

    if isinstance(stimuli, torch.Tensor):
        return stimuli[n_skip:]
    iterator = iter(stimuli)
    consumed = 0
    while consumed < n_skip:
        try:
            next(iterator)
        except StopIteration:
            raise DatasetExtractionResumeError(
                f"Stimulus iterable ended after {consumed} items but the "
                f"artifact's completed shards cover {n_skip} stimuli.",
                code="extraction_resume_signature_mismatch",
                remedy=(
                    "re-run with the original stimuli, or delete the output "
                    "directory to start a fresh extraction"
                ),
                n_ledgered=n_skip,
                n_available=consumed,
            ) from None
        consumed += 1
    return iterator


def _check_resume_signature(
    existing: dict[str, Any], signature: dict[str, Any], manifest_path: Path
) -> None:
    """Refuse a resume whose run parameters differ from the artifact's.

    Parameters
    ----------
    existing:
        Manifest found in the output directory.
    signature:
        Signature block of the current call.
    manifest_path:
        Manifest path, for the refusal message.

    Raises
    ------
    DatasetExtractionResumeError
        If any signature field differs.
    """

    recorded = existing.get("signature")
    if recorded == signature:
        return
    mismatched = sorted(
        key
        for key in set(signature) | set(recorded or {})
        if (recorded or {}).get(key) != signature.get(key)
    )
    raise DatasetExtractionResumeError(
        f"Extraction artifact at {str(manifest_path.parent)!r} was produced by a "
        f"different run configuration (mismatched signature fields: {mismatched}).",
        code="extraction_resume_signature_mismatch",
        remedy=(
            "re-run with the artifact's original layers, batch_size, transform, "
            "and stimuli, or delete the output directory to start fresh"
        ),
        mismatched_fields=mismatched,
        recorded_signature=recorded,
        requested_signature=signature,
    )


@dataclasses.dataclass(frozen=True)
class _RunPlan:
    """Resolved extraction-run configuration shared by the run engines.

    Attributes
    ----------
    model:
        PyTorch model to run (already moved to ``device`` when one was given).
    stimuli:
        Stimulus tensor or iterable, as supplied by the caller.
    layers:
        The caller's original layer spec (mapping-versus-list semantics).
    layer_plan:
        Normalized ``output key -> layer lookup`` plan.
    layers_kind:
        ``"mapping"`` or ``"sequence"``.
    batch_size:
        Number of stimuli per forward pass.
    device:
        Optional device for stimuli movement.
    transform:
        Optional tensor transform applied before storage.
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    stimulus_ids:
        Optional per-stimulus identifiers recorded as provenance.
    """

    model: nn.Module
    stimuli: Any
    layers: Iterable[str] | Mapping[str, str]
    layer_plan: dict[str, str]
    layers_kind: str
    batch_size: int
    device: torch.device | str | None
    transform: Callable[[torch.Tensor], torch.Tensor] | None
    progress: bool
    stimulus_ids: list[str] | None


def extract_dataset(
    model: nn.Module,
    stimuli: Any,
    layers: Iterable[str] | Mapping[str, str],
    batch_size: int = 32,
    device: torch.device | str | None = None,
    output_dir: str | Path | None = None,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    progress: bool = True,
    *,
    resume: bool = False,
    stimulus_ids: Iterable[str] | None = None,
) -> dict[str, torch.Tensor] | list[Path]:
    """Extract outs from an iterable dataset in batches.

    Row ``i`` of every returned tensor corresponds to stimulus ``i`` in
    iteration order. Batch files are consumed in ``batch_00000.pt``,
    ``batch_00001.pt``, ... order.

    In disk mode the output directory is a SELF-DESCRIBING artifact: shard
    writes are atomic, and ``manifest.json`` records the run signature, per-site
    identity (layer label and structural site key where derivable), stimulus
    ordering/provenance, axis semantics, dtypes, devices, and the TorchLens
    version, updated atomically after every shard. Read it back with
    :func:`torchlens.dataset_extraction.load_extraction`.

    Parameters
    ----------
    model:
        PyTorch model to run.
    stimuli:
        Tensor with a leading batch dimension or iterable of stimulus items.
    layers:
        List or mapping accepted by :func:`torchlens.extract`.
    batch_size:
        Number of stimuli per forward pass.
    device:
        Optional device for model and stimuli.
    output_dir:
        Optional directory. When supplied, each batch output is written as
        ``batch_XXXXX.pt`` and paths are returned, alongside ``manifest.json``.
    transform:
        Optional tensor transform applied to each out before storage.
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    resume:
        Disk mode only (DOCUMENTED-UNSTABLE): continue an interrupted run in
        ``output_dir`` from its last completed shard. The recorded run
        signature (layers, batch size, transform disclosure, stimulus
        descriptor) must match; iterable stimuli are assumed to replay in the
        original order, which resume cannot verify. A completed artifact
        returns its shard paths without running the model.
    stimulus_ids:
        Optional per-stimulus identifiers (DOCUMENTED-UNSTABLE), recorded in
        the manifest as provenance in iteration order.

    Returns
    -------
    dict[str, torch.Tensor] | list[pathlib.Path]
        In-memory concatenated outs, or written batch paths.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        If ``resume=True`` is combined with in-memory mode.
    DatasetExtractionResumeError
        If the artifact in ``output_dir`` cannot be safely continued.
    """

    from ._errors import InvalidArgumentError

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if resume and output_dir is None:
        raise InvalidArgumentError(
            "resume=True requires output_dir: only disk-mode extraction leaves "
            "a shard ledger to resume from.",
            code="extraction_resume_requires_output_dir",
            remedy="pass output_dir= (disk mode) or drop resume=True",
        )
    if device is not None:
        model = model.to(device)

    import torchlens as _tl

    plan = _RunPlan(
        model=model,
        stimuli=stimuli,
        layers=layers,
        layer_plan=_tl._normalize_extract_layers(layers),
        layers_kind="mapping" if isinstance(layers, Mapping) else "sequence",
        batch_size=batch_size,
        device=device,
        transform=transform,
        progress=progress,
        stimulus_ids=list(stimulus_ids) if stimulus_ids is not None else None,
    )
    if output_dir is None:
        return _extract_in_memory(plan)
    return _extract_to_disk(plan, Path(output_dir), resume)


def _batch_iterable(plan: _RunPlan, remaining: Any) -> Iterable[Any]:
    """Build the (optionally progress-wrapped) batch iterator for a run.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    remaining:
        Stimuli still to extract (full set, tensor slice, or advanced iterator).

    Returns
    -------
    Iterable[Any]
        Batches suitable for ``model.forward``.
    """

    batches = _iter_batches(remaining, plan.batch_size)
    total = None
    if isinstance(remaining, torch.Tensor):
        total = (remaining.shape[0] + plan.batch_size - 1) // plan.batch_size
    if plan.progress:
        from .utils.display import progress_bar

        batches = progress_bar(batches, total=total, desc="torchlens.extract", enabled=True)
    return batches


def _extract_in_memory(plan: _RunPlan) -> dict[str, torch.Tensor]:
    """Run the in-memory extraction engine.

    Parameters
    ----------
    plan:
        Resolved run configuration.

    Returns
    -------
    dict[str, torch.Tensor]
        Concatenated outs keyed as :func:`torchlens.extract` keys them.
    """

    import torchlens as _tl

    accumulator: dict[str, list[torch.Tensor]] = {}
    for batch in _batch_iterable(plan, plan.stimuli):
        batch = _move_nested_to_device(batch, plan.device)
        _trace, batch_outputs, _views = _tl._extract_layers_with_trace(
            plan.model, batch, plan.layers
        )
        _merge_batch_outputs(accumulator, batch_outputs, plan.transform)
    return {label: torch.cat(tensors, dim=0) for label, tensors in accumulator.items()}


def _prepare_disk_run(
    plan: _RunPlan, container_path: Path, resume: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path] | None]:
    """Prepare the disk-mode manifest and resolve the resume state.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    container_path:
        Extraction directory.
    resume:
        Whether to continue from an existing ledger.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]], list[Path] | None]
        The (written) manifest, the trusted completed-shard ledger rows, and —
        when the artifact is already complete with every shard present — the
        final shard paths (callers return them without running the model).

    Raises
    ------
    DatasetExtractionResumeError
        On unmanifested shard directories or signature mismatches.
    """

    container_path.mkdir(parents=True, exist_ok=True)
    manifest_path = container_path / MANIFEST_FILENAME
    signature = _build_signature(
        plan.layer_plan, plan.layers_kind, plan.batch_size, plan.transform, plan.stimuli
    )
    manifest: dict[str, Any] | None = None
    completed_rows: list[dict[str, Any]] = []
    if resume and manifest_path.exists():
        existing = _load_manifest(manifest_path)
        _check_resume_signature(existing, signature, manifest_path)
        ledgered_total = len(existing.get("batches") or [])
        completed_rows = _completed_prefix(existing, container_path)
        manifest = existing
        manifest["batches"] = list(completed_rows)
        if manifest.get("status") == "complete" and len(completed_rows) == ledgered_total:
            return (
                manifest,
                completed_rows,
                [container_path / str(row["file"]) for row in completed_rows],
            )
        manifest["status"] = "in_progress"
    elif resume and any(container_path.glob("batch_*.pt")):
        raise DatasetExtractionResumeError(
            f"Output directory {str(container_path)!r} contains batch shards "
            "but no manifest; it predates resumable extraction or lost its "
            "ledger, so completed work cannot be verified.",
            code="extraction_resume_unmanifested_dir",
            remedy="delete the output directory (or point output_dir at a fresh one) and re-run",
            output_dir=str(container_path),
        )
    if manifest is None:
        manifest = _base_manifest(signature, plan.stimulus_ids)
    elif plan.stimulus_ids is not None:
        manifest["stimulus_provenance"]["stimulus_ids"] = plan.stimulus_ids
    _clean_orphan_tmp_files(container_path)
    _atomic_write_json(manifest_path, manifest)
    return manifest, completed_rows, None


def _extract_to_disk(plan: _RunPlan, container_path: Path, resume: bool) -> list[Path]:
    """Run the disk-mode extraction engine (atomic shards + manifest ledger).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    container_path:
        Extraction directory.
    resume:
        Whether to continue from an existing ledger.

    Returns
    -------
    list[pathlib.Path]
        Every shard path in consumption order, including resumed prefixes.
    """

    import torchlens as _tl

    manifest, completed_rows, complete_paths = _prepare_disk_run(plan, container_path, resume)
    if complete_paths is not None:
        return complete_paths
    n_skip = sum(int(row["n_stimuli"]) for row in completed_rows)
    remaining = _consume_skipped_stimuli(plan.stimuli, n_skip) if n_skip else plan.stimuli
    start_index = len(completed_rows)
    container_paths = [container_path / str(row["file"]) for row in completed_rows]

    for offset, batch in enumerate(_batch_iterable(plan, remaining)):
        batch_index = start_index + offset
        batch = _move_nested_to_device(batch, plan.device)
        _trace, batch_outputs, layer_views = _tl._extract_layers_with_trace(
            plan.model, batch, plan.layers
        )
        processed = {
            label: (plan.transform(tensor) if plan.transform is not None else tensor).detach().cpu()
            for label, tensor in batch_outputs.items()
        }
        if manifest.get("layers") is None:
            manifest["layers"] = _layer_metadata(layer_views, processed)
        batch_path = container_path / _shard_filename(batch_index)
        _atomic_torch_save(processed, batch_path)
        n_rows = next(iter(processed.values())).shape[0] if processed else 0
        manifest["batches"].append(
            {"index": batch_index, "file": batch_path.name, "n_stimuli": n_rows}
        )
        _atomic_write_json(container_path / MANIFEST_FILENAME, manifest)
        container_paths.append(batch_path)

    manifest["status"] = "complete"
    manifest["stimulus_provenance"]["n_stimuli"] = sum(
        int(row["n_stimuli"]) for row in manifest["batches"]
    )
    _atomic_write_json(container_path / MANIFEST_FILENAME, manifest)
    return container_paths


@dataclasses.dataclass(frozen=True)
class LoadedExtraction:
    """A dataset-extraction artifact read back with its self-description.

    Attributes
    ----------
    manifest:
        Parsed ``manifest.json`` document (site identity, stimulus provenance,
        axis semantics, dtypes, devices, run signature, TorchLens version).
    activations:
        Concatenated activations keyed by output key, stimulus axis leading.
    batch_paths:
        Shard files in consumption order.
    """

    manifest: dict[str, Any]
    activations: dict[str, torch.Tensor]
    batch_paths: list[Path]


def load_extraction(
    output_dir: str | Path,
    layers: Iterable[str] | None = None,
) -> LoadedExtraction:
    """Load a disk-mode extraction artifact with its self-description.

    Parameters
    ----------
    output_dir:
        Directory previously written by disk-mode :func:`extract_dataset`.
    layers:
        Optional subset of output keys to load; defaults to every key.

    Returns
    -------
    LoadedExtraction
        Manifest, concatenated activations, and shard paths.

    Raises
    ------
    DatasetExtractionResumeError
        If the manifest is missing/invalid, the artifact is incomplete, or a
        requested output key is not in the artifact.
    """

    container_path = Path(output_dir)
    manifest = _load_manifest(container_path / MANIFEST_FILENAME)
    if manifest.get("status") != "complete":
        raise DatasetExtractionResumeError(
            f"Extraction artifact at {str(container_path)!r} has status "
            f"{manifest.get('status')!r}, not 'complete'.",
            code="extraction_manifest_invalid",
            remedy="finish the run first: extract_dataset(..., resume=True)",
            status=manifest.get("status"),
        )
    rows = _completed_prefix(manifest, container_path)
    if len(rows) != len(manifest.get("batches") or []):
        raise DatasetExtractionResumeError(
            f"Extraction artifact at {str(container_path)!r} is missing ledgered "
            f"shard files ({len(rows)} of {len(manifest.get('batches') or [])} present).",
            code="extraction_manifest_invalid",
            remedy="re-run extract_dataset(..., resume=True) to restore the missing shards",
            n_present=len(rows),
            n_ledgered=len(manifest.get("batches") or []),
        )
    available = set((manifest.get("layers") or {}).keys())
    selected = list(layers) if layers is not None else None
    if selected is not None:
        missing = sorted(set(selected) - available)
        if missing:
            raise DatasetExtractionResumeError(
                f"Output keys {missing} are not in the artifact at "
                f"{str(container_path)!r} (available: {sorted(available)}).",
                code="extraction_manifest_invalid",
                remedy="request only output keys recorded in the manifest's layers block",
                missing_keys=missing,
                available_keys=sorted(available),
            )
    per_key: dict[str, list[torch.Tensor]] = {}
    batch_paths = [container_path / str(row["file"]) for row in rows]
    for batch_path in batch_paths:
        payload = torch.load(batch_path, weights_only=True)
        for key, tensor in payload.items():
            if selected is not None and key not in selected:
                continue
            per_key.setdefault(key, []).append(tensor)
    activations = {key: torch.cat(tensors, dim=0) for key, tensors in per_key.items()}
    return LoadedExtraction(manifest=manifest, activations=activations, batch_paths=batch_paths)


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "DatasetExtractionResumeError",
    "LoadedExtraction",
    "extract_dataset",
    "load_extraction",
]
