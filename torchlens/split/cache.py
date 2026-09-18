"""Trusted-local replay boundary cache helpers."""

from __future__ import annotations

import hmac
import json
from pathlib import Path
from typing import Any

from .._io import _json
from .adapters import resolve_split_adapter
from .adapters.base import SplitBackendAdapter
from .boundary import ReplayBoundary
from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError
from .ir import BoundarySchema

_CACHE_METADATA_KEYS = (
    "split_id",
    "graph_shape_hash",
    "batch_symbol",
    "runtime_batch_size",
    "shape_program_hash",
    "device_policy",
    "batch_validation",
    "runtime_batch_validation",
    "state_fingerprint",
    "state_prefix_kind",
    "profile_hash",
)

_PAYLOAD_FORMAT = "authenticated_pickle_v1"


def _cache_secret() -> bytes:
    """Derive a split-only signing key from the private local capture-cache key.

    The key deliberately lives outside the supplied boundary directory. A copied
    cache must never be allowed to supply its own authentication key. Reuse the
    capture cache's ownership/mode checks and authenticated single-read loader;
    native backend tensor pickles cannot use the portable metadata allowlist.
    """

    from ..user_funcs import _prepare_capture_cache_dir

    _root, secret = _prepare_capture_cache_dir(None)
    return hmac.digest(secret, b"torchlens.split.boundary.v1", "sha256")


def _shape_to_json(shape: Any) -> Any:
    """Serialize symbolic shape metadata."""

    if shape is None:
        return None
    as_tuple = getattr(shape, "as_tuple", None)
    return list(as_tuple() if callable(as_tuple) else shape)


def _spec_to_json(item: BoundarySchema) -> dict[str, Any]:
    """Serialize one boundary spec item for the manifest."""

    return {
        "canonical_id": item.canonical_id,
        "label": item.label,
        "backend": item.backend,
        "module_path": item.module_path,
        "op_type": item.op_type,
        "shape": _shape_to_json(item.shape),
        "dtype": item.dtype,
        "requires_grad": item.requires_grad,
        "role": item.role,
        "output_index": item.output_index,
        "container_path": [repr(part) for part in item.container_path],
        "device_policy": item.device_policy,
    }


def _cacheable_boundary(boundary: ReplayBoundary, adapter: SplitBackendAdapter) -> ReplayBoundary:
    """Return a detached suffix-only boundary suitable for trusted local pickle cache."""

    if not boundary.metadata.get("supports_prefix_backward"):
        return boundary
    return ReplayBoundary(
        backend=boundary.backend,
        tensors={
            key: adapter.clone(adapter.detach(value)) for key, value in boundary.tensors.items()
        },
        spec=boundary.spec,
        metadata={
            **{
                key: boundary.metadata[key]
                for key in _CACHE_METADATA_KEYS
                if key in boundary.metadata
            },
            "supports_prefix_backward": False,
        },
    )


def _boundary_manifest(boundary: ReplayBoundary) -> dict[str, Any]:
    """Return the manifest fields that must agree with the authenticated payload."""

    return {
        "backend": boundary.backend,
        "split_id": boundary.metadata.get("split_id"),
        "graph_shape_hash": boundary.metadata.get("graph_shape_hash"),
        "runtime_batch_size": boundary.metadata.get("runtime_batch_size"),
        "shape_program_hash": boundary.metadata.get("shape_program_hash"),
        "boundary_spec": {key: _spec_to_json(item) for key, item in boundary.spec.items()},
        "tensor_ids": list(boundary.tensors),
        "payload_format": _PAYLOAD_FORMAT,
    }


def save_boundary(
    boundary: ReplayBoundary,
    path: str | Path,
    adapter: SplitBackendAdapter | None = None,
) -> None:
    """Save a replay boundary cache directory.

    Notes
    -----
    The payload is authenticated with a private, machine-local key before it can
    be unpickled. It remains a local cache, not a portable artifact. Unsigned old
    caches and caches signed by a different key must be regenerated. The shared
    authenticated cache writer enforces its 2-GiB serialized-payload limit.
    """

    from ..user_funcs import _store_authenticated_capture_cache

    resolved_adapter = adapter or resolve_split_adapter(boundary.backend)
    if not resolved_adapter.supports_boundary_cache:
        raise SplitUnsupportedError(
            f"backend={boundary.backend!r} does not support boundary cache.",
            context=SplitErrorContext(
                backend=boundary.backend,
                split_point=str(boundary.metadata.get("split_id", "")),
                reason="unsupported boundary cache",
            ),
        )
    cache_boundary = _cacheable_boundary(boundary, resolved_adapter)
    cache_dir = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _boundary_manifest(cache_boundary)
    if not _store_authenticated_capture_cache(
        cache_boundary, cache_dir / "payload.pkl", _cache_secret()
    ):
        raise SplitBoundaryError("Boundary cache payload could not be written; see cache warning.")
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_boundary(
    path: str | Path,
    adapter: SplitBackendAdapter | None = None,
) -> ReplayBoundary:
    """Load a locally authenticated boundary, refusing unsigned or altered bytes."""

    from ..user_funcs import _load_authenticated_capture_cache

    cache_dir = Path(path)
    manifest_path = cache_dir / "manifest.json"
    payload_path = cache_dir / "payload.pkl"
    manifest = _json.read_bounded(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("payload_format") != _PAYLOAD_FORMAT:
        raise SplitBoundaryError(
            "Boundary cache is not an authenticated cache; regenerate it with save_boundary()."
        )
    boundary = _load_authenticated_capture_cache(payload_path, _cache_secret())
    if boundary is None:
        raise SplitBoundaryError(
            "Boundary cache authentication failed; regenerate it with save_boundary()."
        )
    if not isinstance(boundary, ReplayBoundary):
        raise TypeError("Boundary cache payload did not contain a ReplayBoundary.")
    if manifest != _boundary_manifest(boundary):
        raise SplitBoundaryError("Boundary cache manifest does not match authenticated payload.")
    resolved_adapter = adapter or resolve_split_adapter(boundary.backend)
    boundary.validate(split_id=manifest.get("split_id"), adapter=resolved_adapter)
    return boundary


__all__ = ["load_boundary", "save_boundary"]
