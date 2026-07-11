"""Trusted-local replay boundary cache helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .adapters import resolve_split_adapter
from .adapters.base import SplitBackendAdapter
from .boundary import ReplayBoundary
from .errors import SplitErrorContext, SplitUnsupportedError
from .ir import BoundarySchema


_CACHE_METADATA_KEYS = (
    "split_id",
    "graph_shape_hash",
    "batch_symbol",
    "dynamic_batch",
    "device_policy",
)


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


def save_boundary(
    boundary: ReplayBoundary,
    path: str | Path,
    adapter: SplitBackendAdapter | None = None,
) -> None:
    """Save a replay boundary cache directory.

    Notes
    -----
    The payload is a trusted-local pickle. Do not load boundary caches from
    untrusted sources.
    """

    resolved_adapter = adapter or resolve_split_adapter(boundary.backend)
    if not resolved_adapter.supports_boundary_cache:
        raise SplitUnsupportedError(
            f"backend={boundary.backend!r} does not support boundary cache.",
            context=SplitErrorContext(
                backend=boundary.backend,
                split_point=str(boundary.metadata.get("split_id", "")),
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="unsupported boundary cache",
            ),
        )
    cache_boundary = _cacheable_boundary(boundary, resolved_adapter)
    cache_dir = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": cache_boundary.backend,
        "split_id": cache_boundary.metadata.get("split_id"),
        "graph_shape_hash": cache_boundary.metadata.get("graph_shape_hash"),
        "boundary_spec": {
            key: _spec_to_json(item) for key, item in cache_boundary.spec.items()
        },
        "tensor_ids": list(cache_boundary.tensors),
        "payload_format": "pickle",
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (cache_dir / "payload.pkl").open("wb") as handle:
        pickle.dump(cache_boundary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_boundary(
    path: str | Path,
    adapter: SplitBackendAdapter | None = None,
) -> ReplayBoundary:
    """Load a trusted-local replay boundary cache directory."""

    cache_dir = Path(path)
    manifest_path = cache_dir / "manifest.json"
    payload_path = cache_dir / "payload.pkl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with payload_path.open("rb") as handle:
        boundary = pickle.load(handle)
    if not isinstance(boundary, ReplayBoundary):
        raise TypeError("Boundary cache payload did not contain a ReplayBoundary.")
    resolved_adapter = adapter or resolve_split_adapter(str(manifest["backend"]))
    boundary.validate(split_id=manifest.get("split_id"), adapter=resolved_adapter)
    return boundary


__all__ = ["load_boundary", "save_boundary"]
