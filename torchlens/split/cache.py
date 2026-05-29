"""ReplayBoundary cache helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from .boundary import ReplayBoundary


def save_boundary(boundary: ReplayBoundary, path: str | Path) -> None:
    """Save a ReplayBoundary, normalizing tensors to CPU."""

    cpu_boundary = boundary.cpu()
    torch.save(
        {
            "tensors": cpu_boundary.tensors,
            "spec": cpu_boundary.spec,
            "metadata": cpu_boundary.metadata,
        },
        Path(path),
    )


def load_boundary(path: str | Path) -> ReplayBoundary:
    """Load a ReplayBoundary from disk."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return ReplayBoundary(
        tensors=payload["tensors"],
        spec=payload["spec"],
        metadata=payload["metadata"],
    )


__all__ = ["load_boundary", "save_boundary"]
