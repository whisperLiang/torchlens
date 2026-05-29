"""Split training helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .boundary import ReplayBoundary
from .cache import load_boundary, save_boundary

BoundaryGradients = dict[str, torch.Tensor | None]


def train_suffix(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[torch.Tensor, BoundaryGradients]:
    """Train or differentiate suffix from detached boundary tensors."""

    runtime.validate_boundary(boundary)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    device = runtime._runtime_device()
    detached_tensors = {}
    for key, tensor in boundary.tensors.items():
        root = tensor.detach()
        if device is not None:
            root = root.to(device)
        detached_tensors[key] = root.requires_grad_(root.is_floating_point() or root.is_complex())
    detached = ReplayBoundary(
        tensors=detached_tensors,
        spec=boundary.spec,
        metadata=dict(boundary.metadata),
    )
    outputs = runtime.run_suffix(detached)
    loss = _default_loss(outputs, targets) if loss_fn is None else loss_fn(outputs, targets)
    loss.backward()
    grads = {key: tensor.grad for key, tensor in detached_tensors.items()}
    if optimizer is not None:
        optimizer.step()
    return loss.detach(), grads


def backward_prefix(
    runtime: Any,
    boundary: ReplayBoundary,
    boundary_grads: BoundaryGradients,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Backpropagate suffix boundary gradients through the retained prefix graph."""

    runtime.validate_boundary(boundary)
    if not boundary.metadata.get("supports_prefix_backward", False):
        raise ValueError("Boundary was not produced by run_training_prefix().")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    tensors: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    for key, tensor in boundary.tensors.items():
        grad = boundary_grads.get(key)
        if grad is not None:
            tensors.append(tensor)
            grads.append(grad.to(tensor.device))
    if tensors:
        torch.autograd.backward(tensors, grads)
    if optimizer is not None:
        optimizer.step()


def build_feature_cache(runtime: Any, dataloader: Iterable[Any], cache_dir: str | Path) -> None:
    """Build a CPU ReplayBoundary cache for suffix-only training."""

    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(dataloader):
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        boundary = runtime.run_prefix(*inputs).detach().cpu()
        save_boundary(boundary, path / f"{index:08d}.pt")


class BoundaryCacheDataset(Dataset[ReplayBoundary]):
    """Dataset reading cached ReplayBoundary files."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.files = tuple(sorted(self.cache_dir.glob("*.pt")))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> ReplayBoundary:
        return load_boundary(self.files[index])


def train_suffix_from_cache(
    runtime: Any,
    cache_loader: Iterable[Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> list[torch.Tensor]:
    """Train suffix from cached boundaries and return detached losses."""

    losses: list[torch.Tensor] = []
    for item in cache_loader:
        if isinstance(item, ReplayBoundary):
            boundary = item
            targets = None
        else:
            boundary, targets = item
        loss, _grads = runtime.train_suffix(boundary, targets, loss_fn, optimizer)
        losses.append(loss)
    return losses


def _default_loss(outputs: Any, targets: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
        if targets.dtype == torch.long and outputs.ndim >= 2:
            return F.cross_entropy(outputs, targets)
        return F.mse_loss(outputs, targets)
    raise TypeError("A loss_fn is required for non-tensor outputs or targets.")


__all__ = [
    "BoundaryCacheDataset",
    "BoundaryGradients",
    "backward_prefix",
    "build_feature_cache",
    "train_suffix",
    "train_suffix_from_cache",
]
