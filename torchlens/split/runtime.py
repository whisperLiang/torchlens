"""Prepared TorchLens split runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from .boundary import ReplayBoundary
from .cache import load_boundary, save_boundary
from .codegen import SegmentBundle
from .frontier import SplitPlan
from .spec import SplitSpec
from .trace_graph import TraceGraph
from .training import BoundaryGradients, backward_prefix as _backward_prefix
from .training import train_suffix as _train_suffix


class SplitRuntime:
    """Prepared split replay and split training runtime."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        trace_graph: TraceGraph,
        split_spec: SplitSpec,
        plan: SplitPlan,
        segments: SegmentBundle,
    ) -> None:
        self.model = model
        self.trace_graph = trace_graph
        self.split_spec = split_spec
        self.plan = plan
        self.segments = segments

    @property
    def split_id(self) -> str:
        """Return split id."""

        return self.plan.split_id

    @property
    def boundary_spec(self) -> dict[str, Any]:
        """Return boundary tensor specs."""

        return self.plan.boundary_specs

    def run_prefix(self, *inputs: Any) -> ReplayBoundary:
        """Run inference-style prefix and detach boundary tensors."""

        with torch.no_grad():
            boundary = self.segments.prefix(*inputs)
        boundary.validate(self.plan.boundary_specs, shape_env=self.trace_graph.shape_env, split_id=self.split_id)
        return boundary

    def run_training_prefix(self, *inputs: Any) -> ReplayBoundary:
        """Run prefix while retaining autograd graph for split training."""

        boundary = self.segments.training_prefix(*inputs)
        metadata = dict(boundary.metadata)
        metadata["supports_prefix_backward"] = True
        boundary = ReplayBoundary(boundary.tensors, boundary.spec, metadata)
        boundary.validate(
            self.plan.boundary_specs,
            shape_env=self.trace_graph.shape_env,
            split_id=self.split_id,
            require_grad_match=False,
        )
        return boundary

    def run_suffix(self, boundary: ReplayBoundary) -> Any:
        """Run suffix from a ReplayBoundary."""

        self.validate_boundary(boundary)
        device = self._runtime_device()
        if device is not None:
            boundary = boundary.to(device)
        return self.segments.suffix(boundary)

    def replay(self, *inputs: Any) -> Any:
        """Run ``run_suffix(run_prefix(*inputs))``."""

        return self.run_suffix(self.run_prefix(*inputs))

    def train_suffix(
        self,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[torch.Tensor, BoundaryGradients]:
        """Train or differentiate suffix from detached boundary tensors."""

        return _train_suffix(self, boundary, targets, loss_fn, optimizer)

    def backward_prefix(
        self,
        boundary: ReplayBoundary,
        boundary_grads: BoundaryGradients,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Backpropagate boundary gradients through prefix."""

        _backward_prefix(self, boundary, boundary_grads, optimizer)

    def validate_boundary(self, boundary: ReplayBoundary) -> None:
        """Validate boundary schema and shapes without checking device."""

        boundary.validate(
            self.plan.boundary_specs,
            shape_env=self.trace_graph.shape_env,
            split_id=self.split_id,
            require_grad_match=False,
        )

    def validate_equivalence(
        self,
        model: torch.nn.Module,
        inputs: tuple[Any, ...],
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> bool:
        """Return whether full model and split replay outputs are close."""

        with torch.no_grad():
            full = model(*inputs)
            split = self.replay(*inputs)
        return _tree_allclose(full, split, atol=atol, rtol=rtol)

    def save(self, path: str | Path) -> None:
        """Save portable split runtime metadata.

        Generated callables close over live PyTorch functions and model
        parameters, so executable runtime rehydration is intentionally handled by
        rerunning ``prepare_split`` with the original model.
        """

        torch.save(
            {
                "split_spec": self.split_spec,
                "plan": self.plan,
                "graph_shape_hash": self.trace_graph.graph_shape_hash,
                "boundary_spec": self.plan.boundary_specs,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> Any:
        """Load split runtime metadata saved by :meth:`save`."""

        return torch.load(Path(path), map_location="cpu", weights_only=False)

    def save_boundary(self, boundary: ReplayBoundary, path: str | Path) -> None:
        """Save a boundary payload."""

        save_boundary(boundary, path)

    def load_boundary(self, path: str | Path) -> ReplayBoundary:
        """Load and validate a boundary payload."""

        boundary = load_boundary(path)
        self.validate_boundary(boundary)
        return boundary

    def _runtime_device(self) -> torch.device | None:
        for parameter in self.model.parameters():
            return parameter.device
        for buffer in self.model.buffers():
            return buffer.device
        return None


def _tree_allclose(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        return all(_tree_allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_tree_allclose(left[key], right[key], atol=atol, rtol=rtol) for key in left)
    return left == right


__all__ = ["SplitRuntime"]
