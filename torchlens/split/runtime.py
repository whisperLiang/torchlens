"""Public split runtime object."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters.base import SegmentBundle, SplitBackendAdapter
from .boundary import ReplayBoundary
from .cache import load_boundary, save_boundary
from .errors import SplitBoundaryError, SplitErrorContext
from .graph import SplitTraceGraph
from .planner import SplitPlan
from .spec import BoundaryTensorSpec, SplitSpec
from .validation import nested_allclose


class SplitRuntime:
    """Prepared split runtime for prefix/suffix replay."""

    model: Any
    trace: Any
    trace_graph: SplitTraceGraph
    split_spec: SplitSpec
    plan: SplitPlan
    adapter: SplitBackendAdapter
    segments: SegmentBundle

    def __init__(
        self,
        *,
        model: Any,
        trace: Any,
        trace_graph: SplitTraceGraph,
        split_spec: SplitSpec,
        plan: SplitPlan,
        adapter: SplitBackendAdapter,
        segments: SegmentBundle,
    ) -> None:
        """Create a prepared split runtime."""

        self.model = model
        self.trace = trace
        self.trace_graph = trace_graph
        self.split_spec = split_spec
        self.plan = plan
        self.adapter = adapter
        self.segments = segments

    @property
    def split_id(self) -> str:
        """Stable split ID for this runtime."""

        return self.plan.split_id

    @property
    def boundary_spec(self) -> dict[str, BoundaryTensorSpec]:
        """Boundary ABI spec keyed by boundary tensor ID."""

        return self.plan.boundary_spec

    def run_prefix(self, *inputs: Any) -> ReplayBoundary:
        """Run the detached inference prefix."""

        return self.segments.prefix(*inputs, detach_boundary=True)

    def run_training_prefix(self, *inputs: Any) -> ReplayBoundary:
        """Run the graph-connected training prefix."""

        if self.segments.training_prefix is None:
            from .errors import SplitUnsupportedError

            raise SplitUnsupportedError(
                f"backend={self.adapter.name!r} does not support split training prefixes.",
                context=SplitErrorContext(
                    backend=self.adapter.name,
                    split_point=self.split_spec.boundary,
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="unsupported training prefix",
                ),
            )
        return self.segments.training_prefix(*inputs, detach_boundary=False)

    def validate_boundary(self, boundary: ReplayBoundary) -> None:
        """Validate ``boundary`` for this runtime."""

        if boundary.backend != self.adapter.name:
            raise SplitBoundaryError(
                "Replay boundary backend differs from runtime backend.",
                context=SplitErrorContext(
                    backend=boundary.backend,
                    split_point=self.split_spec.boundary,
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="backend mismatch",
                ),
            )
        boundary.validate(
            self.boundary_spec,
            split_id=self.split_id,
            adapter=self.adapter,
        )

    def run_suffix(self, boundary: ReplayBoundary) -> Any:
        """Validate and run the suffix from a replay boundary."""

        self.validate_boundary(boundary)
        return self.segments.suffix(boundary)

    def replay(self, *inputs: Any) -> Any:
        """Run prefix then suffix."""

        return self.run_suffix(self.run_prefix(*inputs))

    def validate_equivalence(
        self,
        model: Any,
        inputs: tuple[Any, ...],
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> bool:
        """Return whether full model and split replay outputs match."""

        full_output = model(*inputs)
        replay_output = self.replay(*inputs)
        return nested_allclose(self.adapter, full_output, replay_output, atol=atol, rtol=rtol)

    def train_suffix(
        self,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Any = None,
        optimizer: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Train the suffix and return boundary gradients."""

        from .training import train_suffix

        return train_suffix(self, boundary, targets, loss_fn=loss_fn, optimizer=optimizer)

    def backward_prefix(
        self,
        boundary: ReplayBoundary,
        boundary_grads: dict[str, Any],
        optimizer: Any = None,
    ) -> Any:
        """Apply or return suffix boundary gradients for the graph-connected prefix."""

        from .training import backward_prefix

        return backward_prefix(self, boundary, boundary_grads, optimizer=optimizer)

    def save_boundary(self, boundary: ReplayBoundary, path: str | Path) -> None:
        """Save a replay boundary cache."""

        save_boundary(boundary, path, adapter=self.adapter)

    def load_boundary(self, path: str | Path) -> ReplayBoundary:
        """Load a replay boundary cache."""

        return load_boundary(path, adapter=self.adapter)


__all__ = ["SplitRuntime"]
