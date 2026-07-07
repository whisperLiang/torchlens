"""Split backend adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..boundary import ReplayBoundary
    from ..graph import SplitTraceGraph
    from ..planner import SplitPlan
    from ..program import ReplayProgram, ReplaySegment
    from ..spec import SplitSpec
    from ..training import BoundaryGradients, TrainingStepResult


@dataclass(frozen=True)
class SegmentBundle:
    """Backend-built prefix/suffix segment bundle."""

    prefix: Any
    training_prefix: Any | None
    suffix: Any


class TensorOps(Protocol):
    """Backend tensor operations needed by split boundaries."""

    def is_tensor(self, value: Any) -> bool:
        """Return whether ``value`` is a tensor-like value for this backend."""
        ...

    def shape(self, value: Any) -> tuple[int, ...] | None:
        """Return the runtime tensor shape."""
        ...

    def dtype_name(self, value: Any) -> str | None:
        """Return the runtime dtype name."""
        ...

    def requires_grad(self, value: Any) -> bool | None:
        """Return whether ``value`` participates in autograd."""
        ...

    def detach(self, value: Any) -> Any:
        """Return a detached value."""
        ...

    def clone(self, value: Any) -> Any:
        """Return a cloned value."""
        ...

    def to_device(self, value: Any, device: Any) -> Any:
        """Move ``value`` to ``device``."""
        ...

    def collate(self, values: list[Any]) -> Any:
        """Collate same-key boundary tensors."""
        ...

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like ``value``."""
        ...

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two tensor-like values are numerically close."""
        ...


class ReplayLowerer(Protocol):
    """Backend hook for lowering a split graph into replay IR."""

    def lower_program(
        self,
        graph: "SplitTraceGraph",
        plan: "SplitPlan",
        spec: "SplitSpec",
        *,
        segment: "ReplaySegment",
    ) -> "ReplayProgram":
        """Lower one split segment to replay IR."""
        ...


class ReplayExecutor(Protocol):
    """Backend hook that turns a preflighted plan into executable segments."""

    def build_segments(
        self,
        graph: "SplitTraceGraph",
        plan: "SplitPlan",
        spec: "SplitSpec",
    ) -> SegmentBundle:
        """Build backend-specific split replay segments."""
        ...


class TrainingEngine(Protocol):
    """Backend split-training engine."""

    name: str

    def train_suffix(
        self,
        runtime: Any,
        boundary: "ReplayBoundary",
        targets: Any,
        loss_fn: Any = None,
        optimizer: Any = None,
    ) -> "TrainingStepResult":
        """Differentiate or train a split suffix."""
        ...

    def backward_prefix(
        self,
        runtime: Any,
        boundary: "ReplayBoundary",
        boundary_grads: "BoundaryGradients",
        optimizer: Any = None,
    ) -> Any:
        """Propagate suffix boundary gradients through a split prefix."""
        ...


class SplitBackendAdapter(TensorOps, ReplayExecutor, Protocol):
    """Protocol implemented by split replay backend adapters."""

    name: str
    supports_replay: bool
    supports_training: bool
    supports_boundary_cache: bool
    supports_dynamic_batch: bool


__all__ = [
    "ReplayExecutor",
    "ReplayLowerer",
    "SegmentBundle",
    "SplitBackendAdapter",
    "TensorOps",
    "TrainingEngine",
]
