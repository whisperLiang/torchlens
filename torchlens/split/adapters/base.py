"""Split backend adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..boundary import ReplayBoundary
    from ..graph import SplitTraceGraph
    from ..planner import SplitPlan
    from ..program import ReplayProgram, ReplaySegment
    from ..ir import SplitRequest
    from ..training import BoundaryGradients, TrainingStepResult


def boundary_overlay(boundary: "ReplayBoundary", plan: "SplitPlan") -> dict[str, Any]:
    """Return replay values keyed only by canonical graph value IDs."""

    if not plan.boundary_bindings:
        return dict(boundary.tensors)
    missing = tuple(key for key in plan.boundary_bindings if key not in boundary.tensors)
    if missing:
        from ..errors import SplitBoundaryError

        raise SplitBoundaryError(f"Boundary is missing canonical values for keys {missing!r}.")
    return {node_id: boundary.tensors[key] for key, node_id in plan.boundary_bindings.items()}


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

    def resize_batch(self, value: Any, axis: int, batch_size: int) -> Any:
        """Clone one tensor leaf with ``axis`` resized for shape witnessing."""
        ...


class SplitPolicyMixin:
    """Reusable adapter-owned capability and boundary policy hooks."""

    name = ""
    native_target_types: frozenset[str] = frozenset()
    allow_callable_target = False
    native_state_replay = False

    def resize_batch(self, value: Any, axis: int, batch_size: int) -> Any:
        """Resize a batch leaf when an adapter supports shape witnessing."""

        del value, axis, batch_size
        raise NotImplementedError(f"backend={self.name!r} does not support shape witnesses")

    def target_support_reasons(self, node: Any, graph: Any) -> tuple[str, ...]:
        """Validate a node against this adapter's native capture handles."""

        del graph
        if (
            node.is_input
            or (node.is_output and node.target is None)
            or (node.is_buffer or (not node.parents and not node.is_output))
        ):
            return ()
        if node.target is None:
            return ("missing replay target/capture",)
        target_type = type(node.target).__name__
        if target_type in self.native_target_types:
            return ()
        if self.allow_callable_target and callable(node.target):
            return ()
        return (f"unsupported {self.name} target {target_type!r}",)

    def stateful_replay_reasons(self, node: Any, backend: str) -> tuple[str, ...]:
        """Return conservative state/random replay failures for this adapter."""

        del backend
        if self.native_state_replay:
            return ()
        op_text = node.op_type.lower()
        target_name = str(
            getattr(node.target, "op_type", "") or getattr(node.target, "__name__", "")
        )
        target_text = target_name.lower()
        tokens = (op_text, target_text)
        is_dropout = any("dropout" in token for token in tokens)
        if is_dropout and (node.kwargs_template or {}).get("training") is False:
            return ()
        if is_dropout or any("random" in token or "rand" in token for token in tokens):
            return ("stateful/random op requires backend-specific replay policy",)
        if any("assign" in token or "inplace" in token or "write" in token for token in tokens):
            return ("state mutation requires backend-specific replay policy",)
        return ()

    def dynamic_shape_reasons(self, node: Any, graph: Any, spec: Any) -> tuple[str, ...]:
        """Provide an adapter extension point for symbolic shape validation."""

        del node, graph, spec
        return ()


class ReplayLowerer(Protocol):
    """Backend hook for lowering a split graph into replay IR."""

    def lower_program(
        self,
        graph: "SplitTraceGraph",
        plan: "SplitPlan",
        spec: "SplitRequest",
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
        spec: "SplitRequest",
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

    def target_support_reasons(self, node: Any, graph: Any) -> tuple[str, ...]:
        """Validate a captured target against backend replay handles."""
        ...

    def stateful_replay_reasons(self, node: Any, backend: str) -> tuple[str, ...]:
        """Report backend-specific stateful replay limitations."""
        ...

    def dynamic_shape_reasons(self, node: Any, graph: Any, spec: Any) -> tuple[str, ...]:
        """Report backend-specific symbolic-shape limitations."""
        ...


__all__ = [
    "ReplayExecutor",
    "ReplayLowerer",
    "SegmentBundle",
    "SplitPolicyMixin",
    "SplitBackendAdapter",
    "TensorOps",
    "TrainingEngine",
    "boundary_overlay",
]
