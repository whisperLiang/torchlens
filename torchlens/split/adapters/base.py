"""Split backend adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..graph import SplitTraceGraph
    from ..planner import SplitPlan
    from ..spec import SplitSpec


@dataclass(frozen=True)
class SegmentBundle:
    """Backend-built prefix/suffix segment bundle."""

    prefix: Any
    training_prefix: Any | None
    suffix: Any


class SplitBackendAdapter(Protocol):
    """Protocol implemented by split replay backend adapters."""

    name: str
    supports_replay: bool
    supports_training: bool
    supports_boundary_cache: bool
    supports_dynamic_batch: bool

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

    def build_segments(
        self,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitSpec,
    ) -> SegmentBundle:
        """Build backend-specific split replay segments."""
        ...


__all__ = ["SegmentBundle", "SplitBackendAdapter"]
