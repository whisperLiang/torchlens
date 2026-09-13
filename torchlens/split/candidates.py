"""Fine-grained split-point enumeration with explicit unsupported reasons.

Every semantically valid before/after computation boundary in the normalized
captured graph is a candidate.  A candidate is never silently skipped: it is
either supported, or it carries a deterministic structured reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .ir import BoundarySchema, SplitPoint, after, before

if TYPE_CHECKING:
    from .graph import SplitTraceGraph


@dataclass(frozen=True)
class SplitCandidate:
    """One enumerated fine-grained split boundary and its diagnostics."""

    point: SplitPoint
    kind: Literal["before", "after"]
    node_id: str
    label: str
    op_type: str
    module_path: str | None
    boundary_value_ids: tuple[str, ...] = ()
    boundary_schema: tuple[BoundarySchema, ...] = ()
    replay_supported: bool = False
    training_supported: bool = False
    unsupported_reasons: tuple[str, ...] = ()
    shape_unresolved: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        """Return whether this boundary can be replayed."""

        return self.replay_supported

    @property
    def unsupported_reason(self) -> str | None:
        """Return the joined deterministic reason, if any."""

        if not self.unsupported_reasons:
            return None
        return "; ".join(self.unsupported_reasons)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like candidate metadata."""

        return {
            "point": {"kind": self.kind, "target": self.node_id},
            "node_id": self.node_id,
            "label": self.label,
            "op_type": self.op_type,
            "module_path": self.module_path,
            "boundary_value_ids": self.boundary_value_ids,
            "replay_supported": self.replay_supported,
            "training_supported": self.training_supported,
            "unsupported_reasons": self.unsupported_reasons,
            "shape_unresolved": self.shape_unresolved,
        }


@dataclass(frozen=True)
class SplitCandidateReport:
    """Aggregate view over every enumerated boundary in one capture."""

    candidates: tuple[SplitCandidate, ...] = ()

    @property
    def total(self) -> int:
        """Return the number of enumerated boundaries."""

        return len(self.candidates)

    @property
    def supported(self) -> tuple[SplitCandidate, ...]:
        """Return the replayable boundaries."""

        return tuple(candidate for candidate in self.candidates if candidate.replay_supported)

    @property
    def unsupported(self) -> tuple[SplitCandidate, ...]:
        """Return the boundaries that carry a deterministic refusal."""

        return tuple(candidate for candidate in self.candidates if not candidate.replay_supported)

    @property
    def trainable(self) -> tuple[SplitCandidate, ...]:
        """Return the boundaries that also support split training."""

        return tuple(candidate for candidate in self.candidates if candidate.training_supported)

    def reasons(self) -> dict[str, tuple[str, ...]]:
        """Return unsupported reasons keyed by boundary spelling."""

        return {
            candidate.point.as_boundary(): candidate.unsupported_reasons
            for candidate in self.unsupported
        }

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like report metadata."""

        return {
            "total": self.total,
            "supported": len(self.supported),
            "unsupported": len(self.unsupported),
            "trainable": len(self.trainable),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class _CandidateSite:
    """One targetable compute node paired with its boundary kinds."""

    node_id: str
    label: str
    op_type: str
    module_path: str | None
    kinds: tuple[str, ...] = field(default=("before", "after"))


def iter_candidate_sites(graph: SplitTraceGraph) -> tuple[_CandidateSite, ...]:
    """Return every targetable compute node in captured execution order."""

    return tuple(
        _CandidateSite(
            node_id=node.canonical_id,
            label=node.label,
            op_type=node.op_type,
            module_path=node.module_path,
        )
        for node in graph.compute_nodes
    )


def point_for(kind: str, node_id: str) -> SplitPoint:
    """Return the typed split point for one boundary kind and node."""

    return before(node_id) if kind == "before" else after(node_id)


__all__ = [
    "SplitCandidate",
    "SplitCandidateReport",
    "iter_candidate_sites",
    "point_for",
]
