"""Build generated eager split segments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .frontier import SplitPlan
from .generated import GeneratedPrefix, GeneratedSuffix
from .spec import SplitMode
from .trace_graph import TraceGraph


@dataclass(frozen=True)
class SegmentBundle:
    """Generated prefix/suffix modules."""

    prefix: nn.Module
    training_prefix: nn.Module
    suffix: nn.Module


def build_segments(
    graph: TraceGraph,
    plan: SplitPlan,
    *,
    mode: SplitMode = "generated_eager",
) -> SegmentBundle:
    """Build generated eager prefix, training prefix, and suffix modules."""

    prefix: nn.Module = GeneratedPrefix(graph=graph, plan=plan, detach_outputs=True)
    training_prefix: nn.Module = GeneratedPrefix(graph=graph, plan=plan, detach_outputs=False)
    suffix: nn.Module = GeneratedSuffix(graph=graph, plan=plan)
    if mode == "compiled":
        prefix = _compile_segment(prefix)
        training_prefix = _compile_segment(training_prefix)
        suffix = _compile_segment(suffix)
    return SegmentBundle(
        prefix=prefix,
        training_prefix=training_prefix,
        suffix=suffix,
    )


class _CompiledOrEager(nn.Module):
    """Compiled segment wrapper that falls back to eager if compilation fails."""

    def __init__(self, segment: nn.Module) -> None:
        super().__init__()
        self.segment = segment
        self.compiled_segment = torch.compile(segment, dynamic=True, fullgraph=False)
        self.active_backend = "compiled"
        self.fallback_reason: str | None = None

    def forward(self, *args: object, **kwargs: object) -> object:
        """Run compiled segment, falling back to eager on backend failure."""

        if self.fallback_reason is not None:
            return self.segment(*args, **kwargs)
        try:
            return self.compiled_segment(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - backend/environment dependent.
            self.fallback_reason = str(exc)
            self.active_backend = "eager"
            return self.segment(*args, **kwargs)


def _compile_segment(segment: nn.Module) -> nn.Module:
    """Return a dynamic compile wrapper for a generated segment."""

    return _CompiledOrEager(segment)


__all__ = ["SegmentBundle", "build_segments"]
