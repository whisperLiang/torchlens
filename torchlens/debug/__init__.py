"""Power-user debugging helpers for completed TorchLens traces."""

from __future__ import annotations

from ._cost import hot_path
from ._audit import AuditFinding, TraceAudit, audit_trace
from ._dtype_range import DTypeRangeAudit, dtype_range_audit
from ._graph import LineageResult, compare, dead_neurons, lineage
from ._gradients import gradient_flow_audit
from ._graph_breaks import (
    GraphBreak,
    GraphBreakReport,
    GraphBreaksNormalizationError,
    GraphBreaksUnavailableError,
    graph_breaks,
)
from ._infer_input_shape import InferInputShapeResult, infer_input_shape
from ._nan import BisectNanResult, FindNanResult, bisect_nan, find_nan
from ._recompute import recompute_candidates

__all__ = [
    "BisectNanResult",
    "AuditFinding",
    "DTypeRangeAudit",
    "FindNanResult",
    "GraphBreak",
    "GraphBreakReport",
    "GraphBreaksNormalizationError",
    "GraphBreaksUnavailableError",
    "InferInputShapeResult",
    "LineageResult",
    "TraceAudit",
    "audit_trace",
    "bisect_nan",
    "find_nan",
    "compare",
    "dead_neurons",
    "dtype_range_audit",
    "gradient_flow_audit",
    "graph_breaks",
    "hot_path",
    "infer_input_shape",
    "lineage",
    "recompute_candidates",
]
