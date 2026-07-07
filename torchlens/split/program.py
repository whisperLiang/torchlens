"""Backend-neutral split replay program and capability preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .errors import SplitErrorContext, SplitUnsupportedError
from .graph import SplitTraceGraph, SplitTraceNode
from .planner import SplitPlan
from .shape import is_dynamic_batch_shape_sensitive_op
from .spec import SplitSpec


ReplaySegment = Literal["prefix", "suffix"]


@dataclass(frozen=True)
class CapabilityStatus:
    """One split capability gate result."""

    supported: bool
    reason: str | None = None
    details: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return this status as JSON-like data."""

        return asdict(self)


@dataclass(frozen=True)
class ReplayOp:
    """Backend-neutral replay instruction for one trace node."""

    node_id: str
    label: str
    backend: str
    op_type: str
    module_path: str | None
    target_kind: str
    parents: tuple[str, ...]
    output_container_path: tuple[Any, ...]
    replay_source_policy: str
    dynamic_shape_policy: Literal["static", "leading_batch_symbolic"]
    requires_live_params: bool
    unsupported_reasons: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        """Return whether this op passed lowering preflight."""

        return not self.unsupported_reasons


@dataclass(frozen=True)
class ReplayProgram:
    """Lowered replay program for one split segment."""

    backend: str
    segment: ReplaySegment
    split_id: str
    node_ids: frozenset[str]
    input_node_ids: tuple[str, ...]
    output_node_ids: tuple[str, ...]
    boundary_node_ids: tuple[str, ...]
    ops: tuple[ReplayOp, ...]

    @property
    def unsupported_reasons(self) -> tuple[str, ...]:
        """Return flattened unsupported reasons for the program."""

        reasons: list[str] = []
        for op in self.ops:
            reasons.extend(f"{op.label}: {reason}" for reason in op.unsupported_reasons)
        return tuple(reasons)

    @property
    def requires_live_params(self) -> bool:
        """Return whether any op in this program requires live parameter handles."""

        return any(op.requires_live_params for op in self.ops)


@dataclass(frozen=True)
class SplitCapabilityReport:
    """Prepared split capability and preflight report."""

    backend: str
    split_id: str
    replay: CapabilityStatus
    training: CapabilityStatus
    dynamic_batch: CapabilityStatus
    boundary_cache: CapabilityStatus
    preflight: CapabilityStatus
    prefix_ops: int
    suffix_ops: int

    @property
    def preflight_ok(self) -> bool:
        """Return whether replay preflight passed."""

        return self.preflight.supported

    @property
    def unsupported_reasons(self) -> tuple[str, ...]:
        """Return report reasons that block replay or requested training/dynamic replay."""

        reasons: list[str] = []
        for status in (
            self.replay,
            self.training,
            self.dynamic_batch,
            self.preflight,
        ):
            if status.supported:
                continue
            if status.reason is not None:
                reasons.append(status.reason)
            reasons.extend(status.details)
        return tuple(reasons)

    def as_dict(self) -> dict[str, Any]:
        """Return this report as JSON-like data."""

        return asdict(self)


def _target_kind(node: SplitTraceNode) -> str:
    """Return a stable target-kind label for a replay node."""

    if node.is_input:
        return "input"
    if node.is_output and node.target is None:
        return "output"
    if node.target is None:
        return "missing"
    target_type = type(node.target).__name__
    if callable(node.target):
        return "callable"
    return target_type


def _is_source_like(node: SplitTraceNode) -> bool:
    """Return whether a node may be seeded from trace/source state."""

    return node.is_buffer or (not node.parents and not node.is_output)


def _group_has_executor(
    graph: SplitTraceGraph,
    node: SplitTraceNode,
    node_ids: frozenset[str],
) -> bool:
    """Return whether a grouped op has another executable group member."""

    if node.func_call_id is None:
        return False
    return any(
        candidate.func_call_id == node.func_call_id
        and candidate.canonical_id in node_ids
        and candidate.target is not None
        for candidate in graph.nodes
    )


def _target_is_jax_region(target: Any) -> bool:
    """Return whether a target is a JAX control-flow region capture."""

    return type(target).__name__ == "JaxRegionCapture"


def _backend_target_supported(node: SplitTraceNode, graph: SplitTraceGraph) -> tuple[str, ...]:
    """Return backend-specific target preflight failures."""

    if node.is_input or (node.is_output and node.target is None) or _is_source_like(node):
        return ()
    if node.target is None:
        return ("missing replay target/capture",)
    backend = graph.backend
    target_type = type(node.target).__name__
    if backend == "jax":
        if _target_is_jax_region(node.target):
            return ("unsupported JAX control-flow/region replay",)
        if target_type != "JaxEquationCapture":
            return (f"unsupported JAX target {target_type!r}",)
    elif backend in {"tf", "tensorflow"} and target_type != "TFOpCapture":
        return (f"unsupported TensorFlow target {target_type!r}",)
    elif backend == "tinygrad" and target_type != "TinygradUOpCapture":
        return (f"unsupported tinygrad target {target_type!r}",)
    elif backend in {"torch", "paddle"} and not callable(node.target):
        return (f"unsupported callable target {target_type!r}",)
    return ()


def _stateful_replay_reasons(node: SplitTraceNode, backend: str) -> tuple[str, ...]:
    """Return conservative stateful/random replay failures."""

    if backend == "torch":
        return ()
    op_text = node.op_type.lower()
    target_name = str(getattr(node.target, "op_type", "") or getattr(node.target, "__name__", ""))
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


def _dynamic_shape_reasons(
    node: SplitTraceNode,
    graph: SplitTraceGraph,
    spec: SplitSpec,
) -> tuple[str, ...]:
    """Return dynamic-shape preflight failures for a node."""

    if spec.dynamic_batch is None:
        return ()
    func_name = str(getattr(node.target, "op_type", "") or getattr(node.target, "__name__", ""))
    if not is_dynamic_batch_shape_sensitive_op(node.op_type, func_name):
        return ()
    if graph.traced_batch_size is None:
        return ("dynamic batch requested but traced batch size is unavailable",)
    return ()


def _unsupported_reasons(
    graph: SplitTraceGraph,
    node: SplitTraceNode,
    node_ids: frozenset[str],
    spec: SplitSpec,
) -> tuple[str, ...]:
    """Return strict preflight failures for one replay node."""

    if node.canonical_id not in node_ids:
        return ()
    if (
        node.target is None
        and not node.is_input
        and not (node.is_output and node.target is None)
        and not _is_source_like(node)
        and _group_has_executor(graph, node, node_ids)
    ):
        return ()
    reasons: list[str] = []
    reasons.extend(_backend_target_supported(node, graph))
    reasons.extend(_stateful_replay_reasons(node, graph.backend))
    reasons.extend(_dynamic_shape_reasons(node, graph, spec))
    return tuple(dict.fromkeys(reasons))


def lower_replay_program(
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitSpec,
    *,
    segment: ReplaySegment,
) -> ReplayProgram:
    """Lower a split graph segment to backend-neutral replay instructions."""

    node_ids = plan.prefix_node_ids if segment == "prefix" else plan.suffix_node_ids
    ops: list[ReplayOp] = []
    for node in graph.nodes:
        if node.canonical_id not in node_ids:
            continue
        dynamic_shape_policy: Literal["static", "leading_batch_symbolic"] = "static"
        if node.symbolic_output_shape is not None and any(
            dim == spec.batch_symbol for dim in node.symbolic_output_shape
        ):
            dynamic_shape_policy = "leading_batch_symbolic"
        ops.append(
            ReplayOp(
                node_id=node.canonical_id,
                label=node.label,
                backend=node.backend,
                op_type=node.op_type,
                module_path=node.module_path,
                target_kind=_target_kind(node),
                parents=node.parents,
                output_container_path=node.output_container_path,
                replay_source_policy=node.replay_source_policy,
                dynamic_shape_policy=dynamic_shape_policy,
                requires_live_params=bool(node.param_refs or node.is_param_source),
                unsupported_reasons=_unsupported_reasons(graph, node, node_ids, spec),
            )
        )
    return ReplayProgram(
        backend=graph.backend,
        segment=segment,
        split_id=plan.split_id,
        node_ids=node_ids,
        input_node_ids=graph.input_node_ids,
        output_node_ids=graph.output_node_ids,
        boundary_node_ids=plan.boundary_node_ids,
        ops=tuple(ops),
    )


def build_capability_report(
    adapter: Any,
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitSpec,
    *,
    prefix_program: ReplayProgram,
    suffix_program: ReplayProgram,
) -> SplitCapabilityReport:
    """Build a strict split capability report for prepared programs."""

    program_reasons = (
        *prefix_program.unsupported_reasons,
        *suffix_program.unsupported_reasons,
    )
    preflight = CapabilityStatus(
        supported=not program_reasons,
        reason=None if not program_reasons else "split replay preflight failed",
        details=tuple(program_reasons),
    )
    replay_supported = bool(getattr(adapter, "supports_replay", False)) and preflight.supported
    replay = CapabilityStatus(
        supported=replay_supported,
        reason=None if replay_supported else f"backend={graph.backend!r} cannot replay this split",
        details=tuple(program_reasons),
    )
    training_requested = spec.trainable
    training_supported = (not training_requested) or bool(
        getattr(adapter, "supports_training", False)
    )
    training = CapabilityStatus(
        supported=training_supported,
        reason=None
        if training_supported
        else f"backend={graph.backend!r} does not support split training",
    )
    dynamic_requested = spec.dynamic_batch is not None
    dynamic_supported = (not dynamic_requested) or bool(
        getattr(adapter, "supports_dynamic_batch", False)
    )
    dynamic_batch = CapabilityStatus(
        supported=dynamic_supported,
        reason=None
        if dynamic_supported
        else f"backend={graph.backend!r} does not support dynamic-batch split replay",
    )
    cache_supported = bool(getattr(adapter, "supports_boundary_cache", False))
    boundary_cache = CapabilityStatus(
        supported=cache_supported,
        reason=None
        if cache_supported
        else f"backend={graph.backend!r} does not support boundary cache",
    )
    return SplitCapabilityReport(
        backend=graph.backend,
        split_id=plan.split_id,
        replay=replay,
        training=training,
        dynamic_batch=dynamic_batch,
        boundary_cache=boundary_cache,
        preflight=preflight,
        prefix_ops=len(prefix_program.ops),
        suffix_ops=len(suffix_program.ops),
    )


def ensure_capability_report_supported(
    report: SplitCapabilityReport,
    spec: SplitSpec,
) -> None:
    """Raise a structured unsupported error when a split report is blocked."""

    blocking_reasons: list[str] = []
    if not report.preflight.supported:
        blocking_reasons.extend(report.preflight.details)
    if not report.replay.supported:
        blocking_reasons.append(report.replay.reason or "split replay unsupported")
    if spec.trainable and not report.training.supported:
        blocking_reasons.append(report.training.reason or "split training unsupported")
    if spec.dynamic_batch is not None and not report.dynamic_batch.supported:
        blocking_reasons.append(report.dynamic_batch.reason or "dynamic batch unsupported")
    if not blocking_reasons:
        return
    raise SplitUnsupportedError(
        "Split replay preflight rejected this split: " + "; ".join(blocking_reasons),
        context=SplitErrorContext(
            backend=report.backend,
            split_point=spec.boundary,
            module_path=None,
            op_type=None,
            layer_label=None,
            reason="split preflight rejected",
        ),
    )


__all__ = [
    "CapabilityStatus",
    "ReplayOp",
    "ReplayProgram",
    "ReplaySegment",
    "SplitCapabilityReport",
    "build_capability_report",
    "ensure_capability_report_supported",
    "lower_replay_program",
]
