"""Backend-neutral split replay program and capability preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..backends.registry import TORCH_BACKEND_NAME
from .errors import SplitErrorContext, SplitUnsupportedError
from .graph import SplitTraceGraph, SplitTraceNode
from .ir import (
    GraphPartition,
    SplitGraphIR,
    SplitModelProfile,
    SplitRequest,
    SplitVerificationStatus,
)
from .planner import SplitPlan
from .shape import is_batch_shape_sensitive_op

ReplaySegment = Literal["prefix", "suffix"]


@dataclass(frozen=True)
class CapabilityStatus:
    """One split capability gate result."""

    supported: bool
    reason: str | None = None
    details: tuple[str, ...] = ()
    status: SplitVerificationStatus = SplitVerificationStatus.EXACT

    def as_dict(self) -> dict[str, Any]:
        """Return this status as JSON-like data."""

        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class CapabilityEvidence:
    """Separate implementation, graph, device, and sampled numerical evidence.

    ``None`` means unexamined or unknown, never success. Device evidence checks
    requested device availability only, not free memory or kernel coverage.
    Numerical evidence applies exclusively to the named ``evidence_scope``.
    """

    backend_supported: bool | None
    model_supported: bool | None = None
    device_supported: bool | None = None
    numerically_verified: bool | None = None
    evidence_scope: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


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
    boundary_cache: CapabilityStatus
    preflight: CapabilityStatus
    prefix_ops: int
    suffix_ops: int
    verification: SplitVerificationStatus = SplitVerificationStatus.EXACT
    model_profile: str | None = None
    features: dict[str, Any] | None = None
    node_statuses: dict[str, SplitVerificationStatus] | None = None
    backend_capabilities: dict[str, bool] | None = None
    shape_diagnostics: dict[str, Any] | None = None
    feature_evidence: dict[str, CapabilityEvidence] | None = None

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

        result = asdict(self)
        for field_name in (
            "replay",
            "training",
            "boundary_cache",
            "preflight",
        ):
            status = getattr(self, field_name)
            result[field_name]["status"] = status.status.value
        result["verification"] = self.verification.value
        if self.node_statuses is not None:
            result["node_statuses"] = {
                node_id: status.value for node_id, status in self.node_statuses.items()
            }
        return result


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

    return node.is_input or node.is_buffer or (not node.parents and not node.is_output)


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


def _adapter_policy(adapter: Any, method_name: str, *args: Any) -> tuple[str, ...]:
    """Call an adapter-owned capability policy hook when present."""

    method = getattr(adapter, method_name, None)
    if method is None:
        return ()
    result = method(*args)
    return tuple(str(item) for item in result)


def _dynamic_shape_reasons(
    node: SplitTraceNode,
    graph: SplitTraceGraph,
    spec: SplitRequest,
    adapter: Any,
) -> tuple[str, ...]:
    """Return shape failures that prevent even captured-batch replay.

    Unresolved batch relations remain in the capability diagnostics, but do
    not block preparation: each executing segment's runtime binding calls
    ``ShapeProgram.require_batch_resolvable`` before changing the batch.
    """

    shape_program = graph.shape_program
    if shape_program is None:
        return ()
    adapter_reasons = _adapter_policy(adapter, "dynamic_shape_reasons", node, graph, spec)
    if adapter_reasons:
        return adapter_reasons
    func_name = str(getattr(node.target, "op_type", "") or getattr(node.target, "__name__", ""))
    if not is_batch_shape_sensitive_op(node.op_type, func_name):
        return ()
    if graph.traced_batch_size is None:
        return ("batch-symbolic replay requested but traced batch size is unavailable",)
    return ()


def _unsupported_reasons(
    graph: SplitTraceGraph,
    node: SplitTraceNode,
    node_ids: frozenset[str],
    spec: SplitRequest,
    adapter: Any,
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
    target_reasons = _adapter_policy(adapter, "target_support_reasons", node, graph)
    if not target_reasons and node.target is None and not _is_source_like(node):
        target_reasons = ("missing replay target/capture",)
    reasons.extend(target_reasons)
    reasons.extend(_adapter_policy(adapter, "stateful_replay_reasons", node, graph.backend))
    reasons.extend(_dynamic_shape_reasons(node, graph, spec, adapter))
    return tuple(dict.fromkeys(reasons))


def lower_replay_program(
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitRequest,
    *,
    segment: ReplaySegment,
    adapter: Any | None = None,
) -> ReplayProgram:
    """Lower a split graph segment to backend-neutral replay instructions."""

    if adapter is None:
        from .adapters import resolve_split_adapter

        adapter = resolve_split_adapter(graph.backend)
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
                unsupported_reasons=_unsupported_reasons(graph, node, node_ids, spec, adapter),
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
    spec: SplitRequest,
    *,
    prefix_program: ReplayProgram,
    suffix_program: ReplayProgram,
    graph_ir: SplitGraphIR | None = None,
    model_profile: SplitModelProfile | None = None,
    features: dict[str, Any] | None = None,
) -> SplitCapabilityReport:
    """Build a strict split capability report for prepared programs."""

    program_reasons = (
        *prefix_program.unsupported_reasons,
        *suffix_program.unsupported_reasons,
    )
    verification = _overall_verification(graph_ir, program_reasons)
    preflight = CapabilityStatus(
        supported=not program_reasons,
        reason=None if not program_reasons else "split replay preflight failed",
        details=tuple(program_reasons),
        status=verification if not program_reasons else SplitVerificationStatus.UNSUPPORTED,
    )
    replay_supported = bool(getattr(adapter, "supports_replay", False)) and preflight.supported
    replay = CapabilityStatus(
        supported=replay_supported,
        reason=None if replay_supported else f"backend={graph.backend!r} cannot replay this split",
        details=tuple(program_reasons),
        status=verification if replay_supported else SplitVerificationStatus.UNSUPPORTED,
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
        status=verification if training_supported else SplitVerificationStatus.UNSUPPORTED,
    )
    cache_supported = bool(getattr(adapter, "supports_boundary_cache", False))
    boundary_cache = CapabilityStatus(
        supported=cache_supported,
        reason=None
        if cache_supported
        else f"backend={graph.backend!r} does not support boundary cache",
        status=verification if cache_supported else SplitVerificationStatus.UNSUPPORTED,
    )
    return SplitCapabilityReport(
        backend=graph.backend,
        split_id=plan.split_id,
        replay=replay,
        training=training,
        boundary_cache=boundary_cache,
        preflight=preflight,
        prefix_ops=len(prefix_program.ops),
        suffix_ops=len(suffix_program.ops),
        verification=verification,
        model_profile=None if model_profile is None else model_profile.id,
        features=features,
        node_statuses=_node_statuses(graph_ir),
        backend_capabilities={
            "replay": bool(getattr(adapter, "supports_replay", False)),
            "training": bool(getattr(adapter, "supports_training", False)),
            "boundary_cache": bool(getattr(adapter, "supports_boundary_cache", False)),
            "state_placement": bool(getattr(adapter, "supports_state_placement", False)),
        },
        feature_evidence=_feature_evidence(adapter, graph, spec, preflight.supported),
        shape_diagnostics=(
            None
            if graph.shape_program is None
            else {
                "fingerprint": graph.shape_program.fingerprint,
                "inference_mode": graph.shape_program.inference_mode,
                "traced_batch_size": graph.shape_program.traced_batch_size,
                "batch_validation": (
                    {}
                    if graph.shape_program.batch_probe is None
                    else graph.shape_program.batch_probe.as_dict()
                ),
                "input_batch_axes": dict(graph.shape_program.input_batch_axes),
                "witness_batch_sizes": graph.shape_program.witness_batch_sizes,
                "proof_sources": dict(graph.shape_program.proof_sources),
                "witness_axis_diagnostics": {
                    node_id: dict(diagnostics)
                    for node_id, diagnostics in graph.shape_program.witness_axis_diagnostics.items()
                },
                "unresolved": dict(graph.shape_program.unresolved),
                "recipes": {
                    node_id: [recipe.recipe_id for recipe in recipes]
                    for node_id, recipes in graph.shape_program.recipes.items()
                },
            }
        ),
    )


def _device_evidence(adapter: Any, spec: SplitRequest) -> bool | None:
    """Inspect explicit Torch device availability without allocating replay state."""

    if adapter.name != "torch" or not spec.placement.is_explicit:
        return None
    import torch

    results: list[bool | None] = []
    for placement in (spec.placement.prefix, spec.placement.suffix):
        if not placement.is_explicit:
            results.append(None)
            continue
        try:
            device = torch.device(placement.device)
        except (TypeError, ValueError, RuntimeError):
            return False
        if device.type == "cpu":
            results.append(True)
        elif device.type == "cuda":
            results.append(
                torch.cuda.is_available()
                and (device.index is None or device.index < torch.cuda.device_count())
            )
        else:
            results.append(None)
    if False in results:
        return False
    return True if all(value is True for value in results) else None


def _feature_evidence(
    adapter: Any, graph: SplitTraceGraph, spec: SplitRequest, preflight_ok: bool
) -> dict[str, CapabilityEvidence]:
    """Describe feature evidence without treating static lowering as numerical validation."""

    torch_backend = adapter.name == TORCH_BACKEND_NAME
    replay = bool(getattr(adapter, "supports_replay", False))
    training = bool(getattr(adapter, "supports_training", False))
    cache = bool(getattr(adapter, "supports_boundary_cache", False))
    placement = bool(getattr(adapter, "supports_state_placement", False))
    devices = _device_evidence(adapter, spec)
    static_scope = ("selected_partition_static_lowering",)
    shape = graph.shape_program
    probe = None if shape is None else shape.batch_probe
    probe_passed = (
        None if probe is None or probe.status == "unavailable" else probe.status == "passed"
    )
    numerical_probe = (
        True
        if probe_passed is True
        else False
        if probe is not None and str(probe.reason or "").startswith("compare outputs:")
        else None
    )
    dynamic_model = (
        False
        if probe_passed is False
        else True
        if probe_passed and shape is not None and not shape.unresolved
        else None
    )
    dynamic_scope = (
        ()
        if probe is None
        else (
            f"independent_batch_probe:B={probe.probe_batch_size}",
            "unplaced_forward_replay_only",
            "other_batches_are_extrapolated",
        )
    )
    return {
        "arbitrary_dag_partition": CapabilityEvidence(
            backend_supported=True if torch_backend else None,
            model_supported=preflight_ok if isinstance(spec.point, GraphPartition) else None,
            device_supported=devices,
            evidence_scope=static_scope if isinstance(spec.point, GraphPartition) else (),
            reasons=() if isinstance(spec.point, GraphPartition) else ("partition_not_analyzed",),
        ),
        "split_inference": CapabilityEvidence(
            replay, preflight_ok, devices, evidence_scope=static_scope
        ),
        "split_training": CapabilityEvidence(
            training,
            preflight_ok if spec.trainable else None,
            devices,
            evidence_scope=static_scope if spec.trainable else (),
            reasons=() if spec.trainable else ("training_not_requested",),
        ),
        "compact_training": CapabilityEvidence(
            torch_backend,
            preflight_ok
            if torch_backend and spec.trainable and spec.features.retain_trace is False
            else None,
            devices,
            evidence_scope=("training_source_lifetime_check", "selected_partition_static_lowering")
            if torch_backend and spec.trainable and spec.features.retain_trace is False
            else (),
        ),
        "dynamic_batch": CapabilityEvidence(
            True if torch_backend else None,
            dynamic_model,
            None,
            numerical_probe,
            evidence_scope=dynamic_scope,
            reasons=() if probe is not None else ("batch_probe_unavailable",),
        ),
        "cross_device": CapabilityEvidence(
            placement,
            device_supported=devices,
            evidence_scope=("requested_device_availability_only",),
            reasons=("placement_numerics_not_verified",),
        ),
        "repartition": CapabilityEvidence(
            replay,
            evidence_scope=("same_capture_plan_reuse",),
            reasons=("target_partition_and_current_state_require_analysis",),
        ),
        "state_migration": CapabilityEvidence(
            torch_backend,
            evidence_scope=("model_state_only_optimizer_owned_by_caller",),
            reasons=("specific_migration_not_inspected",),
        ),
        "microbatch": CapabilityEvidence(
            torch_backend or adapter.name == "mlx",
            reasons=("chunk_shapes_and_loss_semantics_require_runtime_validation",),
        ),
        "boundary_cache": CapabilityEvidence(
            cache,
            preflight_ok if cache else False,
            evidence_scope=("authenticated_local_cache_only",),
        ),
    }


def _overall_verification(
    graph_ir: SplitGraphIR | None,
    program_reasons: tuple[str, ...],
) -> SplitVerificationStatus:
    """Return the most conservative verification level for a split graph."""

    if program_reasons:
        return SplitVerificationStatus.UNSUPPORTED
    if graph_ir is None:
        return SplitVerificationStatus.EXACT
    statuses = [operation.verification for operation in graph_ir.ops]
    statuses.extend(region.verification for region in graph_ir.regions)
    if SplitVerificationStatus.UNVERIFIED in statuses:
        return SplitVerificationStatus.UNVERIFIED
    if SplitVerificationStatus.REGION_VERIFIED in statuses:
        return SplitVerificationStatus.REGION_VERIFIED
    if SplitVerificationStatus.UNSUPPORTED in statuses:
        return SplitVerificationStatus.UNSUPPORTED
    return SplitVerificationStatus.EXACT


def _node_statuses(
    graph_ir: SplitGraphIR | None,
) -> dict[str, SplitVerificationStatus] | None:
    """Return verification levels keyed by normalized node ID."""

    if graph_ir is None:
        return None
    result: dict[str, SplitVerificationStatus] = {}
    for operation in graph_ir.ops:
        result[operation.node_id] = operation.verification
    for region in graph_ir.regions:
        result[region.node_id] = region.verification
    return result


def ensure_capability_report_supported(
    report: SplitCapabilityReport,
    spec: SplitRequest,
) -> None:
    """Raise a structured unsupported error when a split report is blocked."""

    blocking_reasons: list[str] = []
    if not report.preflight.supported:
        blocking_reasons.extend(report.preflight.details)
    if not report.replay.supported:
        blocking_reasons.append(report.replay.reason or "split replay unsupported")
    if spec.trainable and not report.training.supported:
        blocking_reasons.append(report.training.reason or "split training unsupported")
    if not blocking_reasons:
        return
    raise SplitUnsupportedError(
        "Split replay preflight rejected this split: " + "; ".join(blocking_reasons),
        context=SplitErrorContext(
            backend=report.backend,
            split_point=spec.boundary,
            reason="split preflight rejected",
        ),
    )


__all__ = [
    "CapabilityEvidence",
    "CapabilityStatus",
    "ReplayOp",
    "ReplayProgram",
    "ReplaySegment",
    "SplitCapabilityReport",
    "build_capability_report",
    "ensure_capability_report_supported",
    "lower_replay_program",
]
