"""Named phases of the backend-neutral split preparation pipeline."""

from __future__ import annotations

from typing import Any

from ..backends import resolve_backend_spec
from .adapters.base import SegmentBundle
from .graph import SplitTraceGraph, split_graph_from_trace
from .ir import SplitGraphIR, SplitModelProfile, SplitRequest
from .planner import SplitPlan, plan_split
from .program import (
    ReplayProgram,
    ReplaySegment,
    SplitCapabilityReport,
    build_capability_report,
    lower_replay_program,
)


def capture_model(
    model: Any,
    inputs: tuple[Any, ...],
    spec: SplitRequest,
    *,
    input_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Capture one complete model execution with the native backend."""

    backend_spec = resolve_backend_spec(spec.backend, model, inputs, input_kwargs)
    from ..user_funcs import trace

    common: dict[str, Any] = {
        "input_kwargs": input_kwargs,
        "layers_to_save": "all",
        "keep_orphans": True,
        "backend": str(backend_spec.name),
    }
    if str(backend_spec.name) == "torch":
        common.update(
            {
                "intervention_ready": True,
                "capture_container_structure": True,
                "save_arg_values": True,
                "save_rng_states": True,
                "detach_saved_activations": False,
                "backward_ready": spec.trainable,
            }
        )
    return trace(model, inputs, **common)


def normalize_to_split_ir(
    capture: Any,
    spec: SplitRequest,
    *,
    plan: SplitPlan | None = None,
    model_profile: SplitModelProfile | None = None,
) -> tuple[SplitTraceGraph, SplitGraphIR]:
    """Normalize a backend capture into the runtime graph and portable Split IR."""

    graph = (
        capture
        if isinstance(capture, SplitTraceGraph)
        else split_graph_from_trace(
            capture,
            batch_symbol=spec.batch_symbol,
            dynamic_batch=spec.dynamic_batch,
        )
    )
    resolved_plan = plan if plan is not None else plan_split(graph, spec)
    return graph, SplitGraphIR.from_trace_graph(
        graph,
        plan=resolved_plan,
        profile_hash=None if model_profile is None else model_profile.profile_hash,
    )


def analyze_split_capabilities(
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
    """Analyze replay, dynamic-shape, training, and cache capabilities."""

    return build_capability_report(
        adapter,
        graph,
        plan,
        spec,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
        graph_ir=graph_ir,
        model_profile=model_profile,
        features=features,
    )


def lower_split_program(
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitRequest,
    *,
    adapter: Any,
    segment: ReplaySegment,
) -> ReplayProgram:
    """Lower one IR segment through the selected backend adapter."""

    return lower_replay_program(graph, plan, spec, segment=segment, adapter=adapter)


def execute_split_runtime(
    adapter: Any,
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitRequest,
) -> SegmentBundle:
    """Build executable backend recipes from a lowered split plan."""

    return adapter.build_segments(graph, plan, spec)


__all__ = [
    "analyze_split_capabilities",
    "capture_model",
    "execute_split_runtime",
    "lower_split_program",
    "normalize_to_split_ir",
]
