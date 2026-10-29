"""Resolve test selection recipes after capture; production only receives GraphPartition."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from torchlens.split import (
    GraphPartition,
    PlacementPlan,
    SplitFeatures,
    SplitRequest,
    prepare_graph,
)
from torchlens.split.errors import SplitRequestError, SplitUnsupportedError
from torchlens.split.planner import plan_partition


@dataclass(frozen=True)
class FixtureRequest:
    """Capture policies and a test-only recipe for selecting canonical IDs."""

    partition: GraphPartition | str
    backend: str | None = None
    features: SplitFeatures = SplitFeatures()
    placement: Any = PlacementPlan()
    validation: str = "strict"
    model_profile: Any = None

    @property
    def trainable(self) -> bool:
        """Capture-only fixtures expose the requested training policy."""
        return self.features.training

    @property
    def boundary(self) -> str:
        """Use fixture identity only for capture error context."""
        return (
            self.partition.as_boundary()
            if isinstance(self.partition, GraphPartition)
            else self.partition
        )

    @property
    def use_live_param_sources(self) -> bool | None:
        """Pass capture source policy through unchanged."""
        return self.features.live_param_sources

    batch_symbol: str = "B"
    device_policy: str = "runtime"


def request_fixture(selection: GraphPartition | str, **kwargs: Any) -> FixtureRequest:
    """Defer fixture selection until the graph exists."""
    return FixtureRequest(selection, **kwargs)


def split_request(
    boundary: GraphPartition | str,
    *,
    trainable: bool = False,
    boundary_cache: bool = False,
    live_param_sources: bool | None = None,
    batch_axes: dict[str, int] | None = None,
    retain_trace: bool | None = None,
    **kwargs: Any,
) -> FixtureRequest:
    """Keep backend matrix options separate from canonical node selection."""
    return request_fixture(
        boundary,
        features=SplitFeatures(
            training=trainable,
            boundary_cache=boundary_cache,
            live_param_sources=live_param_sources,
            batch_axes=batch_axes,
            retain_trace=retain_trace,
        ),
        **kwargs,
    )


def fixture_partition(subject: Any, selection: GraphPartition | str) -> GraphPartition:
    """Choose a native-call-atomic topological prefix for a fixture."""
    if isinstance(selection, GraphPartition):
        return selection
    graph = getattr(subject, "trace_graph", subject)
    compute = graph.compute_nodes
    if selection.startswith(("through:", "until:")):
        strategy, label = selection.split(":", 1)
        matches = [
            n for n in compute if label in {n.canonical_id, n.label, n.raw_label, n.module_path}
        ]
        if not matches:
            matches = [
                n
                for n in compute
                if any(
                    label in str(v)
                    for v in (n.canonical_id, n.label, n.raw_label, n.module_path)
                    if v is not None
                )
            ]
        if len(matches) != 1:
            raise SplitRequestError(
                f"Fixture selection {label!r} matched {len(matches)} compute nodes (ambiguous or missing)."
            )
        call = graph.replay_call_by_output_id[matches[0].canonical_id]
        indexes = [i for i, n in enumerate(compute) if n.canonical_id in call.output_node_ids]
        count = max(indexes) + 1 if strategy == "through" else min(indexes)
    else:
        fraction = float(selection.removeprefix("fraction:").removesuffix("%"))
        if not 0 < fraction < 100:
            raise ValueError("fixture fraction must be between 0 and 100")
        count = int(fraction / 100 * (len(compute) - 1)) + 1
        call = graph.replay_call_by_output_id[compute[count - 1].canonical_id]
        count = max(i + 1 for i, n in enumerate(compute) if n.canonical_id in call.output_node_ids)
    return GraphPartition(frozenset(n.canonical_id for n in compute[:count]))


def capture_fixture(
    model: Any, inputs: Any, request: FixtureRequest, *, input_kwargs: dict[str, Any] | None = None
) -> Any:
    """Capture unsupported partition fixtures without constructing executable segments."""
    return prepare_graph(
        model,
        inputs,
        backend=request.backend,
        features=request.features,
        placement=request.placement,
        validation=request.validation,
        input_kwargs=input_kwargs,
    )


def resolved_request(graph: Any, request: FixtureRequest | SplitRequest) -> SplitRequest:
    """Build the real request after resolving fixture nodes."""
    if isinstance(request, SplitRequest):
        return request
    return SplitRequest(
        fixture_partition(graph, request.partition),
        backend=request.backend,
        features=request.features,
        placement=request.placement,
        validation=request.validation,
    )


def prepare_fixture(
    model: Any,
    inputs: Any,
    request: FixtureRequest | SplitRequest,
    *,
    input_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Exercise capture, canonical selection, analysis and materialization."""
    options = {
        "backend": request.backend,
        "features": request.features,
        "validation": request.validation,
        "model_profile": request.model_profile,
        "input_kwargs": input_kwargs,
    }
    if request.placement is not None:
        options["placement"] = request.placement
    graph = prepare_graph(model, inputs, **options)
    return graph.materialize(graph.analyze(fixture_partition(graph, request.partition)))


def analyze_fixture(runtime: Any, selection: GraphPartition | str) -> Any:
    """Analyze the selected canonical set."""
    return runtime.analyze(fixture_partition(runtime, selection))


def repartition_fixture(runtime: Any, selection: GraphPartition | str) -> Any:
    """Repartition using the same graph and current state."""
    return runtime.repartition(fixture_partition(runtime, selection))


def plan_fixture(graph: Any, request: FixtureRequest | SplitRequest) -> Any:
    """Exercise the sole production planner with canonical IDs."""
    if isinstance(request, SplitRequest):
        return plan_partition(graph, request)
    return plan_partition(
        graph,
        SplitRequest(
            fixture_partition(graph, request.partition),
            backend=request.backend,
            features=request.features,
            validation=request.validation,
            **({} if request.placement is None else {"placement": request.placement}),
        ),
    )


def candidate_report(runtime: Any, *, diagnose: bool = True) -> Any:
    """Collect diagnostics over bounded native-call-atomic partition candidates."""
    rows = []
    for partition in runtime.partition_candidates():
        try:
            analysis = runtime.analyze(partition) if diagnose else None
            report = None if analysis is None else analysis.capability_report
            reasons = () if report is None else report.unsupported_reasons
        except (SplitUnsupportedError, SplitRequestError) as exc:
            analysis, report, reasons = None, None, (str(exc),)
        rows.append(
            SimpleNamespace(
                partition=partition,
                replay_supported=not reasons,
                supported=not reasons,
                training_supported=bool(report and report.training.supported),
                unsupported_reasons=reasons,
                unsupported_reason="; ".join(reasons) or None,
                boundary_value_ids=() if analysis is None else analysis.plan.boundary_node_ids,
                boundary_schema=() if analysis is None else analysis.graph_ir.boundary_schema,
            )
        )
    supported = tuple(r for r in rows if r.replay_supported)
    unsupported = tuple(r for r in rows if not r.replay_supported)
    return SimpleNamespace(
        candidates=tuple(rows),
        total=len(rows),
        supported=supported,
        unsupported=unsupported,
        trainable=tuple(r for r in rows if r.training_supported),
        reasons=lambda: {r.partition.as_boundary(): r.unsupported_reasons for r in unsupported},
    )
