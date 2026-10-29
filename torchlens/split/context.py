"""Captured graph context shared by analysis and executable split runtimes."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..capture import flops as capture_flops
from .adapters.base import SegmentBundle, SplitBackendAdapter
from .batching import BatchSpec
from .candidates import PartitionAnalysis, topological_partitions
from .errors import SplitErrorContext, SplitUnsupportedError
from .graph import SplitTraceGraph
from .ir import GraphPartition, SplitGraphIR, SplitModelProfile, SplitRequest
from .pipeline import analyze_split_capabilities, execute_split_runtime, lower_split_program
from .placement import PlacementPlan
from .planner import SplitPlan, plan_partition
from .program import ensure_capability_report_supported

if TYPE_CHECKING:
    from .graph_profile import GraphProfile
    from .runtime import SplitRuntime


class SplitGraph:
    """One captured graph, shape contract and bounded immutable analysis cache.

    This context has no executable segments or mutable client training replicas.
    Materialization creates fresh segments and binds the live model state.
    """

    def __init__(
        self,
        *,
        model: Any,
        trace: Any,
        trace_graph: SplitTraceGraph,
        request: SplitRequest,
        adapter: SplitBackendAdapter,
        graph_ir: SplitGraphIR | None = None,
        model_profile: SplitModelProfile | None = None,
        prepared_input_kwargs: dict[str, Any] | None = None,
        batch_spec: BatchSpec | None = None,
    ) -> None:
        """Retain one normalized graph without constructing a dummy runtime."""
        self.model = model
        self._trace = trace
        self.trace_graph = trace_graph
        self.request = request
        self.adapter = adapter
        self.graph_ir = graph_ir
        self.model_profile = model_profile
        self.prepared_input_kwargs = dict(prepared_input_kwargs or {})
        self.batch_spec = batch_spec
        self._analysis_cache: OrderedDict[tuple[Any, ...], PartitionAnalysis] = OrderedDict()
        self._profile_cache: OrderedDict[tuple[int | None, int], GraphProfile] = OrderedDict()
        self._cache_graph = trace_graph
        self._cache_counts = {"hits": 0, "misses": 0}

    def partition_candidates(self) -> tuple[GraphPartition, ...]:
        """Return optional topological prefixes without analyzing or executing them."""
        return topological_partitions(self.trace_graph)

    @contextmanager
    def _temporary_segment_state_snapshots(self) -> Iterator[dict[str, dict[str, Any]]]:
        """A capture-only graph has no segment replicas to inspect."""
        yield {}

    def _validate_recut_state_for_plan(
        self, plan: SplitPlan, *, state_snapshots: dict[str, dict[str, Any]], split_point: str
    ) -> None:
        """A first materialization has no prior client state to merge."""

    def _inherit_segment_state(
        self,
        segments: SegmentBundle,
        *,
        recut: bool,
        state_snapshots: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """First materialization binds current sources lazily in the adapter."""

    @property
    def retains_trace(self) -> bool:
        """Return whether the runtime owns a complete diagnostic capture."""

        return self._trace is not None

    @property
    def trace(self) -> Any:
        """Return the optional diagnostic Trace, refusing compact-runtime access.

        Raises
        ------
        SplitUnsupportedError
            If preparation discarded the diagnostic capture. Prepare with
            ``SplitFeatures(retain_trace=True)`` to inspect saved activations.
            Graph metadata remains available through ``trace_graph`` in either mode.
        """

        if self._trace is None:
            raise SplitUnsupportedError(
                "This compact split runtime does not retain a diagnostic Trace. "
                "Prepare with SplitFeatures(retain_trace=True) to inspect historical "
                "activations, or use runtime.trace_graph for execution graph metadata.",
                context=SplitErrorContext(
                    backend=self.trace_graph.backend,
                    split_point=self.request.boundary,
                    reason="diagnostic_trace_not_retained",
                ),
            )
        return self._trace

    @property
    def graph_identity(self) -> str | None:
        """Stable identity of the reused capture/graph."""

        return None if self.graph_ir is None else self.graph_ir.graph_hash

    @property
    def traced_batch_size(self) -> int | None:
        """Canonical capture batch used to produce this runtime's graph."""

        if self.trace_graph.shape_program is not None:
            return self.trace_graph.shape_program.traced_batch_size
        return self.trace_graph.traced_batch_size

    @property
    def placement(self) -> PlacementPlan:
        """Declared prefix/suffix placement for this runtime."""

        return self.request.placement

    @property
    def batch_validation(self) -> dict[str, Any]:
        """Disclose the sampled scope separately from captured-graph verification."""

        program = self.trace_graph.shape_program
        probe = None if program is None else program.batch_probe
        return {} if probe is None else probe.as_dict()

    def graph_profile(self, *, batch_size: int | None = None) -> GraphProfile:
        """Describe static graph costs without executing the model.

        Parameters
        ----------
        batch_size
            Batch to evaluate using the retained ShapeProgram; defaults to the
            capture batch. Unknown costs remain None, including unsupported FLOPs.

        Returns
        -------
        GraphProfile
            Immutable node, value and state metadata for placement algorithms.
        """

        from .graph_profile import build_graph_profile

        if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        self._check_cache_graph()
        key = (batch_size, capture_flops._rule_revision)
        if key not in self._profile_cache:
            self._profile_cache[key] = build_graph_profile(
                self.trace_graph, model=self.model, batch_size=batch_size
            )
            if len(self._profile_cache) > 8:
                self._profile_cache.popitem(last=False)
        self._profile_cache.move_to_end(key)
        return self._profile_cache[key]

    def clear_analysis_cache(self) -> None:
        """Invalidate this capture lineage's bounded plan and graph-cost caches."""

        self._analysis_cache.clear()
        self._profile_cache.clear()
        self._cache_counts.update(hits=0, misses=0)

    def analysis_cache_info(self) -> dict[str, int]:
        """Return capacity and hit/miss counts without exposing cached plans."""

        return {**self._cache_counts, "size": len(self._analysis_cache), "capacity": 32}

    def _check_cache_graph(self) -> None:
        """Detach caches when a runtime leaves its shared capture lineage."""

        if self._cache_graph is not self.trace_graph:
            self._analysis_cache = OrderedDict()
            self._profile_cache = OrderedDict()
            self._cache_counts = {"hits": 0, "misses": 0}
            self._cache_graph = self.trace_graph

    def analyze(self, partition: GraphPartition) -> PartitionAnalysis:
        """Validate a GraphPartition without building segments or binding current state."""
        if not isinstance(partition, GraphPartition):
            raise TypeError("partition must be a GraphPartition")
        with self._temporary_segment_state_snapshots() as state_snapshots:
            return self._analyze_partition(partition, state_snapshots=state_snapshots)

    def _analyze_partition(
        self,
        partition: GraphPartition,
        *,
        state_snapshots: dict[str, dict[str, Any]],
    ) -> PartitionAnalysis:
        """Analyze a partition, optionally reusing one report's effective-state snapshot."""

        from .graph_profile import partition_metrics

        request = replace(self.request, partition=partition)
        self._check_cache_graph()
        key = (
            self.graph_identity,
            request.partition,
            capture_flops._rule_revision,
            self.adapter.name,
            getattr(self.trace_graph.shape_program, "fingerprint", None),
            repr(request.features.as_dict()),
            repr(request.placement.as_dict()),
            request.validation,
            request.batch_symbol,
        )
        cached = self._analysis_cache.get(key)
        if cached is not None:
            self._validate_recut_state_for_plan(
                cached.plan, state_snapshots=state_snapshots, split_point=request.boundary
            )
            self._analysis_cache.move_to_end(key)
            self._cache_counts["hits"] += 1
            return _copy_analysis(cached)
        self._cache_counts["misses"] += 1
        plan = plan_partition(self.trace_graph, request)
        self._validate_recut_state_for_plan(
            plan,
            state_snapshots=state_snapshots,
            split_point=request.boundary,
        )
        graph_ir = (
            replace(self.graph_ir, boundary_schema=tuple(plan.boundary_spec.values()))
            if self.graph_ir is not None
            else SplitGraphIR.from_trace_graph(
                self.trace_graph,
                plan=plan,
                profile_hash=(
                    None if self.model_profile is None else self.model_profile.profile_hash
                ),
            )
        )
        prefix_program = lower_split_program(
            self.trace_graph,
            plan,
            request,
            segment="prefix",
            adapter=self.adapter,
        )
        suffix_program = lower_split_program(
            self.trace_graph,
            plan,
            request,
            segment="suffix",
            adapter=self.adapter,
        )
        capability_report = analyze_split_capabilities(
            self.adapter,
            self.trace_graph,
            plan,
            request,
            prefix_program=prefix_program,
            suffix_program=suffix_program,
            graph_ir=graph_ir,
            model_profile=self.model_profile,
            features=request.features.as_dict(),
        )
        if request.validation == "strict":
            ensure_capability_report_supported(capability_report, request)
        analysis = PartitionAnalysis(
            source_graph=self.trace_graph,
            request=request,
            plan=plan,
            graph_ir=graph_ir,
            prefix_program=prefix_program,
            suffix_program=suffix_program,
            capability_report=capability_report,
            metrics=partition_metrics(self.graph_profile(), plan, training=request.trainable),
        )
        self._analysis_cache[key] = analysis
        if len(self._analysis_cache) > 32:
            self._analysis_cache.popitem(last=False)
        return _copy_analysis(analysis)

    def materialize(self, analysis: PartitionAnalysis) -> SplitRuntime:
        """Build executable segments for an analysis of this captured graph.

        Parameters
        ----------
        analysis
            Partition analysis obtained from this captured graph.

        Returns
        -------
        SplitRuntime
            Executable runtime with migrated segment state.
        """

        return self._materialize(analysis)

    def _materialize(
        self,
        analysis: PartitionAnalysis,
        *,
        state_snapshots: dict[str, dict[str, Any]] | None = None,
    ) -> SplitRuntime:
        """Build fresh client state while optionally reusing this call's state inspection."""
        from .runtime import SplitRuntime

        if analysis.source_graph is not self.trace_graph:
            raise ValueError("Partition analysis belongs to another captured graph.")
        segments = execute_split_runtime(
            self.adapter, self.trace_graph, analysis.plan, analysis.request
        )
        self._inherit_segment_state(segments, recut=True, state_snapshots=state_snapshots)
        runtime = SplitRuntime(
            model=self.model,
            trace=self._trace,
            trace_graph=self.trace_graph,
            request=analysis.request,
            plan=analysis.plan,
            adapter=self.adapter,
            segments=segments,
            capability_report=analysis.capability_report,
            prefix_program=analysis.prefix_program,
            suffix_program=analysis.suffix_program,
            graph_ir=analysis.graph_ir,
            model_profile=self.model_profile,
            prepared_input_kwargs=self.prepared_input_kwargs,
            batch_spec=self.batch_spec,
        )
        runtime._analysis_cache = self._analysis_cache
        runtime._profile_cache = self._profile_cache
        runtime._cache_counts = self._cache_counts
        return runtime


def _copy_analysis(analysis: PartitionAnalysis) -> PartitionAnalysis:
    """Isolate mutable report/schema containers while sharing frozen replay programs."""

    report = analysis.capability_report
    return replace(
        analysis,
        plan=replace(
            analysis.plan,
            boundary_spec=dict(analysis.plan.boundary_spec),
            boundary_bindings=dict(analysis.plan.boundary_bindings),
        ),
        capability_report=replace(
            report,
            features=None if report.features is None else _copy_metadata(report.features),
            node_statuses=None if report.node_statuses is None else dict(report.node_statuses),
            backend_capabilities=(
                None if report.backend_capabilities is None else dict(report.backend_capabilities)
            ),
            shape_diagnostics=_copy_metadata(report.shape_diagnostics),
            feature_evidence=(
                None if report.feature_evidence is None else dict(report.feature_evidence)
            ),
        ),
    )


def _copy_metadata(value: Any) -> Any:
    """Copy JSON-like report containers without visiting immutable scalar metadata."""
    if isinstance(value, dict):
        return {key: _copy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata(item) for item in value)
    return value
