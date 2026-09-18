"""Public split runtime object."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from .._state import pause_logging
from .adapters.base import SegmentBundle, SplitBackendAdapter
from .batching import BatchSpec
from .boundary import ReplayBoundary
from .cache import load_boundary, save_boundary
from .candidates import (
    SplitCandidate,
    SplitCandidateReport,
    iter_candidate_sites,
    point_for,
)
from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError
from .graph import SplitTraceGraph
from .ir import BoundarySchema, SplitGraphIR, SplitModelProfile, SplitPoint, SplitRequest
from .pipeline import analyze_split_capabilities, execute_split_runtime, lower_split_program
from .placement import PlacementPlan, move_tree, require_placement_support
from .planner import SplitPlan, plan_split
from .program import ReplayProgram, SplitCapabilityReport, ensure_capability_report_supported
from .state import SegmentState, StateEntry
from .validation import nested_allclose


class SplitRuntime:
    """Prepared split runtime for prefix/suffix replay."""

    model: Any
    _trace: Any | None
    trace_graph: SplitTraceGraph
    request: SplitRequest
    plan: SplitPlan
    adapter: SplitBackendAdapter
    segments: SegmentBundle
    capability_report: SplitCapabilityReport | None
    prefix_program: ReplayProgram | None
    suffix_program: ReplayProgram | None
    graph_ir: SplitGraphIR | None
    model_profile: SplitModelProfile | None
    prepared_input_kwargs: dict[str, Any]
    batch_spec: BatchSpec | None

    def __init__(
        self,
        *,
        model: Any,
        trace: Any,
        trace_graph: SplitTraceGraph,
        request: SplitRequest,
        plan: SplitPlan,
        adapter: SplitBackendAdapter,
        segments: SegmentBundle,
        capability_report: SplitCapabilityReport | None = None,
        prefix_program: ReplayProgram | None = None,
        suffix_program: ReplayProgram | None = None,
        graph_ir: SplitGraphIR | None = None,
        model_profile: SplitModelProfile | None = None,
        prepared_input_kwargs: dict[str, Any] | None = None,
        batch_spec: BatchSpec | None = None,
    ) -> None:
        """Create a prepared split runtime."""

        self.model = model
        self._trace = trace
        self.trace_graph = trace_graph
        self.request = request
        self.plan = plan
        self.adapter = adapter
        self.segments = segments
        self.capability_report = capability_report
        self.prefix_program = prefix_program
        self.suffix_program = suffix_program
        self.graph_ir = graph_ir
        self.model_profile = model_profile
        self.prepared_input_kwargs = dict(prepared_input_kwargs or {})
        self.batch_spec = batch_spec

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
    def split_id(self) -> str:
        """Stable semantic split ID for this runtime.

        The identifier does not include a runtime batch or a batch range:
        every compatible runtime B shares this identity.
        """

        return self.plan.split_id

    @property
    def graph_identity(self) -> str | None:
        """Stable identity of the reused capture/graph."""

        return None if self.graph_ir is None else self.graph_ir.graph_hash

    @property
    def split_ir_identity(self) -> str | None:
        """Stable identity of the reused SplitIR graph."""

        return self.graph_identity

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

    @property
    def boundary_spec(self) -> dict[str, BoundarySchema]:
        """Boundary ABI spec keyed by boundary tensor ID."""

        return self.plan.boundary_spec

    @property
    def boundary_schema(self) -> tuple[BoundarySchema, ...]:
        """Return the backend-neutral v2 boundary ABI schema."""

        if self.graph_ir is None:
            return ()
        return self.graph_ir.boundary_schema

    def explain_capabilities(self) -> dict[str, Any]:
        """Return structured capability and verification diagnostics."""

        if self.capability_report is None:
            return {}
        return self.capability_report.as_dict()

    def at(self, point: SplitPoint) -> SplitRuntime:
        """Return a runtime at another boundary in the captured graph.

        The normalized Split IR and optional diagnostic capture are immutable
        for a model/input pair. Reusing them lets a contract test validate every
        compute-node ``before``/``after`` boundary without recapturing the
        model for each point.  Backend lowering and capability analysis still
        run independently for the requested boundary.

        Parameters
        ----------
        point
            Typed boundary in the already captured graph.

        Returns
        -------
        SplitRuntime
            A runtime sharing the capture while owning a new split plan.
        """

        request = replace(self.request, point=point)
        plan = plan_split(self.trace_graph, request)
        graph_ir = SplitGraphIR.from_trace_graph(
            self.trace_graph,
            plan=plan,
            profile_hash=(None if self.model_profile is None else self.model_profile.profile_hash),
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
        segments = execute_split_runtime(self.adapter, self.trace_graph, plan, request)
        self._inherit_segment_state(segments, recut=True)
        return SplitRuntime(
            model=self.model,
            trace=self._trace,
            trace_graph=self.trace_graph,
            request=request,
            plan=plan,
            adapter=self.adapter,
            segments=segments,
            capability_report=capability_report,
            prefix_program=prefix_program,
            suffix_program=suffix_program,
            graph_ir=graph_ir,
            model_profile=self.model_profile,
            prepared_input_kwargs=self.prepared_input_kwargs,
            batch_spec=self.batch_spec,
        )

    def with_placement(self, plan: PlacementPlan) -> SplitRuntime:
        """Return a runtime at the same split with a different placement.

        The capture, graph, SplitIR, and plan are reused.  Only the segment
        executables are rebuilt so they can bind state on the requested
        devices.
        """

        require_placement_support(self.adapter, plan, split_point=self.request.boundary)
        request = replace(self.request, placement=plan)
        segments = execute_split_runtime(self.adapter, self.trace_graph, self.plan, request)
        self._inherit_segment_state(segments, recut=False)
        return SplitRuntime(
            model=self.model,
            trace=self._trace,
            trace_graph=self.trace_graph,
            request=request,
            plan=self.plan,
            adapter=self.adapter,
            segments=segments,
            capability_report=self.capability_report,
            prefix_program=self.prefix_program,
            suffix_program=self.suffix_program,
            graph_ir=self.graph_ir,
            model_profile=self.model_profile,
            prepared_input_kwargs=self.prepared_input_kwargs,
            batch_spec=self.batch_spec,
        )

    def _inherit_segment_state(self, segments: SegmentBundle, *, recut: bool) -> None:
        """Rebind current effective state, selecting origins by executed node identity."""

        previous = {
            "prefix": self.segments.prefix,
            "training_prefix": self.segments.training_prefix,
            "suffix": self.segments.suffix,
        }
        snapshots: dict[str, dict[str, Any]] = {}
        with pause_logging():
            for name, segment in previous.items():
                getter = getattr(segment, "bound_state_values", None)
                if callable(getter):
                    snapshots[name] = getter()
            if recut:
                self._validate_recut_prefix_state(previous, snapshots, segments)
            seen: set[int] = set()
            for name in previous:
                target = getattr(segments, name)
                state = getattr(target, "_state", None)
                if not isinstance(state, SegmentState) or id(state) in seen:
                    continue
                seen.add(id(state))
                origins: tuple[str, ...] = (name,)
                if recut:
                    origins = (
                        ("training_prefix", "suffix")
                        if name == "training_prefix"
                        else ("prefix", "suffix")
                    )
                state.inherit_entries(
                    _inherited_segment_entries(previous, snapshots, target, origins, recut)
                )
            # Settle migration now, including a typed refusal for incompatible
            # replicas of a tied value that the new cut would have to coalesce.
            for name in previous:
                getter = getattr(getattr(segments, name), "bound_state_values", None)
                if callable(getter):
                    getter()

    def _validate_recut_prefix_state(
        self,
        previous: dict[str, Any],
        snapshots: dict[str, dict[str, Any]],
        segments: SegmentBundle,
    ) -> None:
        """Refuse combining distinct inference/training values in a shared suffix."""

        old_prefix_state = getattr(previous["prefix"], "_state", None)
        suffix_node_ids: frozenset[str] = getattr(segments.suffix, "node_ids", frozenset())
        training_values = snapshots.get("training_prefix", {})
        if isinstance(old_prefix_state, SegmentState):
            for key, value in snapshots.get("prefix", {}).items():
                node_id = key.removesuffix(":source").rsplit(":literal:", 1)[0]
                if node_id not in suffix_node_ids or key not in training_values:
                    continue
                if not old_prefix_state._same_value(value, training_values[key]):
                    raise SplitUnsupportedError(
                        "Cannot recut divergent inference and training prefix state "
                        "into one shared suffix; synchronize the prefix values or "
                        "keep those operations in the prefix.",
                        context=SplitErrorContext(
                            backend=self.adapter.name,
                            split_point=self.request.boundary,
                            reason="divergent prefix state replicas",
                        ),
                    )

    def _state_fingerprint(self, prefix_kind: str) -> str | None:
        """Hash the state that actually executes, including device-owned replicas."""

        prefix = (
            self.segments.training_prefix if prefix_kind == "training" else self.segments.prefix
        )
        prefix_getter = getattr(prefix, "bound_state_values", None)
        suffix_getter = getattr(self.segments.suffix, "bound_state_values", None)
        with pause_logging():
            if callable(prefix_getter) and callable(suffix_getter):
                return _state_values_fingerprint({**prefix_getter(), **suffix_getter()})
            return _model_state_fingerprint(self.model)

    def split_points(self, *, diagnose: bool = True) -> SplitCandidateReport:
        """Enumerate every semantically valid before/after compute boundary.

        Unsupported positions are never silently skipped: they remain in the
        report with a deterministic reason.
        """

        candidates: list[SplitCandidate] = []
        for site in iter_candidate_sites(self.trace_graph):
            for kind in site.kinds:
                point = point_for(kind, site.node_id)
                if not diagnose:
                    candidates.append(
                        SplitCandidate(
                            point=point,
                            kind=kind,  # type: ignore[arg-type]
                            node_id=site.node_id,
                            label=site.label,
                            op_type=site.op_type,
                            module_path=site.module_path,
                        )
                    )
                    continue
                candidates.append(self._diagnose_point(point, site, kind))
        return SplitCandidateReport(tuple(candidates))

    def _diagnose_point(self, point: SplitPoint, site: Any, kind: str) -> SplitCandidate:
        """Lower one candidate without raising, capturing a structured reason."""

        from .errors import SplitRequestError, SplitUnsupportedError as _Unsupported

        try:
            runtime = self.at(point)
        except (_Unsupported, SplitRequestError) as exc:
            return SplitCandidate(
                point=point,
                kind=kind,  # type: ignore[arg-type]
                node_id=site.node_id,
                label=site.label,
                op_type=site.op_type,
                module_path=site.module_path,
                boundary_value_ids=(),
                boundary_schema=(),
                replay_supported=False,
                training_supported=False,
                unsupported_reasons=(str(exc),),
            )
        reasons: tuple[str, ...] = ()
        if runtime.capability_report is not None:
            reasons = runtime.capability_report.unsupported_reasons
        unresolved: tuple[str, ...] = ()
        if runtime.trace_graph.shape_program is not None:
            unresolved = tuple(runtime.trace_graph.shape_program.unresolved)
        training_ok = (
            runtime.capability_report is None or runtime.capability_report.training.supported
        )
        replay_ok = not reasons
        return SplitCandidate(
            point=point,
            kind=kind,  # type: ignore[arg-type]
            node_id=site.node_id,
            label=site.label,
            op_type=site.op_type,
            module_path=site.module_path,
            boundary_value_ids=runtime.plan.boundary_node_ids,
            boundary_schema=runtime.boundary_schema,
            replay_supported=replay_ok,
            training_supported=replay_ok and training_ok,
            unsupported_reasons=reasons,
            shape_unresolved=unresolved,
        )

    def run_prefix(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> ReplayBoundary:
        """Run the detached inference prefix."""

        placed_inputs, placed_kwargs = self._place_inputs(inputs, input_kwargs)
        return self._annotate_boundary(
            self.segments.prefix(
                *placed_inputs,
                input_kwargs=placed_kwargs,
                detach_boundary=True,
            )
        )

    def run_training_prefix(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> ReplayBoundary:
        """Run the graph-connected training prefix."""

        if self.segments.training_prefix is None:
            raise SplitUnsupportedError(
                f"backend={self.adapter.name!r} does not support split training prefixes.",
                context=SplitErrorContext(
                    backend=self.adapter.name,
                    split_point=self.request.boundary,
                    reason="unsupported training prefix",
                ),
            )
        placed_inputs, placed_kwargs = self._place_inputs(inputs, input_kwargs)
        return self._annotate_boundary(
            self.segments.training_prefix(
                *placed_inputs,
                input_kwargs=placed_kwargs,
                detach_boundary=False,
            )
        )

    def _place_inputs(
        self,
        inputs: tuple[Any, ...],
        input_kwargs: dict[str, Any] | None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Move runtime inputs onto the prefix placement."""

        runtime_kwargs = self.prepared_input_kwargs if input_kwargs is None else input_kwargs
        placement = self.placement.prefix
        if not placement.is_explicit:
            return inputs, dict(runtime_kwargs or {})
        placed_inputs = tuple(move_tree(self.adapter, value, placement) for value in inputs)
        placed_kwargs = {
            key: move_tree(self.adapter, value, placement)
            for key, value in dict(runtime_kwargs or {}).items()
        }
        return placed_inputs, placed_kwargs

    def _annotate_boundary(self, boundary: ReplayBoundary) -> ReplayBoundary:
        """Attach v2 graph/profile/state identity to a backend boundary."""

        metadata = dict(boundary.metadata)
        if self.graph_ir is not None:
            metadata["graph_shape_hash"] = self.graph_ir.graph_hash
            metadata["profile_hash"] = self.graph_ir.profile_hash
        prefix_kind = "training" if metadata.get("supports_prefix_backward") else "inference"
        metadata["state_prefix_kind"] = prefix_kind
        state_fingerprint = self._state_fingerprint(prefix_kind)
        if state_fingerprint is not None:
            metadata["state_fingerprint"] = state_fingerprint
        metadata["batch_symbol"] = self.request.batch_symbol
        if self.batch_validation:
            metadata["batch_validation"] = self.batch_validation
            batch = metadata.get("runtime_batch_size")
            metadata["runtime_batch_validation"] = (
                "captured"
                if batch == self.traced_batch_size
                else "sampled"
                if batch == self.batch_validation["probe_batch_size"]
                else "extrapolated"
            )
        return ReplayBoundary(
            backend=boundary.backend,
            tensors=boundary.tensors,
            spec=boundary.spec,
            metadata=metadata,
        )

    def validate_boundary(self, boundary: ReplayBoundary, *, validate_state: bool = True) -> None:
        """Validate ``boundary`` for this runtime."""

        if boundary.backend != self.adapter.name:
            raise SplitBoundaryError(
                "Replay boundary backend differs from runtime backend.",
                context=SplitErrorContext(
                    backend=boundary.backend,
                    split_point=self.request.boundary,
                    reason="backend mismatch",
                ),
            )
        program = self.trace_graph.shape_program
        if program is not None and program.batch_probe is not None:
            batch = boundary.metadata.get("runtime_batch_size")
            if batch is None:
                raise SplitBoundaryError("Boundary is missing runtime batch metadata.")
            program.batch_probe.require_batch(
                int(batch), backend=self.adapter.name, split_point=self.request.boundary
            )
        boundary.validate(
            self.boundary_spec,
            split_id=self.split_id,
            graph_hash=None if self.graph_ir is None else self.graph_ir.graph_hash,
            profile_hash=None if self.graph_ir is None else self.graph_ir.profile_hash,
            state_fingerprint=(
                self._state_fingerprint(
                    str(boundary.metadata.get("state_prefix_kind", "inference"))
                )
                if validate_state
                else None
            ),
            shape_program_hash=(
                None
                if self.trace_graph.shape_program is None
                else self.trace_graph.shape_program.fingerprint
            ),
            shape_program=self.trace_graph.shape_program,
            adapter=self.adapter,
        )

    def run_suffix(self, boundary: ReplayBoundary) -> Any:
        """Validate and run the suffix from a replay boundary."""

        self.validate_boundary(boundary)
        return self.segments.suffix(self._transport_boundary(boundary, self.placement.suffix))

    def _transport_boundary(
        self,
        boundary: ReplayBoundary,
        placement: Any,
    ) -> ReplayBoundary:
        """Move boundary tensors onto a destination placement."""

        if not placement.is_explicit:
            return boundary
        return boundary.to(placement.device, adapter=self.adapter)

    def prefix_parameters(self) -> list[Any]:
        """Return owned trainable prefix replicas, if the adapter exposes them."""

        getter = getattr(self.segments.training_prefix, "trainable_parameters", None)
        if not callable(getter):
            return []
        return list(getter())

    def suffix_parameters(self) -> list[Any]:
        """Return owned trainable suffix replicas, if the adapter exposes them."""

        getter = getattr(self.segments.suffix, "trainable_parameters", None)
        if not callable(getter):
            return []
        return list(getter())

    def replay(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run prefix then suffix."""

        return self.run_suffix(self.run_prefix(*inputs, input_kwargs=input_kwargs))

    def validate_equivalence(
        self,
        model: Any,
        inputs: tuple[Any, ...],
        atol: float = 1e-5,
        rtol: float = 1e-4,
        input_kwargs: dict[str, Any] | None = None,
    ) -> bool:
        """Return whether full model and split replay outputs match."""

        runtime_kwargs = self.prepared_input_kwargs if input_kwargs is None else input_kwargs
        full_output = model(*inputs, **(runtime_kwargs or {}))
        replay_output = self.replay(*inputs, input_kwargs=runtime_kwargs)
        return nested_allclose(self.adapter, full_output, replay_output, atol=atol, rtol=rtol)

    def train_suffix(
        self,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Any = None,
        optimizer: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Train the suffix and return boundary gradients."""

        from .training import train_suffix

        return train_suffix(self, boundary, targets, loss_fn=loss_fn, optimizer=optimizer)

    def train_suffix_result(
        self,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Any = None,
        optimizer: Any = None,
    ) -> Any:
        """Train the suffix and return a structured training result."""

        from .training import train_suffix_result

        return train_suffix_result(self, boundary, targets, loss_fn=loss_fn, optimizer=optimizer)

    def backward_prefix(
        self,
        boundary: ReplayBoundary,
        boundary_grads: dict[str, Any],
        optimizer: Any = None,
    ) -> Any:
        """Apply or return suffix boundary gradients for the graph-connected prefix."""

        from .training import backward_prefix

        return backward_prefix(self, boundary, boundary_grads, optimizer=optimizer)

    def save_boundary(self, boundary: ReplayBoundary, path: str | Path) -> None:
        """Save a replay boundary cache."""

        save_boundary(boundary, path, adapter=self.adapter)

    def load_boundary(self, path: str | Path) -> ReplayBoundary:
        """Load a replay boundary cache."""

        return load_boundary(path, adapter=self.adapter)


__all__ = ["SplitRuntime"]


def _inherited_segment_entries(
    previous: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
    target: Any,
    origins: tuple[str, ...],
    recut: bool,
) -> list[StateEntry]:
    """Select existing entries consumed by the target's executed node identities."""

    inherited: list[StateEntry] = []
    for origin in origins:
        old_state = getattr(previous[origin], "_state", None)
        if not isinstance(old_state, SegmentState):
            continue
        used_values = {
            id(value)
            for key, value in snapshots.get(origin, {}).items()
            if not recut or key.removesuffix(":source").rsplit(":literal:", 1)[0] in target.node_ids
        }
        inherited.extend(entry for entry in old_state.entries() if id(entry.value) in used_values)
    return inherited


def _model_state_fingerprint(model: Any) -> str | None:
    """Return a value-sensitive state fingerprint when a model exposes state."""

    state_dict = getattr(model, "state_dict", None)
    if callable(state_dict):
        try:
            values = state_dict()
        except Exception:
            return None
    else:
        values = getattr(model, "variables", None)
        if values is None:
            values = getattr(model, "parameters", None)
        if callable(values):
            try:
                values = tuple(values())
            except Exception:
                return None
        if values is None:
            return None
        if not hasattr(values, "items"):
            values = {str(index): value for index, value in enumerate(values)}
    return _state_values_fingerprint(values)


def _state_values_fingerprint(values: dict[str, Any]) -> str:
    """Return a stable value digest, independent of source identity and placement."""

    digest = sha256()
    for name in sorted(values):
        value = values[name]
        digest.update(str(name).encode("utf-8"))
        digest.update(repr(getattr(value, "shape", None)).encode("utf-8"))
        digest.update(str(getattr(value, "dtype", None)).encode("utf-8"))
        try:
            detached = value.detach() if hasattr(value, "detach") else value
            if hasattr(detached, "cpu"):
                detached = detached.cpu()
            if hasattr(detached, "numpy"):
                try:
                    payload = detached.numpy().tobytes()
                except TypeError:
                    # NumPy has no bfloat16 dtype; hash the raw Torch bytes so
                    # updates outside the abbreviated tensor repr stay visible.
                    import torch

                    payload = detached.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                digest.update(payload)
            else:
                digest.update(repr(detached).encode("utf-8"))
        except Exception:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()
