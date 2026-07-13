"""Public split runtime object."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from hashlib import sha256
from typing import Any

from .adapters.base import SegmentBundle, SplitBackendAdapter
from .boundary import ReplayBoundary
from .cache import load_boundary, save_boundary
from .errors import SplitBoundaryError, SplitErrorContext
from .graph import SplitTraceGraph
from .ir import BoundarySchema, SplitGraphIR, SplitModelProfile, SplitPoint, SplitRequest
from .pipeline import analyze_split_capabilities, execute_split_runtime, lower_split_program
from .planner import SplitPlan
from .planner import plan_split
from .program import ReplayProgram, SplitCapabilityReport, ensure_capability_report_supported
from .validation import nested_allclose


class SplitRuntime:
    """Prepared split runtime for prefix/suffix replay."""

    model: Any
    trace: Any
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
    ) -> None:
        """Create a prepared split runtime."""

        self.model = model
        self.trace = trace
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

    @property
    def split_id(self) -> str:
        """Stable split ID for this runtime."""

        return self.plan.split_id

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

    def at(self, point: SplitPoint) -> "SplitRuntime":
        """Return a runtime at another boundary in the captured graph.

        The complete backend capture and normalized Split IR are immutable for
        a model/input pair.  Reusing them lets a contract test validate every
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
            graph_ir=self.graph_ir,
            model_profile=self.model_profile,
            features=_features_as_dict(request),
        )
        if request.validation == "strict":
            ensure_capability_report_supported(capability_report, request)
        segments = execute_split_runtime(self.adapter, self.trace_graph, plan, request)
        return SplitRuntime(
            model=self.model,
            trace=self.trace,
            trace_graph=self.trace_graph,
            request=request,
            plan=plan,
            adapter=self.adapter,
            segments=segments,
            capability_report=capability_report,
            prefix_program=prefix_program,
            suffix_program=suffix_program,
            graph_ir=self.graph_ir,
            model_profile=self.model_profile,
            prepared_input_kwargs=self.prepared_input_kwargs,
        )

    def run_prefix(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> ReplayBoundary:
        """Run the detached inference prefix."""

        runtime_kwargs = self.prepared_input_kwargs if input_kwargs is None else input_kwargs
        return self._annotate_boundary(
            self.segments.prefix(
                *inputs,
                input_kwargs=runtime_kwargs,
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
            from .errors import SplitUnsupportedError

            raise SplitUnsupportedError(
                f"backend={self.adapter.name!r} does not support split training prefixes.",
                context=SplitErrorContext(
                    backend=self.adapter.name,
                    split_point=self.request.boundary,
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="unsupported training prefix",
                ),
            )
        runtime_kwargs = self.prepared_input_kwargs if input_kwargs is None else input_kwargs
        return self._annotate_boundary(
            self.segments.training_prefix(
                *inputs,
                input_kwargs=runtime_kwargs,
                detach_boundary=False,
            )
        )

    def _annotate_boundary(self, boundary: ReplayBoundary) -> ReplayBoundary:
        """Attach v2 graph/profile/state identity to a backend boundary."""

        metadata = dict(boundary.metadata)
        if self.graph_ir is not None:
            metadata["graph_shape_hash"] = self.graph_ir.graph_hash
            metadata["profile_hash"] = self.graph_ir.profile_hash
        state_fingerprint = _model_state_fingerprint(self.model)
        if state_fingerprint is not None:
            metadata["state_fingerprint"] = state_fingerprint
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
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="backend mismatch",
                ),
            )
        boundary.validate(
            self.boundary_spec,
            split_id=self.split_id,
            graph_hash=None if self.graph_ir is None else self.graph_ir.graph_hash,
            profile_hash=None if self.graph_ir is None else self.graph_ir.profile_hash,
            state_fingerprint=(_model_state_fingerprint(self.model) if validate_state else None),
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
        return self.segments.suffix(boundary)

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
                digest.update(detached.numpy().tobytes())
            else:
                digest.update(repr(detached).encode("utf-8"))
        except Exception:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _features_as_dict(request: SplitRequest) -> dict[str, Any]:
    """Serialize request features for a re-bound capability report."""

    features = request.features
    return {
        "replay": features.replay,
        "dynamic_batch": features.dynamic_batch,
        "training": features.training,
        "boundary_cache": features.boundary_cache,
        "batch_axes": dict(features.batch_axes),
        "cross_device": features.cross_device,
        "live_param_sources": features.live_param_sources,
    }
