"""Public split runtime object."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._state import pause_logging
from .adapters.base import SegmentBundle, SplitBackendAdapter
from .batching import BatchSpec
from .boundary import ReplayBoundary
from .cache import load_boundary, save_boundary
from .context import SplitGraph
from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError
from .graph import SplitTraceGraph
from .ir import (
    BoundarySchema,
    GraphPartition,
    SplitGraphIR,
    SplitModelProfile,
    SplitRequest,
)
from .pipeline import analyze_split_capabilities, execute_split_runtime
from .placement import PlacementPlan, move_tree, require_placement_support
from .planner import SplitPlan, plan_partition
from .program import ReplayProgram, SplitCapabilityReport
from .state import SegmentState, StateEntry, _StateReplicaPool
from .validation import nested_allclose

if TYPE_CHECKING:
    from .boundary_layout import BoundaryLayout
    from .ownership import ParameterOwnership, StateMigrationPlan
    from .profiling import SplitProfile


class SplitRuntime(SplitGraph):
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

        super().__init__(
            model=model,
            trace=trace,
            trace_graph=trace_graph,
            request=request,
            adapter=adapter,
            graph_ir=graph_ir,
            model_profile=model_profile,
            prepared_input_kwargs=prepared_input_kwargs,
            batch_spec=batch_spec,
        )
        self.plan = plan
        self.segments = segments
        self.capability_report = capability_report
        self.prefix_program = prefix_program
        self.suffix_program = suffix_program
        initializer = getattr(adapter, "initialize_runtime_state", None)
        if callable(initializer):
            with pause_logging():
                initializer(self)

    @property
    def split_id(self) -> str:
        """Stable semantic split ID for this runtime.

        The identifier does not include a runtime batch or a batch range:
        every compatible runtime B shares this identity.
        """

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

    def parameter_ownership(self) -> tuple[ParameterOwnership, ...]:
        """Describe stable parameter/buffer identities and current segment ownership."""

        from .ownership import parameter_ownership

        return parameter_ownership(self)

    def boundary_layout(
        self, *, batch_size: int | None = None, training: bool = False
    ) -> BoundaryLayout:
        """Describe unique boundary payloads from static shape and ABI metadata."""

        from .boundary_layout import build_boundary_layout

        return build_boundary_layout(
            self.graph_profile(batch_size=batch_size), self.plan, training=training
        )

    def profile_split(
        self,
        inputs: Any,
        *,
        warmup: int = 3,
        repeats: int = 10,
        targets: Any = None,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        input_kwargs: dict[str, Any] | None = None,
    ) -> SplitProfile:
        """Measure synchronized phases on isolated state; never step an optimizer."""

        from .profiling import _ProfileOptions, profile_split

        return profile_split(
            self,
            inputs,
            _ProfileOptions(
                warmup=warmup,
                repeats=repeats,
                targets=targets,
                loss_fn=loss_fn,
                input_kwargs=input_kwargs,
            ),
        )

    def parameter_bindings(self) -> Mapping[str, Mapping[str, Any]]:
        """Bind stable parameter IDs to effective tensors for caller-owned optimizers."""

        from .ownership import parameter_bindings

        return parameter_bindings(self)

    def plan_state_migration(self, partition: GraphPartition) -> StateMigrationPlan:
        """Describe a recut and external optimizer reassociation without moving state."""

        from .ownership import plan_state_migration

        return plan_state_migration(self, partition)

    def repartition(self, partition: GraphPartition) -> SplitRuntime:
        """Materialize another DAG partition using the current effective training state."""
        if not isinstance(partition, GraphPartition):
            raise TypeError("partition must be a GraphPartition")
        with self._temporary_segment_state_snapshots() as snapshots:
            analysis = self._analyze_partition(partition, state_snapshots=snapshots)
            return self._materialize(analysis, state_snapshots=snapshots)

    def with_placement(self, plan: PlacementPlan) -> SplitRuntime:
        """Return a runtime at the same split with a different placement.

        The capture, graph, SplitIR, and plan are reused.  Only the segment
        executables are rebuilt so they can bind state on the requested
        devices.
        """

        require_placement_support(self.adapter, plan, split_point=self.request.boundary)
        request = replace(self.request, placement=plan)
        plan_partition(self.trace_graph, request)
        capability_report = self.capability_report
        if self.prefix_program is not None and self.suffix_program is not None:
            capability_report = analyze_split_capabilities(
                self.adapter,
                self.trace_graph,
                self.plan,
                request,
                prefix_program=self.prefix_program,
                suffix_program=self.suffix_program,
                graph_ir=self.graph_ir,
                model_profile=self.model_profile,
                features=request.features.as_dict(),
            )
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
            capability_report=capability_report,
            prefix_program=self.prefix_program,
            suffix_program=self.suffix_program,
            graph_ir=self.graph_ir,
            model_profile=self.model_profile,
            prepared_input_kwargs=self.prepared_input_kwargs,
            batch_spec=self.batch_spec,
        )

    def _inherit_segment_state(
        self,
        segments: SegmentBundle,
        *,
        recut: bool,
        state_snapshots: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Rebind current effective state, selecting origins by executed node identity."""

        previous = {
            "prefix": self.segments.prefix,
            "training_prefix": self.segments.training_prefix,
            "suffix": self.segments.suffix,
        }
        with pause_logging():
            snapshots = (
                self._segment_state_snapshots() if state_snapshots is None else state_snapshots
            )
            if recut:
                self._validate_recut_prefix_state(
                    previous, snapshots, getattr(segments.suffix, "node_ids", frozenset())
                )
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
                    _inherited_segment_entries(previous, snapshots, target.node_ids, origins, recut)
                )
            # Settle migration now, including a typed refusal for incompatible
            # replicas of a tied value that the new cut would have to coalesce.
            settled: set[tuple[int, frozenset[str], bool]] = set()
            for name in previous:
                target = getattr(segments, name)
                getter = getattr(target, "bound_state_values", None)
                if callable(getter):
                    key: tuple[int, frozenset[str], bool] = (
                        id(getattr(target, "_state", target)),
                        getattr(target, "node_ids", frozenset()),
                        getattr(target, "use_live_param_sources", False),
                    )
                    if key not in settled:
                        getter()
                        settled.add(key)
            inheritor = getattr(self.adapter, "inherit_runtime_state", None)
            if callable(inheritor):
                inheritor(self, segments)

    def _segment_state_snapshots(self) -> dict[str, dict[str, Any]]:
        """Bind current effective state once without building new segments."""

        snapshots: dict[str, dict[str, Any]] = {}
        shared: dict[tuple[int, frozenset[str], bool], dict[str, Any]] = {}
        with pause_logging():
            for name in ("prefix", "training_prefix", "suffix"):
                segment = getattr(self.segments, name)
                getter = getattr(segment, "bound_state_values", None)
                if callable(getter):
                    key: tuple[int, frozenset[str], bool] = (
                        id(getattr(segment, "_state", segment)),
                        getattr(segment, "node_ids", frozenset()),
                        getattr(segment, "use_live_param_sources", False),
                    )
                    if key not in shared:
                        shared[key] = getter()
                    snapshots[name] = shared[key]
        return snapshots

    @contextmanager
    def _temporary_segment_state_snapshots(self) -> Iterator[dict[str, dict[str, Any]]]:
        """Inspect prospective bindings without freezing this runtime's lazy state.

        Yields
        ------
        dict[str, dict[str, Any]]
            Effective state by segment and replay occurrence. Temporary lazy
            bindings and replica-cache additions are restored on exit.
        """

        saved_entries: dict[int, tuple[SegmentState, dict[int, StateEntry]]] = {}
        saved_pools: dict[
            int, tuple[_StateReplicaPool, dict[tuple[str, int, str], tuple[Any, Any]]]
        ] = {}
        for name in ("prefix", "training_prefix", "suffix"):
            state = getattr(getattr(self.segments, name), "_state", None)
            if isinstance(state, SegmentState) and id(state) not in saved_entries:
                saved_entries[id(state)] = (state, dict(state._entries))
                pool = state._replica_pool
                if pool is not None and id(pool) not in saved_pools:
                    saved_pools[id(pool)] = (pool, dict(pool._replicas))
        try:
            yield self._segment_state_snapshots()
        finally:
            for state, entries in saved_entries.values():
                state._entries.clear()
                state._entries.update(entries)
            for pool, replicas in saved_pools.values():
                pool._replicas.clear()
                pool._replicas.update(replicas)

    def _validate_recut_state_for_plan(
        self,
        plan: SplitPlan,
        *,
        state_snapshots: dict[str, dict[str, Any]],
        split_point: str,
    ) -> None:
        """Detect state mergers that would fail during segment materialization."""

        previous = {
            "prefix": self.segments.prefix,
            "training_prefix": self.segments.training_prefix,
            "suffix": self.segments.suffix,
        }
        state = getattr(previous["prefix"], "_state", None)
        if not isinstance(state, SegmentState):
            return
        snapshots = state_snapshots
        with pause_logging():
            self._validate_recut_prefix_state(previous, snapshots, plan.suffix_node_ids)
            for name, node_ids in (
                ("prefix", plan.prefix_node_ids),
                ("training_prefix", plan.prefix_node_ids),
                ("suffix", plan.suffix_node_ids),
            ):
                origins = (
                    ("training_prefix", "suffix")
                    if name == "training_prefix"
                    else ("prefix", "suffix")
                )
                entries = _inherited_segment_entries(previous, snapshots, node_ids, origins, True)
                by_source: dict[int, StateEntry] = {}
                for entry in entries:
                    prior = by_source.setdefault(entry.source_id, entry)
                    if not state._same_value(prior.value, entry.value):
                        raise SplitUnsupportedError(
                            "Cannot recut divergent replicas of a shared state value; synchronize "
                            "the segment replicas before changing the split point.",
                            context=SplitErrorContext(
                                backend=self.adapter.name,
                                split_point=split_point,
                                reason="divergent segment state replicas",
                            ),
                        )

    def _validate_recut_prefix_state(
        self,
        previous: dict[str, Any],
        snapshots: dict[str, dict[str, Any]],
        suffix_node_ids: frozenset[str],
    ) -> None:
        """Refuse combining distinct inference/training values in a shared suffix."""

        old_prefix_state = getattr(previous["prefix"], "_state", None)
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

    def run_prefix(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> ReplayBoundary:
        """Run the detached inference prefix."""

        return self._run_prefix(*inputs, input_kwargs=input_kwargs)

    def _run_prefix(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None,
    ) -> ReplayBoundary:
        """Build a prefix boundary with structural replay metadata."""

        placed_inputs, placed_kwargs = self._place_inputs(inputs, input_kwargs)
        return self._annotate_boundary(
            self.segments.prefix(
                *placed_inputs,
                input_kwargs=placed_kwargs,
                detach_boundary=True,
            ),
        )

    def run_training_prefix(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> ReplayBoundary:
        """Run the graph-connected training prefix."""

        if self.plan.training_boundary_writes:
            raise SplitUnsupportedError(
                "This inference partition writes floating boundary leaves; choose a training-safe partition.",
                context=SplitErrorContext(
                    backend=self.adapter.name,
                    split_point=self.request.boundary,
                    reason="partition_training_boundary_write",
                ),
            )
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
        memo: dict[int, Any] = {}
        placed_inputs = tuple(
            move_tree(self.adapter, value, placement, memo=memo) for value in inputs
        )
        placed_kwargs = {
            key: move_tree(self.adapter, value, placement, memo=memo)
            for key, value in dict(runtime_kwargs or {}).items()
        }
        return placed_inputs, placed_kwargs

    def _annotate_boundary(self, boundary: ReplayBoundary) -> ReplayBoundary:
        """Attach graph/profile identity to a boundary."""

        metadata = dict(boundary.metadata)
        if self.graph_ir is not None:
            metadata["graph_shape_hash"] = self.graph_ir.graph_hash
            metadata["profile_hash"] = self.graph_ir.profile_hash
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

    def validate_boundary(self, boundary: ReplayBoundary) -> None:
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
            shape_program_hash=(
                None
                if self.trace_graph.shape_program is None
                else self.trace_graph.shape_program.fingerprint
            ),
            shape_program=self.trace_graph.shape_program,
            adapter=self.adapter,
        )

    def run_suffix(self, boundary: ReplayBoundary) -> Any:
        """Validate structural compatibility and run the suffix."""

        self.validate_boundary(boundary)
        return self._run_suffix_unchecked(boundary)

    def _run_suffix_unchecked(self, boundary: ReplayBoundary, *, transported: bool = False) -> Any:
        """Execute a boundary that was already validated by this runtime.

        This is the internal suffix fast path.  The public ``run_suffix`` keeps
        all boundary checks, while replay and split-training can avoid
        revalidating a boundary that they created and checked in the same call.
        """

        # Rebind before entering the suffix so a transported replay does not
        # retain the source boundary through this helper's argument frame.
        if not transported:
            boundary = self._transport_boundary(boundary, self.placement.suffix)
        return self.segments.suffix(boundary)

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

    def state_dict(self) -> dict[str, Any]:
        """Export current named split state as independent array snapshots.

        Returns
        -------
        dict[str, Any]
            Full native flattened module state, including frozen parameters,
            running buffers, and unused parameters. Currently supported for MLX
            native modules. Training updates are read from the runtime's owned
            state; the source model remains unchanged.

        Raises
        ------
        SplitUnsupportedError
            If the backend lacks named state exchange or tied replicas diverge.
        """

        getter = self._state_exchange_method("state_dict")
        with pause_logging():
            return dict(getter(self))

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load aggregated state into all segment bindings without writing the source model.

        Parameters
        ----------
        state_dict
            Full flattened state mapping compatible with :meth:`state_dict`.
            MLX requires exact keys, shapes, dtypes, and consistent tied values.

        Raises
        ------
        SplitUnsupportedError
            If state exchange is unsupported or validation fails. Validation and
            device copies finish before any state commits. Successful loading
            invalidates previously connected training boundaries; generate a
            fresh boundary before the next training step. Optimizer state is
            managed by the caller and is not loaded by this method.
        """

        loader = self._state_exchange_method("load_state_dict")
        with pause_logging():
            loader(self, state_dict)

    def _state_exchange_method(self, name: str) -> Callable[..., Any]:
        """Resolve an optional adapter state method or refuse with backend context."""

        method = getattr(self.adapter, name, None)
        if not callable(method):
            raise SplitUnsupportedError(
                f"backend={self.adapter.name!r} does not support split {name}().",
                context=SplitErrorContext(
                    backend=self.adapter.name,
                    split_point=self.request.boundary,
                    reason="named_state_exchange_unsupported",
                ),
            )
        return method

    def replay(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run prefix then suffix."""

        boundary = self._run_prefix(*inputs, input_kwargs=input_kwargs)
        self.validate_boundary(boundary)
        # This boundary belongs to replay(), so replacing the reference after
        # transport releases the source payload before the suffix starts. The
        # public run_suffix(boundary) path continues to borrow caller state.
        boundary = self._transport_boundary(boundary, self.placement.suffix)
        return self._run_suffix_unchecked(boundary, transported=True)

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
        validator = getattr(self.adapter, "validate_equivalence", None)
        if callable(validator):
            return bool(
                validator(self, model, inputs, atol=atol, rtol=rtol, input_kwargs=runtime_kwargs)
            )
        full_output = model(*inputs, **(runtime_kwargs or {}))
        replay_output = self.replay(*inputs, input_kwargs=runtime_kwargs)
        return nested_allclose(self.adapter, full_output, replay_output, atol=atol, rtol=rtol)

    def train_suffix(
        self,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Any = None,
        optimizer: Any = None,
        *,
        microbatch_size: int | None = None,
        microbatch_reduction: str = "mean",
        target_slicer: Callable[[Any, int, int, int], Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Train the suffix and return logical-batch boundary gradients.

        Parameters
        ----------
        boundary, targets
            Logical boundary and task targets, with targets on the suffix device.
        loss_fn
            Optional scalar loss. With microbatching, it must reduce independent
            samples using ``microbatch_reduction``.
        optimizer
            Caller-owned optimizer; zeroed and stepped once per logical batch.
        microbatch_size
            Optional positive Torch suffix batch size. The final chunk may be smaller.
        microbatch_reduction
            ``"mean"`` weights chunk losses by chunk size / logical size;
            ``"sum"`` adds custom sum-reduced losses without scaling.
        target_slicer
            Optional ``(targets, start, end, logical_batch) -> chunk`` callback
            for target structures whose sample axis cannot be inferred.

        Returns
        -------
        tuple
            Loss and boundary gradients suitable for one ``backward_prefix`` call.
            The microbatch loss is detached; suffix graphs are freed per chunk.
        """

        from .training import train_suffix

        return train_suffix(
            self,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
            microbatch_size=microbatch_size,
            microbatch_reduction=microbatch_reduction,
            target_slicer=target_slicer,
        )

    def train_suffix_result(
        self,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Any = None,
        optimizer: Any = None,
        *,
        microbatch_size: int | None = None,
        microbatch_reduction: str = "mean",
        target_slicer: Callable[[Any, int, int, int], Any] | None = None,
    ) -> Any:
        """Train the suffix and return a structured training result."""

        from .training import train_suffix_result

        return train_suffix_result(
            self,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
            microbatch_size=microbatch_size,
            microbatch_reduction=microbatch_reduction,
            target_slicer=target_slicer,
        )

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
    target_node_ids: frozenset[str],
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
            if not recut or key.removesuffix(":source").rsplit(":literal:", 1)[0] in target_node_ids
        }
        inherited.extend(entry for entry in old_state.entries() if id(entry.value) in used_values)
    return inherited
