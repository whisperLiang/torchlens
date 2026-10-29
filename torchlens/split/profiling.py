"""Opt-in native execution measurements with isolated effective state."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from math import ceil
from statistics import median
from time import perf_counter
from typing import Any

from ..backends.registry import TORCH_BACKEND_NAME
from .candidates import SplitPointAnalysis
from .errors import SplitErrorContext, SplitUnsupportedError
from .planner import _validate_partition_effects


@dataclass(frozen=True)
class LatencyStats:
    """Synchronized wall times in milliseconds, excluding warmup."""

    median_ms: float
    p95_ms: float
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class SplitProfile:
    """Observed phase costs; unavailable measurements are explicitly unknown."""

    prefix_forward: LatencyStats
    suffix_forward: LatencyStats
    prefix_backward: LatencyStats | None
    suffix_backward: LatencyStats | None
    materialization_ms: float
    boundary_bytes: int
    peak_allocated_bytes: int | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class _ProfileOptions:
    """Configuration carried through one isolated measurement session."""

    warmup: int = 3
    repeats: int = 10
    targets: Any = None
    loss_fn: Callable[[Any, Any], Any] | None = None
    input_kwargs: dict[str, Any] | None = None

    @property
    def training(self) -> bool:
        """Whether the caller supplied an explicit backward objective."""
        return self.targets is not None or self.loss_fn is not None

    def validate(self, runtime: Any) -> None:
        """Reject invalid iteration counts and unsupported execution contracts."""
        if (
            type(self.warmup) is not int
            or self.warmup < 0
            or type(self.repeats) is not int
            or self.repeats < 1
        ):
            raise ValueError("warmup must be a nonnegative integer and repeats a positive integer")
        if runtime.adapter.name != TORCH_BACKEND_NAME:
            raise SplitUnsupportedError(
                "Isolated execution profiling currently requires Torch.",
                context=SplitErrorContext(
                    backend=runtime.adapter.name,
                    split_point=runtime.request.boundary,
                    reason="profiling_unsupported",
                ),
            )
        _validate_partition_effects(runtime.trace_graph, runtime.request)
        if self.training and not runtime.request.trainable:
            raise SplitUnsupportedError("Backward profiling requires SplitFeatures(training=True).")


def _stats(samples: list[float]) -> LatencyStats:
    """Summarize observations without interpolating beyond observed values."""

    return LatencyStats(
        median(samples), sorted(samples)[ceil(0.95 * len(samples)) - 1], tuple(samples)
    )


@dataclass
class _ProfileSession:
    """Measurement-owned tensor clones, device synchronization and phase samples."""

    torch: Any
    device_ids: set[Any] = field(default_factory=set)
    clones: dict[int, Any] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(
        default_factory=lambda: {
            key: []
            for key in ("prefix_forward", "suffix_forward", "prefix_backward", "suffix_backward")
        }
    )
    measured: bool = False

    @classmethod
    def for_runtime(cls, runtime: Any) -> _ProfileSession:
        """Include explicit placements and captured devices, even for parameter-free graphs."""
        import torch

        session = cls(torch=torch)
        for placement in (runtime.placement.prefix, runtime.placement.suffix):
            if placement.is_explicit:
                device = torch.device(runtime.adapter.normalize_device(placement.device))
                if device.type == "cuda":
                    session.device_ids.add(device)
        for node in runtime.trace_graph.nodes:
            captured_device = getattr(node.output_ref, "device", None)
            if captured_device is not None and str(captured_device).startswith("cuda"):
                session.device_ids.add(torch.device(captured_device))
        return session

    def clone_tree(self, value: Any, memo: dict[int, Any]) -> Any:
        """Clone tensor leaves while retaining repeated-object identities."""
        if isinstance(value, self.torch.Tensor):
            if id(value) not in memo:
                memo[id(value)] = value.detach().clone().requires_grad_(value.requires_grad)
            result = memo[id(value)]
            if result.device.type == "cuda":
                self.device_ids.add(result.device)
            return result
        if isinstance(value, dict):
            return {key: self.clone_tree(item, memo) for key, item in value.items()}
        if isinstance(value, tuple):
            items = tuple(self.clone_tree(item, memo) for item in value)
            return type(value)(*items) if hasattr(value, "_fields") else items
        if isinstance(value, list):
            return [self.clone_tree(item, memo) for item in value]
        return value

    def sync(self) -> None:
        """Finish work on all devices used by this measurement."""
        for device in self.device_ids:
            self.torch.cuda.synchronize(device)

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Time a real phase after synchronizing prior work."""
        self.sync()
        start = perf_counter()
        yield
        self.sync()
        if self.measured:
            self.samples[name].append((perf_counter() - start) * 1000)


def _materialize_trial(runtime: Any, session: _ProfileSession) -> tuple[Any, float]:
    """Build one trial through the existing builder and isolate its effective state."""
    with runtime._temporary_segment_state_snapshots():
        start = perf_counter()
        trial = runtime.materialize(
            SplitPointAnalysis(
                runtime.trace_graph,
                runtime.request,
                runtime.plan,
                runtime.graph_ir,
                runtime.prefix_program,
                runtime.suffix_program,
                runtime.capability_report,
            )
        )
        for name in ("prefix", "training_prefix", "suffix"):
            state = getattr(trial.segments, name)._state
            for key, entry in tuple(state._entries.items()):
                value = session.clone_tree(entry.value, session.clones)
                state._entries[key] = replace(entry, value=value, ownership="owned")
            state._inherited.clear()
        session.sync()
        materialization_ms = (perf_counter() - start) * 1000
    return trial, materialization_ms


def _profile_iteration(
    trial: Any, inputs: Any, options: _ProfileOptions, session: _ProfileSession
) -> int:
    """Measure one complete native forward/backward while keeping all roots private."""
    import torch

    from .training import _execute_torch_suffix, _TorchSuffixOptions

    for value in session.clones.values():
        value.grad = None
    memo: dict[int, Any] = {}
    args = session.clone_tree(inputs if isinstance(inputs, tuple) else (inputs,), memo)
    kwargs = session.clone_tree(
        trial.prepared_input_kwargs if options.input_kwargs is None else options.input_kwargs, memo
    )
    expected = session.clone_tree(options.targets, memo)
    with session.timer("prefix_forward"):
        boundary = (
            trial.run_training_prefix(*args, input_kwargs=kwargs)
            if options.training
            else trial.run_prefix(*args, input_kwargs=kwargs)
        )
    boundary_bytes = sum(
        value.numel() * value.element_size() for value in boundary.tensors.values()
    )
    if options.training:
        _, gradients, _ = _execute_torch_suffix(
            trial,
            boundary,
            expected,
            _TorchSuffixOptions(
                loss_fn=options.loss_fn,
                phase_timer=session.timer,
                parameter_inputs=tuple(
                    value for value in session.clones.values() if value.requires_grad
                ),
            ),
        )
        with session.timer("prefix_backward"):
            trial.backward_prefix(boundary, gradients)
    else:
        with torch.no_grad(), session.timer("suffix_forward"):
            trial.run_suffix(boundary)
    return boundary_bytes


def profile_split(runtime: Any, inputs: Any, options: _ProfileOptions) -> SplitProfile:
    """Measure existing native segments on isolated tensors, without optimizer steps.

    Parameters
    ----------
    runtime:
        Prepared Torch runtime. Unknown effects and writes currently refuse
        because independent tensor clones cannot certify their alias semantics.
    inputs:
        Runtime positional inputs, cloned before every iteration.
    options:
        Iteration counts, input keywords and optional explicit training objective.
        A custom objective must not mutate external state; it receives isolated
        outputs/targets, but its Python closure is caller-owned.

    Returns
    -------
    SplitProfile
        Synchronized wall times (including native validation and transport).
        Peak memory remains unknown: allocator high-water marks are process-wide
        and this API deliberately does not reset the caller's allocator counters.
    """
    from ..utils.rng import log_current_rng_states, set_rng_from_saved_states

    options.validate(runtime)
    rng = log_current_rng_states()
    session = _ProfileSession.for_runtime(runtime)
    try:
        trial, materialization_ms = _materialize_trial(runtime, session)
        boundary_bytes = 0
        for iteration in range(options.warmup + options.repeats):
            session.measured = iteration >= options.warmup
            boundary_bytes = _profile_iteration(trial, inputs, options, session)
        return SplitProfile(
            _stats(session.samples["prefix_forward"]),
            _stats(session.samples["suffix_forward"]),
            _stats(session.samples["prefix_backward"]) if options.training else None,
            _stats(session.samples["suffix_backward"]) if options.training else None,
            materialization_ms,
            boundary_bytes,
            None,
            (
                "peak_allocated_bytes:process_allocator_counters_not_reset",
                "materialization_includes_isolation_copies",
                *(
                    ()
                    if options.training
                    else ("backward:explicit_training_objective_not_provided",)
                ),
            ),
        )
    finally:
        set_rng_from_saved_states(rng)
