"""MLX suffix microbatches with one coordinated logical optimizer update."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ._mlx_optimizer import stage_suffix_step, validate_mlx_optimizer
from .adapters._mlx_devices import mlx_execution_context
from .boundary import ReplayBoundary
from .errors import SplitBoundaryError


@dataclass(frozen=True)
class MlxMicrobatchOptions:
    """Loss, chunk size, reduction, and native optimizer for one logical batch."""

    size: int
    reduction: str = "mean"
    loss_fn: Callable[[Any, Any], Any] | None = None
    optimizer: Any = None
    target_slicer: Callable[[Any, int, int, int], Any] | None = None


def _validate_options(options: MlxMicrobatchOptions) -> None:
    """Check chunk/reduction semantics before stateful suffix execution begins."""

    if type(options.size) is not int or options.size < 1:
        raise ValueError("microbatch_size must be a positive integer")
    if options.reduction not in {"mean", "sum"}:
        raise ValueError("microbatch_reduction must be 'mean' or 'sum'")
    if options.reduction == "sum" and options.loss_fn is None:
        raise ValueError("microbatch_reduction='sum' requires a custom sum-reduced loss_fn")
    validate_mlx_optimizer(options.optimizer)


def _slice_targets(value: Any, start: int, end: int, batch: int) -> Any:
    """Slice native target arrays and sample lists while preserving nested containers."""

    import mlx.core as mx

    if isinstance(value, mx.array):
        return value[start:end] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, Mapping):
        return {key: _slice_targets(item, start, end, batch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_targets(item, start, end, batch) for item in value)
    if isinstance(value, list):
        if len(value) == batch:
            return value[start:end]
        return [_slice_targets(item, start, end, batch) for item in value]
    return value


def _chunk_boundary(
    boundary: ReplayBoundary, axes: dict[str, int | None], start: int, end: int
) -> ReplayBoundary:
    """Create independent array handles for one symbolic boundary chunk."""

    import mlx.core as mx

    values = {}
    for key, value in boundary.tensors.items():
        axis = axes[key]
        index = [slice(None)] * value.ndim
        if axis is not None:
            index[axis] = slice(start, end)
            value = value[tuple(index)]
        values[key] = mx.stop_gradient(value).astype(value.dtype)
    metadata = {
        key: value for key, value in boundary.metadata.items() if not key.startswith("mlx_")
    }
    metadata.update(runtime_batch_size=end - start, supports_prefix_backward=False)
    return ReplayBoundary(boundary.backend, values, boundary.spec, metadata)


@dataclass
class _GradientAccumulator:
    """Evaluated logical gradients with no retained per-chunk activation archive."""

    boundary: ReplayBoundary
    axes: dict[str, int | None]
    boundary_grads: dict[str, Any]
    parameter_grads: dict[str, Any]
    loss: Any = None

    def add(self, result: Any, start: int, end: int) -> None:
        """Add shared-value gradients and splice each sample's boundary cotangent."""

        import mlx.core as mx

        self.loss = result.loss if self.loss is None else self.loss + result.loss
        for name, value in result.parameter_grads.items():
            self.parameter_grads[name] = (
                self.parameter_grads[name] + value if name in self.parameter_grads else value
            )
        for key, value in result.boundary_grads.items():
            axis = self.axes[key]
            if axis is None:
                self.boundary_grads[key] = (
                    self.boundary_grads[key] + value if key in self.boundary_grads else value
                )
            else:
                full = self.boundary_grads.setdefault(
                    key, mx.zeros_like(self.boundary.tensors[key])
                )
                index = [slice(None)] * full.ndim
                index[axis] = slice(start, end)
                full[tuple(index)] = value
        mx.eval(self.loss, self.parameter_grads, self.boundary_grads)


def _run_chunks(
    runtime: Any, accumulator: _GradientAccumulator, targets: Any, options: MlxMicrobatchOptions
) -> None:
    """Differentiate weighted suffix chunks, keeping parameters fixed for the logical batch."""

    from ._mlx_training import MlxTrainingEngine, _default_loss

    batch = accumulator.boundary.metadata["runtime_batch_size"]
    for start in range(0, batch, options.size):
        end = min(start + options.size, batch)
        chunk_targets = (
            options.target_slicer(targets, start, end, batch)
            if options.target_slicer is not None
            else _slice_targets(targets, start, end, batch)
        )
        weight = (end - start) / batch if options.reduction == "mean" else 1.0

        def loss(output: Any, target: Any, scale: float = weight) -> Any:
            """Weight a scalar chunk loss using the declared logical-batch reduction."""

            return (options.loss_fn or _default_loss)(output, target) * scale

        chunk = _chunk_boundary(accumulator.boundary, accumulator.axes, start, end)
        result = MlxTrainingEngine().train_suffix(runtime, chunk, chunk_targets, loss_fn=loss)
        accumulator.add(result, start, end)


def train_mlx_microbatches(
    runtime: Any, boundary: ReplayBoundary, targets: Any, *, options: MlxMicrobatchOptions
) -> Any:
    """Train additive suffix chunks and apply the native optimizer once per logical batch.

    Parameters
    ----------
    runtime, boundary, targets
        Prepared MLX runtime, logical boundary, and matching targets.
    options
        Maximum chunk size, mean/sum semantics, loss, and optional optimizer.

    Notes
    -----
    Stateful/random suffixes follow native sequential chunk execution. BatchNorm
    statistics and random draws therefore follow each chunk's own forward.
    """

    from ._microbatch import _boundary_axes
    from .training import TrainingStepResult

    _validate_options(options)
    runtime.validate_boundary(boundary)
    axes = _boundary_axes(runtime, boundary)
    batch = boundary.metadata.get("runtime_batch_size")
    if type(batch) is not int or batch < 1:
        raise SplitBoundaryError("Microbatch training requires a positive logical batch.")
    sizes = {min(batch, options.size)}
    if batch % options.size:
        sizes.add(batch % options.size)
    for size in sizes:
        runtime.trace_graph.shape_program.require_batch_resolvable(
            size,
            runtime.plan.suffix_node_ids,
            backend=runtime.adapter.name,
            split_point=runtime.request.boundary,
        )
    with mlx_execution_context(runtime.placement.suffix.device):
        accumulator = _GradientAccumulator(boundary, axes, {}, {})
        _run_chunks(runtime, accumulator, targets, options)
    result = TrainingStepResult(
        accumulator.loss, accumulator.boundary_grads, parameter_grads=accumulator.parameter_grads
    )
    applied = stage_suffix_step(
        boundary, runtime.segments.suffix._binding, result, options.optimizer
    )
    return replace(
        result,
        optimizer_applied=applied,
        optimizer_pending=bool(
            options.optimizer is not None and result.parameter_grads and not applied
        ),
    )


__all__ = ["MlxMicrobatchOptions", "train_mlx_microbatches"]
