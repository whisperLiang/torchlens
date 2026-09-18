"""Torch suffix microbatch execution using the existing symbolic boundary ABI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .boundary import ReplayBoundary
from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError
from .training import TrainingStepResult, _default_loss, _is_diff_tensor


def _unsupported(runtime: Any, message: str, reason: str) -> SplitUnsupportedError:
    """Build a structured refusal for an unsupported execution policy."""

    return SplitUnsupportedError(
        message,
        context=SplitErrorContext(
            backend=runtime.adapter.name, split_point=runtime.request.boundary, reason=reason
        ),
    )


def _boundary_axes(runtime: Any, boundary: ReplayBoundary) -> dict[str, int | None]:
    """Resolve each boundary's batch axis without inferring semantics from extents."""

    program = runtime.trace_graph.shape_program
    if program is None or not program.has_batch_axes:
        raise _unsupported(
            runtime,
            "Microbatch training requires symbolic batch axes.",
            "microbatch_batch_unavailable",
        )
    result: dict[str, int | None] = {}
    for key, schema in runtime.boundary_spec.items():
        shape = program.value_shapes.get(schema.canonical_id)
        if shape is None or schema.canonical_id in program.unresolved:
            raise _unsupported(
                runtime,
                f"Boundary {key!r} has no resolved symbolic shape for microbatch slicing.",
                "microbatch_boundary_shape_unsupported",
            )
        axes = [index for index, dim in enumerate(shape.dims) if dim.contains(program.batch_symbol)]
        if len(axes) > 1 or any(shape.dims[index].op != "symbol" for index in axes):
            raise _unsupported(
                runtime,
                f"Boundary {key!r} must have at most one plain batch symbol axis; "
                "compound and multiple batch axes cannot be sliced independently.",
                "microbatch_boundary_shape_unsupported",
            )
        result[key] = axes[0] if axes else None
    return result


def _slice_targets(value: Any, start: int, end: int, batch: int) -> Any:
    """Slice tensor leaves and sample lists, recursively preserving target containers."""

    import torch

    if isinstance(value, torch.Tensor):
        return value[start:end] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, Mapping):
        # A plain mapping preserves keys for immutable/custom mapping classes
        # whose constructors need more than an iterable (e.g. mappingproxy).
        return {key: _slice_targets(item, start, end, batch) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [_slice_targets(item, start, end, batch) for item in value]
        return type(value)(*items) if hasattr(type(value), "_fields") else tuple(items)
    if isinstance(value, list):
        if len(value) == batch:
            return value[start:end]
        return [_slice_targets(item, start, end, batch) for item in value]
    return value


def _run_chunk(
    runtime: Any,
    boundary: ReplayBoundary,
    axes: dict[str, int | None],
    targets: Any,
    start: int,
    end: int,
    weight: float,
    loss_fn: Callable[[Any, Any], Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """Backpropagate one suffix graph; only detached results escape this scope."""

    import torch

    roots: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        axis = axes[key]
        sliced = value if axis is None else value.narrow(axis, start, end - start)
        if _is_diff_tensor(torch, sliced):
            # The transported logical boundary is already detached. A per-chunk
            # clone isolates public boundary storage from replay's in-place ops.
            sliced = sliced.detach().clone().requires_grad_(True)
            roots[key] = sliced
        values[key] = sliced
    replay = ReplayBoundary(
        backend=boundary.backend,
        tensors=values,
        spec=boundary.spec,
        metadata={**boundary.metadata, "runtime_batch_size": end - start},
    )
    # Logical boundary identity and every chunk's batch eligibility were checked
    # before accumulation; suffix execution still performs its normal shape guards.
    output = runtime.segments.suffix(replay)
    loss = (
        loss_fn(output, targets) if loss_fn is not None else _default_loss(torch, output, targets)
    )
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or loss.is_complex():
        raise _unsupported(
            runtime,
            "Microbatch loss_fn must return a real scalar tensor.",
            "microbatch_loss_invalid",
        )
    scaled = loss * weight
    scaled.backward()
    return scaled.detach(), {
        key: root.grad.detach() for key, root in roots.items() if root.grad is not None
    }


def _accumulate_gradients(
    full: dict[str, Any],
    chunk: dict[str, Any],
    boundary: ReplayBoundary,
    axes: dict[str, int | None],
    start: int,
    end: int,
) -> None:
    """Copy batched gradients into final storage and sum shared-value contributions."""

    import torch

    for key, grad in chunk.items():
        axis = axes[key]
        if axis is None:
            if key in full:
                full[key].add_(grad)
            else:
                full[key] = grad
        else:
            if key not in full:
                # Unused roots in a chunk contribute zero; allocate once, with
                # no list of gradient clones and no final concatenation copy.
                full[key] = torch.zeros_like(boundary.tensors[key])
            full[key].narrow(axis, start, end - start).copy_(grad)


def train_torch_microbatches(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    *,
    loss_fn: Callable[[Any, Any], Any] | None,
    optimizer: Any | None,
    microbatch_size: int,
    microbatch_reduction: str,
    target_slicer: Callable[[Any, int, int, int], Any] | None,
) -> TrainingStepResult:
    """Train one logical batch with one optimizer step and bounded suffix graphs.

    Parameters
    ----------
    runtime, boundary, targets
        Prepared Torch runtime, logical boundary, and targets on the suffix device.
    loss_fn, optimizer
        Scalar loss and optional caller-owned optimizer.
    microbatch_size
        Positive maximum chunk size; an uneven last chunk is supported.
    microbatch_reduction
        Mean losses are weighted by chunk/logical size. Sum requires an explicit
        custom sum loss. Equivalence requires independent samples and an additive
        objective with a sample-count denominator for mean reduction.
    target_slicer
        Optional callback ``(targets, start, end, logical_batch) -> chunk``.

    Returns
    -------
    TrainingStepResult
        Detached logical loss, full boundary gradients, and optimizer-step status.
    """

    if type(microbatch_size) is not int or microbatch_size < 1:
        raise ValueError("microbatch_size must be a positive integer")
    if microbatch_reduction not in {"mean", "sum"}:
        raise ValueError("microbatch_reduction must be 'mean' or 'sum'")
    if microbatch_reduction == "sum" and loss_fn is None:
        raise ValueError("microbatch_reduction='sum' requires a custom sum-reduced loss_fn")
    runtime.validate_boundary(boundary)
    axes = _boundary_axes(runtime, boundary)
    batch = boundary.metadata.get("runtime_batch_size")
    if type(batch) is not int or batch < 1:
        raise _unsupported(
            runtime,
            "Microbatch training requires a positive logical batch.",
            "microbatch_batch_unavailable",
        )
    program = runtime.trace_graph.shape_program
    sizes = {min(batch, microbatch_size)}
    if batch % microbatch_size:
        sizes.add(batch % microbatch_size)
    try:
        for size in sizes:
            program.require_batch_resolvable(
                size,
                runtime.plan.suffix_node_ids,
                backend=runtime.adapter.name,
                split_point=runtime.request.boundary,
            )
    except SplitBoundaryError as exc:
        raise _unsupported(runtime, str(exc), "microbatch_shape_unsupported") from exc
    # Transport the logical payload once, detached before copying. Keep original
    # graph-connected prefix tensors solely in the caller's training boundary.
    metadata = {
        key: value for key, value in boundary.metadata.items() if key != "prefix_boundary_tensors"
    }
    metadata["supports_prefix_backward"] = False
    logical = ReplayBoundary(
        boundary.backend,
        {key: runtime.adapter.detach(value) for key, value in boundary.tensors.items()},
        boundary.spec,
        metadata,
    )
    logical = runtime._transport_boundary(logical, runtime.placement.suffix)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    full_grads: dict[str, Any] = {}
    loss_total: Any = None
    for start in range(0, batch, microbatch_size):
        end = min(start + microbatch_size, batch)
        chunk_targets = (
            target_slicer(targets, start, end, batch)
            if target_slicer is not None
            else _slice_targets(targets, start, end, batch)
        )
        loss, grads = _run_chunk(
            runtime,
            logical,
            axes,
            chunk_targets,
            start,
            end,
            (end - start) / batch if microbatch_reduction == "mean" else 1.0,
            loss_fn,
        )
        if loss_total is None:
            loss_total = loss
        else:
            loss_total.add_(loss)
        _accumulate_gradients(full_grads, grads, logical, axes, start, end)
        del loss, grads, chunk_targets
    if optimizer is not None:
        optimizer.step()
    return TrainingStepResult(loss_total, full_grads, optimizer_applied=optimizer is not None)
