"""Functional MLX differentiation and optimizer commits for split segments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .. import _state
from ._mlx_optimizer import finish_prefix_step, stage_suffix_step, validate_mlx_optimizer
from .adapters._mlx_devices import mlx_execution_context
from .boundary import ReplayBoundary
from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError

if TYPE_CHECKING:
    from .training import TrainingStepResult


def _context(runtime: Any, reason: str) -> SplitErrorContext:
    """Return the runtime's MLX training refusal context."""

    return SplitErrorContext(
        backend=runtime.adapter.name, split_point=runtime.request.boundary, reason=reason
    )


def _is_differentiable(value: Any) -> bool:
    """Return whether MLX can differentiate this real floating array."""

    import mlx.core as mx

    return isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating)


def _default_loss(output: Any, targets: Any) -> Any:
    """Compute MSE or classification cross entropy for native MLX arrays."""

    import mlx.core as mx
    import mlx.nn as nn

    if not isinstance(output, mx.array) or not isinstance(targets, mx.array):
        raise SplitUnsupportedError("Non-array MLX outputs require an explicit loss_fn.")
    if (
        mx.issubdtype(targets.dtype, mx.integer)
        and output.ndim >= 2
        and targets.ndim == output.ndim - 1
    ):
        return nn.losses.cross_entropy(output, targets, reduction="mean")
    return nn.losses.mse_loss(output, targets, reduction="mean")


def _map_arrays(value: Any, replace: Callable[[Any], Any], *, keep_literals: bool = True) -> Any:
    """Map array leaves while preserving the built-in runtime input structure."""

    import mlx.core as mx

    if isinstance(value, mx.array):
        return replace(value)
    if isinstance(value, dict):
        return {
            key: _map_arrays(item, replace, keep_literals=keep_literals)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            _map_arrays(item, replace, keep_literals=keep_literals) for item in value
        )
    return value if keep_literals else None


def _input_roots(inputs: Any, kwargs: Any) -> list[Any]:
    """Collect unique differentiable input identities, including keyword arrays."""

    roots: dict[int, Any] = {}

    def collect(value: Any) -> Any:
        """Retain a differentiable array once, preserving tied input semantics."""

        if _is_differentiable(value):
            roots.setdefault(id(value), value)
        return value

    _map_arrays((inputs, kwargs), collect)
    return list(roots.values())


def _checked_prefix(runtime: Any, boundary: ReplayBoundary) -> Any:
    """Require the original training prefix and its unchanged capture-time state."""

    prefix = runtime.segments.training_prefix
    if prefix is None or not boundary.metadata.get("supports_prefix_backward"):
        raise SplitUnsupportedError(
            "backward_prefix requires a boundary from run_training_prefix().",
            context=_context(runtime, "boundary is not graph-connected"),
        )
    if (
        boundary.metadata.get("mlx_prefix_owner") is not prefix._backward_token
        or boundary.metadata.get("mlx_prefix_state_version") != prefix._state.version
    ):
        raise SplitBoundaryError(
            "MLX prefix boundary belongs to a different or updated training prefix.",
            context=_context(runtime, "stale MLX training boundary"),
        )
    saved_state = boundary.metadata.get("mlx_prefix_state_values", {})
    for entry in prefix._state.entries():
        if entry.source_id in prefix._binding.coordinator.mutable_sources:
            continue
        saved_value = saved_state.get(entry.source_id)
        if saved_value is None or not runtime.adapter.allclose(
            entry.value, saved_value, atol=0, rtol=0
        ):
            raise SplitBoundaryError(
                "MLX prefix state changed after run_training_prefix().",
                context=_context(runtime, "stale MLX training boundary"),
            )
    return prefix


def _prefix_cotangents(
    runtime: Any, boundary: ReplayBoundary, boundary_grads: dict[str, Any]
) -> tuple[list[str], list[Any]]:
    """Validate incoming gradient keys and shape/dtype before transporting cotangents."""

    import mlx.core as mx

    unknown = boundary_grads.keys() - boundary.tensors.keys()
    if unknown:
        raise SplitBoundaryError(f"Unknown MLX boundary gradient keys: {sorted(unknown)!r}.")
    keys = [key for key, value in boundary.tensors.items() if _is_differentiable(value)]
    cotangents: list[Any] = []
    for key in keys:
        value = boundary.tensors[key]
        grad = boundary_grads.get(key)
        if grad is None:
            grad = mx.zeros_like(value)
        elif (
            not isinstance(grad, mx.array) or grad.shape != value.shape or grad.dtype != value.dtype
        ):
            raise SplitBoundaryError(f"MLX boundary gradient {key!r} has the wrong shape or dtype.")
        cotangents.append(runtime.adapter.to_device(grad, runtime.placement.prefix.device))
    nondiff = boundary_grads.keys() - set(keys)
    if nondiff:
        raise SplitBoundaryError(
            f"MLX boundary gradients target non-floating arrays: {sorted(nondiff)!r}."
        )
    return keys, cotangents


class MlxTrainingEngine:
    """Differentiate MLX suffixes and recompute prefix VJPs over private parameters."""

    name = "mlx"

    def train_microbatches(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        **options: Any,
    ) -> TrainingStepResult:
        """Differentiate suffix chunks with one coordinated native optimizer step."""

        from ._mlx_microbatch import MlxMicrobatchOptions, train_mlx_microbatches

        return train_mlx_microbatches(
            runtime,
            boundary,
            targets,
            options=MlxMicrobatchOptions(
                options["microbatch_size"],
                options["microbatch_reduction"],
                options["loss_fn"],
                options["optimizer"],
                options["target_slicer"],
            ),
        )

    def train_suffix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        optimizer: Any | None = None,
    ) -> TrainingStepResult:
        """Return suffix boundary/parameter gradients and optionally update its state."""

        import mlx.core as mx

        from .training import TrainingStepResult

        runtime.validate_boundary(boundary)
        validate_mlx_optimizer(optimizer)
        binding = runtime.segments.suffix._binding
        parameters = binding.parameters()
        roots = {key: value for key, value in boundary.tensors.items() if _is_differentiable(value)}

        def loss(variables: dict[str, Any]) -> Any:
            """Evaluate the suffix loss with independent functional AD roots."""

            replay_boundary = ReplayBoundary(
                backend=boundary.backend,
                tensors={**boundary.tensors, **variables["boundary"]},
                spec=boundary.spec,
                metadata=dict(boundary.metadata),
            )
            with binding.using_parameters(variables["parameters"]):
                output = runtime._run_suffix_unchecked(replay_boundary)
            result = (
                loss_fn(output, targets) if loss_fn is not None else _default_loss(output, targets)
            )
            if not isinstance(result, mx.array) or result.ndim != 0:
                raise SplitUnsupportedError(
                    "MLX split loss_fn must return a scalar MLX array.",
                    context=_context(runtime, "invalid MLX scalar loss"),
                )
            return result

        with _state.pause_logging(), mlx_execution_context(runtime.placement.suffix.device):
            with binding.functional_forward() as buffer_updates:
                value, gradients = mx.value_and_grad(loss)(
                    {"boundary": roots, "parameters": parameters}
                )
                mx.eval(value, gradients, buffer_updates)
            binding.coordinator.publish(buffer_updates)
        result = TrainingStepResult(
            value,
            gradients["boundary"],
            parameter_grads=gradients["parameters"],
        )
        applied = stage_suffix_step(boundary, binding, result, optimizer)
        return replace(
            result,
            optimizer_applied=applied,
            optimizer_pending=bool(optimizer is not None and parameters and not applied),
        )

    def backward_prefix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        boundary_grads: dict[str, Any],
        optimizer: Any | None = None,
    ) -> dict[str, Any]:
        """Recompute a prefix VJP and return input and named parameter gradients."""

        import mlx.core as mx

        runtime.validate_boundary(boundary)
        prefix = _checked_prefix(runtime, boundary)
        keys, cotangents = _prefix_cotangents(runtime, boundary, boundary_grads)
        inputs = boundary.metadata["prefix_inputs"]
        kwargs = boundary.metadata["prefix_input_kwargs"]
        input_roots = _input_roots(inputs, kwargs)
        binding = prefix._binding
        parameters = binding.parameters()
        names = list(parameters)
        primals = [*input_roots, *parameters.values()]

        def forward(*values: Any) -> list[Any]:
            """Rebuild the original input tree and captured parameter identities."""

            replacements = {id(root): values[index] for index, root in enumerate(input_roots)}
            replay_inputs = _map_arrays(inputs, lambda value: replacements.get(id(value), value))
            replay_kwargs = _map_arrays(kwargs, lambda value: replacements.get(id(value), value))
            replay_parameters = dict(zip(names, values[len(input_roots) :], strict=True))
            with (
                binding.using_parameters(replay_parameters),
                binding.replaying(
                    boundary.metadata["mlx_prefix_replay_state"],
                    boundary.metadata["mlx_prefix_rng"],
                ),
            ):
                replayed = prefix(*replay_inputs, input_kwargs=replay_kwargs, detach_boundary=False)
            return [replayed.tensors[key] for key in keys]

        with _state.pause_logging(), mlx_execution_context(runtime.placement.prefix.device):
            if primals and keys:
                outputs, grads = mx.vjp(forward, primals, cotangents)
                mx.eval(outputs, grads)
                for key, output in zip(keys, outputs, strict=True):
                    if not runtime.adapter.allclose(output, boundary.tensors[key], atol=0, rtol=0):
                        raise SplitBoundaryError(
                            "MLX prefix inputs or state changed after run_training_prefix()."
                        )
            else:
                grads = [mx.zeros_like(value) for value in primals]
            parameter_grads = dict(zip(names, grads[len(input_roots) :], strict=True))
            input_grads = dict(
                zip((id(value) for value in input_roots), grads[: len(input_roots)], strict=True)
            )
            completed = finish_prefix_step(
                boundary, binding, parameter_grads, optimizer, boundary_grads
            )
        return {
            "inputs": _map_arrays(
                inputs, lambda value: input_grads.get(id(value)), keep_literals=False
            ),
            "input_kwargs": _map_arrays(
                kwargs, lambda value: input_grads.get(id(value)), keep_literals=False
            ),
            "parameter_grads": parameter_grads,
            **completed,
        }


__all__ = ["MlxTrainingEngine"]
