"""Coordinated MLX optimizer steps after both halves of a split have differentiated."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters._mlx_devices import mlx_execution_context
from .errors import SplitUnsupportedError


def validate_mlx_optimizer(optimizer: Any) -> None:
    """Require the native functional optimizer interface when an optimizer is supplied."""

    if optimizer is not None and not callable(getattr(optimizer, "apply_gradients", None)):
        raise SplitUnsupportedError("MLX split training requires an apply_gradients optimizer.")


@dataclass
class _OptimizerGroup:
    """One native optimizer invocation over unique logical parameter identities."""

    optimizer: Any
    binding: Any
    sources: dict[str, int] = field(default_factory=dict)
    gradients: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, source_id: int, gradient: Any) -> None:
        """Assign a parameter exactly once to this optimizer invocation."""

        self.sources[name] = source_id
        self.gradients[name] = gradient

    def apply(self) -> bool:
        """Evaluate, validate, and synchronize one functional native optimizer result."""

        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        if not self.sources:
            return False
        adapter, device = self.binding.adapter, self.binding.state.placement.device
        coordinator = self.binding.coordinator
        with mlx_execution_context(device):
            parameters = {
                name: adapter.to_device(coordinator.values[source_id], device)
                for name, source_id in self.sources.items()
            }
            gradients = {
                name: adapter.to_device(value, device) for name, value in self.gradients.items()
            }
            updated_tree = self.optimizer.apply_gradients(
                tree_unflatten(list(gradients.items())), tree_unflatten(list(parameters.items()))
            )
            updated = dict(tree_flatten(updated_tree))
            _validate_update(adapter, parameters, updated)
            mx.eval(updated, getattr(self.optimizer, "state", {}))
        coordinator.publish(
            {self.sources[name]: value for name, value in updated.items()}, parameters=True
        )
        return True


def _validate_update(adapter: Any, parameters: dict[str, Any], updated: Any) -> None:
    """Check the complete optimizer output before any segment parameter is committed."""

    if not isinstance(updated, dict) or updated.keys() != parameters.keys():
        raise SplitUnsupportedError("MLX optimizer returned a different parameter inventory.")
    for name, value in updated.items():
        if (
            not adapter.is_tensor(value)
            or value.shape != parameters[name].shape
            or value.dtype != parameters[name].dtype
        ):
            raise SplitUnsupportedError(f"MLX optimizer changed parameter {name!r} shape or dtype.")


def _parameter_sources(binding: Any) -> dict[str, int]:
    """Map the segment's effective trainable inventory to canonical source identities."""

    names = binding.parameters()
    return {
        name: source_id
        for source_id, (name, _trainable) in binding._parameters.items()
        if name in names
    }


@dataclass
class _SuffixStep:
    """Evaluated suffix gradients awaiting the corresponding prefix pullback."""

    binding: Any
    result: Any
    optimizer: Any


def stage_suffix_step(boundary: Any, binding: Any, result: Any, optimizer: Any) -> bool:
    """Defer connected steps; apply detached/cache suffix-only steps immediately."""

    validate_mlx_optimizer(optimizer)
    if boundary.metadata.get("supports_prefix_backward"):
        boundary.metadata.setdefault("mlx_suffix_steps", []).append(
            _SuffixStep(binding, result, optimizer)
        )
        return False
    if optimizer is None:
        return False
    group = _OptimizerGroup(optimizer, binding)
    for name, source_id in _parameter_sources(binding).items():
        group.add(name, source_id, result.parameter_grads[name])
    return group.apply()


def _matching_steps(boundary: Any, boundary_grads: Any) -> list[_SuffixStep]:
    """Select an individual suffix pullback, or all accumulated suffix contributions."""

    steps = boundary.metadata.get("mlx_suffix_steps", [])
    matched = [step for step in steps if step.result.boundary_grads is boundary_grads]
    selected = matched or list(steps)
    selected_ids = {id(step) for step in selected}
    boundary.metadata["mlx_suffix_steps"] = [step for step in steps if id(step) not in selected_ids]
    return selected


def _combined_gradients(
    binding: Any, local: dict[str, Any], steps: list[_SuffixStep]
) -> dict[str, Any]:
    """Sum every contribution to tied weights while retaining each exclusive gradient."""

    import mlx.core as mx

    result = dict(local)
    for step in steps:
        for name, value in step.result.parameter_grads.items():
            if name in result:
                with mlx_execution_context(binding.state.placement.device):
                    result[name] = mx.add(
                        result[name],
                        binding.adapter.to_device(value, binding.state.placement.device),
                    )
            else:
                result[name] = value
    mx.eval(result)
    return result


def _optimizer_groups(
    binding: Any, gradients: dict[str, Any], optimizer: Any, steps: list[_SuffixStep]
) -> list[_OptimizerGroup]:
    """Assign tied weights to the prefix optimizer, falling back to the first suffix owner."""

    groups: dict[int, _OptimizerGroup] = {}
    owners: set[int] = set()

    def assign(owner: Any, native_optimizer: Any) -> None:
        """Include only parameters not already assigned to another optimizer."""

        if native_optimizer is None:
            return
        group = groups.setdefault(id(native_optimizer), _OptimizerGroup(native_optimizer, owner))
        for name, source_id in _parameter_sources(owner).items():
            if source_id not in owners:
                group.add(name, source_id, gradients[name])
                owners.add(source_id)

    assign(binding, optimizer)
    for step in steps:
        assign(step.binding, step.optimizer)
    return list(groups.values())


def finish_prefix_step(
    boundary: Any,
    binding: Any,
    local_gradients: dict[str, Any],
    optimizer: Any,
    boundary_grads: Any,
) -> dict[str, Any]:
    """Complete one logical training step, including summed tied-parameter gradients."""

    validate_mlx_optimizer(optimizer)
    steps = _matching_steps(boundary, boundary_grads)
    gradients = _combined_gradients(binding, local_gradients, steps)
    groups = _optimizer_groups(binding, gradients, optimizer, steps)
    applied = sum(group.apply() for group in groups)
    shared = {
        name: gradients[name]
        for name in local_gradients
        if any(name in step.result.parameter_grads for step in steps)
    }
    return {
        "all_parameter_grads": gradients,
        "shared_parameter_grads": shared,
        "optimizer_applied": bool(applied),
        "optimizer_step_count": applied,
    }


__all__ = ["finish_prefix_step", "stage_suffix_step", "validate_mlx_optimizer"]
