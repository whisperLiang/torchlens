"""One logical MLX parameter/buffer value with device-local segment replicas."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ._mlx_devices import mlx_execution_context


def snapshot_mlx_values(values: dict[int, Any]) -> dict[int, Any]:
    """Save evaluated, independent array handles for a functional forward replay."""

    import mlx.core as mx

    saved = {key: mx.stop_gradient(value).astype(value.dtype) for key, value in values.items()}
    mx.eval(*saved.values())
    return saved


class MlxStateCoordinator:
    """Synchronize tied state without materializing lazy bindings during construction."""

    def __init__(self, *bindings: Any) -> None:
        """Connect a runtime's segment bindings to a private shared state authority."""

        self.bindings = bindings
        self.values: dict[int, Any] = {}
        self.mutable_sources: set[int] = set()
        for binding in bindings:
            binding.coordinator = self

    def publish(self, values: dict[int, Any], *, parameters: bool = False) -> None:
        """Commit one logical update and transport it to each existing segment replica."""

        import mlx.core as mx

        mx.eval(*values.values())
        self.values.update(values)
        if not parameters:
            self.mutable_sources.update(values)
        for binding in self.bindings:
            changed_parameters = False
            for source_id, value in values.items():
                entry = binding.state._entries.get(source_id)
                if entry is None:
                    continue
                with mlx_execution_context(binding.state.placement.device):
                    local = binding.adapter.replicate_state(
                        value, binding.state.placement.device, trainable=entry.trainable
                    )
                    mx.eval(local)
                binding.state._entries[source_id] = replace(entry, ownership="owned", value=local)
                changed_parameters |= parameters and entry.trainable
            if changed_parameters:
                binding.state.version += 1


__all__ = ["MlxStateCoordinator", "snapshot_mlx_values"]
