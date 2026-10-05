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
        self.inventory: Any = None
        for binding in bindings:
            binding.coordinator = self

    def publish(
        self, values: dict[int, Any], *, parameters: bool = False, invalidate: bool = False
    ) -> None:
        """Commit one logical update and transport it to each existing segment replica."""

        import mlx.core as mx

        mx.eval(*values.values())
        staged = []
        for binding in self.bindings:
            entries = {}
            for source_id, value in values.items():
                entry = binding.state._entries.get(source_id)
                if entry is None:
                    continue
                with mlx_execution_context(binding.state.placement.device):
                    local = binding.adapter.replicate_state(
                        value, binding.state.placement.device, trainable=entry.trainable
                    )
                    mx.eval(local)
                entries[source_id] = replace(entry, ownership="owned", value=local)
            staged.append((binding.state, entries))
        # A failed evaluation or placement must leave every binding unchanged.
        self.values.update(values)
        if not parameters:
            self.mutable_sources.update(values)
        for state, entries in staged:
            state._entries.update(entries)
            if invalidate or (parameters and any(entry.trainable for entry in entries.values())):
                state.version += 1


__all__ = ["MlxStateCoordinator", "snapshot_mlx_values"]
