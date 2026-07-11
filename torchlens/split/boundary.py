"""Replay boundary container and ABI validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapters.base import SplitBackendAdapter
from .errors import SplitBoundaryError, SplitErrorContext
from .shape import validate_tensor_against_symbolic_shape
from .ir import BoundarySchema


_COLLATE_METADATA_KEYS = (
    "split_id",
    "graph_shape_hash",
    "batch_symbol",
    "dynamic_batch",
    "device_policy",
    "profile_hash",
    "state_fingerprint",
)


@dataclass(frozen=True)
class ReplayBoundary:
    """Immutable split replay boundary payload."""

    backend: str
    tensors: dict[str, Any]
    spec: dict[str, BoundarySchema]
    metadata: dict[str, Any]

    @staticmethod
    def _collate_metadata_abi(boundary: "ReplayBoundary") -> tuple[Any, ...]:
        """Return metadata fields that identify a compatible boundary ABI."""

        return tuple(boundary.metadata.get(key) for key in _COLLATE_METADATA_KEYS)

    def _adapter(self, adapter: SplitBackendAdapter | None) -> SplitBackendAdapter:
        """Resolve an adapter for boundary tensor operations."""

        from .adapters import resolve_split_adapter

        return adapter if adapter is not None else resolve_split_adapter(self.backend)

    def validate(
        self,
        expected: dict[str, BoundarySchema] | None = None,
        *,
        split_id: str | None = None,
        graph_hash: str | None = None,
        profile_hash: str | None = None,
        state_fingerprint: str | None = None,
        adapter: SplitBackendAdapter | None = None,
    ) -> None:
        """Validate this boundary against its ABI spec.

        Parameters
        ----------
        expected:
            Expected runtime boundary spec. Defaults to ``self.spec``.
        split_id:
            Optional expected split ID.
        adapter:
            Optional split backend adapter.
        """

        resolved_adapter = self._adapter(adapter)
        expected_spec = self.spec if expected is None else expected
        if split_id is not None and self.metadata.get("split_id") != split_id:
            raise SplitBoundaryError(
                "Replay boundary split_id does not match this runtime.",
                context=SplitErrorContext(
                    backend=self.backend,
                    split_point=str(split_id),
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="split_id mismatch",
                ),
            )
        expected_metadata = (
            ("graph_shape_hash", graph_hash),
            ("profile_hash", profile_hash),
            ("state_fingerprint", state_fingerprint),
        )
        for field_name, expected_value in expected_metadata:
            if expected_value is None or field_name not in self.metadata:
                continue
            if self.metadata[field_name] != expected_value:
                raise SplitBoundaryError(
                    f"Replay boundary {field_name} does not match this runtime.",
                    context=SplitErrorContext(
                        backend=self.backend,
                        split_point=str(split_id or self.metadata.get("split_id", "")),
                        module_path=None,
                        op_type=None,
                        layer_label=None,
                        reason=f"{field_name} mismatch",
                    ),
                )
        if set(self.tensors) != set(expected_spec):
            raise SplitBoundaryError(
                "Replay boundary tensor IDs do not match the boundary spec.",
                context=SplitErrorContext(
                    backend=self.backend,
                    split_point=str(split_id or self.metadata.get("split_id", "")),
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="tensor id set mismatch",
                ),
            )
        if set(self.spec) != set(expected_spec):
            raise SplitBoundaryError(
                "Replay boundary spec IDs do not match the expected spec.",
                context=SplitErrorContext(
                    backend=self.backend,
                    split_point=str(split_id or self.metadata.get("split_id", "")),
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="spec id set mismatch",
                ),
            )
        for key, expected_item in expected_spec.items():
            actual_item = self.spec[key]
            if actual_item.canonical_id != expected_item.canonical_id:
                raise SplitBoundaryError(f"Boundary canonical ID mismatch for {key!r}.")
            if actual_item.role != expected_item.role:
                raise SplitBoundaryError(f"Boundary role mismatch for {key!r}.")
            if actual_item.output_index != expected_item.output_index:
                raise SplitBoundaryError(f"Boundary output index mismatch for {key!r}.")
            if tuple(actual_item.container_path) != tuple(expected_item.container_path):
                raise SplitBoundaryError(f"Boundary container path mismatch for {key!r}.")
            value = self.tensors[key]
            dtype = resolved_adapter.dtype_name(value)
            if (
                expected_item.dtype is not None
                and dtype is not None
                and dtype != expected_item.dtype
            ):
                raise SplitBoundaryError(
                    f"Boundary tensor {key!r} dtype is {dtype}, expected {expected_item.dtype}.",
                    context=SplitErrorContext(
                        backend=self.backend,
                        split_point=str(split_id or self.metadata.get("split_id", "")),
                        module_path=expected_item.module_path,
                        op_type=expected_item.op_type,
                        layer_label=expected_item.label,
                        reason="dtype mismatch",
                        dtype=dtype,
                    ),
                )
            validate_tensor_against_symbolic_shape(
                value,
                expected_item.shape,
                adapter=resolved_adapter,
                dynamic_batch=self.metadata.get("dynamic_batch"),
                batch_symbol=self.metadata.get("batch_symbol", "B"),
                backend=self.backend,
                split_point=str(split_id or self.metadata.get("split_id", "")),
                label=expected_item.label,
            )

    def detach(self, adapter: SplitBackendAdapter | None = None) -> "ReplayBoundary":
        """Return a boundary with detached tensor values."""

        resolved_adapter = self._adapter(adapter)
        return ReplayBoundary(
            backend=self.backend,
            tensors={key: resolved_adapter.detach(value) for key, value in self.tensors.items()},
            spec=self.spec,
            metadata=dict(self.metadata),
        )

    def clone(self, adapter: SplitBackendAdapter | None = None) -> "ReplayBoundary":
        """Return a boundary with cloned tensor values."""

        resolved_adapter = self._adapter(adapter)
        return ReplayBoundary(
            backend=self.backend,
            tensors={key: resolved_adapter.clone(value) for key, value in self.tensors.items()},
            spec=self.spec,
            metadata=dict(self.metadata),
        )

    def to(self, device: Any, adapter: SplitBackendAdapter | None = None) -> "ReplayBoundary":
        """Return a boundary with tensors moved to ``device``."""

        resolved_adapter = self._adapter(adapter)
        return ReplayBoundary(
            backend=self.backend,
            tensors={
                key: resolved_adapter.to_device(value, device)
                for key, value in self.tensors.items()
            },
            spec=self.spec,
            metadata=dict(self.metadata),
        )

    def cpu(self, adapter: SplitBackendAdapter | None = None) -> "ReplayBoundary":
        """Return a CPU boundary."""

        return self.to("cpu", adapter=adapter)

    def cuda(self, adapter: SplitBackendAdapter | None = None) -> "ReplayBoundary":
        """Return a CUDA boundary."""

        return self.to("cuda", adapter=adapter)

    @classmethod
    def collate(
        cls,
        boundaries: list["ReplayBoundary"],
        adapter: SplitBackendAdapter | None = None,
    ) -> "ReplayBoundary":
        """Collate a list of same-spec boundaries by tensor key."""

        if not boundaries:
            raise SplitBoundaryError("ReplayBoundary.collate requires at least one boundary.")
        first = boundaries[0]
        resolved_adapter = first._adapter(adapter)
        first_metadata_abi = cls._collate_metadata_abi(first)
        for boundary in boundaries[1:]:
            if (
                boundary.backend != first.backend
                or set(boundary.tensors) != set(first.tensors)
                or boundary.spec != first.spec
                or cls._collate_metadata_abi(boundary) != first_metadata_abi
            ):
                raise SplitBoundaryError("Cannot collate boundaries with different ABI IDs.")
        return cls(
            backend=first.backend,
            tensors={
                key: resolved_adapter.collate([boundary.tensors[key] for boundary in boundaries])
                for key in first.tensors
            },
            spec=first.spec,
            metadata={
                **first.metadata,
                "collated": True,
                "batch_size": len(boundaries),
            },
        )


__all__ = ["ReplayBoundary"]
