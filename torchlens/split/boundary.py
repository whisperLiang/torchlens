"""Replay boundary payloads for TorchLens split runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .errors import SplitBoundaryError
from .shape import ShapeEnv, SymbolicShape, validate_tensor_against_symbolic_shape


@dataclass(frozen=True)
class BoundaryTensorSpec:
    """ABI metadata for one boundary tensor."""

    canonical_id: str
    torchlens_label: str
    module_path: str
    op_type: str
    shape: SymbolicShape | None
    dtype: torch.dtype | None
    requires_grad: bool
    role: str
    output_index: int | None = None
    device_policy: str = "runtime"

    @classmethod
    def from_trace_node(cls, node: Any, *, role: str) -> "BoundaryTensorSpec":
        """Build a boundary tensor spec from a TraceNode."""

        output_index = getattr(node.layer, "iterable_output_index", None)
        return cls(
            canonical_id=node.canonical_id,
            torchlens_label=node.torchlens_label,
            module_path=node.module_path,
            op_type=node.op_type,
            shape=node.symbolic_output_shape,
            dtype=node.dtype,
            requires_grad=node.requires_grad,
            role=role,
            output_index=output_index,
        )


@dataclass(frozen=True)
class ReplayBoundary:
    """Boundary tensors and schema passed between prefix and suffix."""

    tensors: dict[str, torch.Tensor]
    spec: dict[str, BoundaryTensorSpec]
    metadata: dict[str, Any]

    @property
    def split_id(self) -> str | None:
        """Return split id metadata."""

        return self.metadata.get("split_id")

    @property
    def batch_size(self) -> int | None:
        """Return boundary batch size metadata."""

        value = self.metadata.get("batch_size")
        return int(value) if value is not None else None

    def validate(
        self,
        expected_spec: dict[str, BoundaryTensorSpec] | None = None,
        *,
        shape_env: ShapeEnv | None = None,
        split_id: str | None = None,
        require_grad_match: bool = False,
    ) -> None:
        """Validate boundary ABI without checking device."""

        if split_id is not None and self.metadata.get("split_id") != split_id:
            raise SplitBoundaryError(
                f"Boundary split_id {self.metadata.get('split_id')!r} does not match {split_id!r}."
            )
        expected = self.spec if expected_spec is None else expected_spec
        if set(self.tensors) != set(expected):
            raise SplitBoundaryError(
                f"Boundary tensor ids {sorted(self.tensors)!r} do not match expected "
                f"{sorted(expected)!r}."
            )
        if set(self.spec) != set(expected):
            raise SplitBoundaryError("Boundary spec tensor ids do not match runtime spec.")
        for tensor_id, spec in expected.items():
            actual_spec = self.spec[tensor_id]
            _validate_spec_compatible(tensor_id, actual_spec, spec)
            tensor = self.tensors[tensor_id]
            env = shape_env or ShapeEnv()
            validate_tensor_against_symbolic_shape(
                tensor,
                spec.shape,
                dtype=spec.dtype,
                shape_env=env,
                name=tensor_id,
            )
            if require_grad_match and bool(tensor.requires_grad) != spec.requires_grad:
                raise SplitBoundaryError(
                    f"{tensor_id} requires_grad={tensor.requires_grad} does not match "
                    f"expected {spec.requires_grad}."
                )

    def to(self, device: torch.device | str) -> "ReplayBoundary":
        """Return boundary with tensors moved to ``device``."""

        return replace(self, tensors={key: tensor.to(device) for key, tensor in self.tensors.items()})

    def detach(self) -> "ReplayBoundary":
        """Return boundary with detached tensors."""

        return replace(self, tensors={key: tensor.detach() for key, tensor in self.tensors.items()})

    def cpu(self) -> "ReplayBoundary":
        """Return CPU boundary."""

        return self.to("cpu")

    def cuda(self) -> "ReplayBoundary":
        """Return CUDA boundary."""

        return self.to("cuda")

    def clone(self) -> "ReplayBoundary":
        """Return boundary with cloned tensors."""

        return replace(self, tensors={key: tensor.clone() for key, tensor in self.tensors.items()})

    @classmethod
    def collate(cls, boundaries: list["ReplayBoundary"]) -> "ReplayBoundary":
        """Collate single-sample or batch boundaries along dim 0."""

        if not boundaries:
            raise SplitBoundaryError("ReplayBoundary.collate() requires at least one boundary.")
        first = boundaries[0]
        for boundary in boundaries[1:]:
            if boundary.spec != first.spec:
                raise SplitBoundaryError("Cannot collate ReplayBoundary objects with different specs.")
        tensors: dict[str, torch.Tensor] = {}
        for tensor_id in first.tensors:
            values = [boundary.tensors[tensor_id] for boundary in boundaries]
            if values[0].ndim == 0:
                tensors[tensor_id] = torch.stack(values, dim=0)
            else:
                tensors[tensor_id] = torch.cat(values, dim=0)
        metadata = dict(first.metadata)
        batch_size = None
        for tensor in tensors.values():
            if tensor.ndim > 0:
                batch_size = int(tensor.shape[0])
                break
        metadata["batch_size"] = batch_size
        return cls(tensors=tensors, spec=first.spec, metadata=metadata)


def _validate_spec_compatible(
    tensor_id: str,
    actual: BoundaryTensorSpec,
    expected: BoundaryTensorSpec,
) -> None:
    fields = ("canonical_id", "shape", "dtype", "role", "output_index")
    for field_name in fields:
        if getattr(actual, field_name) != getattr(expected, field_name):
            raise SplitBoundaryError(
                f"Boundary spec mismatch for {tensor_id}.{field_name}: "
                f"{getattr(actual, field_name)!r} != {getattr(expected, field_name)!r}."
            )


__all__ = ["BoundaryTensorSpec", "ReplayBoundary"]
