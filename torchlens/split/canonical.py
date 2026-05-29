"""Canonical split identities derived from TorchLens metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .shape import ShapeEnv, SymbolicShape


@dataclass(frozen=True)
class CanonicalTensorId:
    """Device-neutral tensor identity for split ABI."""

    canonical_id: str
    module_path: str
    op_type: str
    local_occurrence: int
    symbolic_shape: SymbolicShape | None
    torchlens_layer_label: str
    dtype: torch.dtype | None
    requires_grad: bool


def build_canonical_ids(
    layers: Sequence[object],
    shape_env: ShapeEnv,
) -> dict[str, CanonicalTensorId]:
    """Create canonical ids for TorchLens layer pass logs."""

    counts: dict[tuple[str, str, SymbolicShape | None], int] = defaultdict(int)
    result: dict[str, CanonicalTensorId] = {}
    for layer in layers:
        label = str(getattr(layer, "layer_label", ""))
        module_path = normalized_module_path(layer)
        op_type = str(getattr(layer, "func_name", None) or getattr(layer, "layer_type", "op"))
        symbolic_shape = shape_env.symbolic_shape(getattr(layer, "tensor_shape", None))
        key = (module_path, op_type, symbolic_shape)
        counts[key] += 1
        local_occurrence = counts[key]
        shape_text = _format_shape(symbolic_shape)
        canonical_id = f"{module_path}/{op_type}_{local_occurrence}/{shape_text}"
        activation = getattr(layer, "activation", None)
        result[label] = CanonicalTensorId(
            canonical_id=canonical_id,
            module_path=module_path,
            op_type=op_type,
            local_occurrence=local_occurrence,
            symbolic_shape=symbolic_shape,
            torchlens_layer_label=label,
            dtype=getattr(layer, "tensor_dtype", None),
            requires_grad=bool(activation.requires_grad)
            if isinstance(activation, torch.Tensor)
            else False,
        )
    return result


def normalized_module_path(layer: object) -> str:
    """Return the semantic module path used by split identity."""

    path = getattr(layer, "module_address_normalized", None)
    if path:
        return str(path)
    containing = getattr(layer, "containing_module", None)
    if containing:
        return str(containing).split(":", maxsplit=1)[0]
    if getattr(layer, "is_input_layer", False):
        return "input"
    if getattr(layer, "is_output_layer", False):
        return "output"
    return "global"


def _format_shape(shape: SymbolicShape | None) -> str:
    if shape is None:
        return "()"
    return "(" + ",".join(str(dim) for dim in shape) + ")"


__all__ = ["CanonicalTensorId", "build_canonical_ids", "normalized_module_path"]
