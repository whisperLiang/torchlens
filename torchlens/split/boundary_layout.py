"""Static communication layout derived from the existing boundary ABI.

This module describes logical payloads, not a wire format. It neither packs
tensors nor assumes that unknown alias groups have disjoint storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import SplitBoundaryError, SplitErrorContext
from .graph_profile import GraphProfile, _may_require_gradient, _sum_known
from .planner import SplitPlan


@dataclass(frozen=True)
class BoundaryTensorLayout:
    """One unique canonical tensor and its public boundary dictionary key.

    ``backward_required=None`` means a differentiable payload whose use by a
    future loss is unknown. ``requires_grad`` describes capture evidence; the
    Torch suffix training engine can create roots even for captured inputs that
    did not require gradients. ``alias_group=None`` means unestablished aliasing.
    """

    key: str
    value_id: str
    shape: tuple[int, ...] | None
    dtype: str | None
    bytes: int | None
    requires_grad: bool | None
    container_path: tuple[Any, ...]
    alias_group: str | None
    role: str
    forward_required: bool
    backward_required: bool | None


@dataclass(frozen=True)
class BoundaryLayout:
    """Immutable logical payload requirements for a split and requested batch."""

    graph_identity: str | None
    split_id: str
    batch_size: int | None
    training: bool
    tensors: tuple[BoundaryTensorLayout, ...]
    unique_tensor_count: int
    forward_payload_bytes: int | None
    backward_payload_bytes: int | None
    unknown_cost_diagnostics: tuple[str, ...]


def build_boundary_layout(
    profile: GraphProfile, plan: SplitPlan, *, training: bool = False
) -> BoundaryLayout:
    """Describe boundary values using static graph shapes and canonical bindings.

    Parameters
    ----------
    profile:
        Graph costs at the desired batch, obtained without executing the model.
    plan:
        Validated point or DAG partition plan on this graph.
    training:
        Whether to include the backward payload contract. Loss-dependent
        gradient presence remains unknown until observed by actual training.

    Returns
    -------
    BoundaryLayout
        Deduplicated value rows. Byte totals describe dense logical tensors,
        excluding transport headers, allocator overhead and encoding choices.
    """

    tensors = {tensor.value_id: tensor for tensor in profile.tensors}
    rows: list[BoundaryTensorLayout] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for key, schema in plan.boundary_spec.items():
        value_id = plan.boundary_bindings[key]
        tensor = tensors.get(value_id)
        if tensor is None:
            raise SplitBoundaryError(
                f"Boundary value {value_id!r} is absent from the supplied graph profile.",
                context=SplitErrorContext(
                    backend=profile.backend,
                    split_point=plan.split_id,
                    reason="boundary_layout_value_unavailable",
                ),
            )
        if value_id in seen:
            continue
        seen.add(value_id)
        dtype = schema.dtype or tensor.dtype
        # Every backend uses floating/complex values for differentiation. The
        # future loss can ignore any leaf, so these are potential requirements.
        backward: bool | None = False
        if training and _may_require_gradient(dtype):
            backward = None
            unknown.append(f"backward_payload:{value_id}:loss_path_not_observed")
        if tensor.bytes is None:
            unknown.append(f"forward_payload:{value_id}:shape_or_dtype_unavailable")
        alias_group = schema.alias_group or tensor.alias_group
        if alias_group is None:
            unknown.append(f"alias_group:{value_id}:unavailable")
        rows.append(
            BoundaryTensorLayout(
                key=key,
                value_id=value_id,
                shape=tensor.shape,
                dtype=dtype,
                bytes=tensor.bytes,
                requires_grad=schema.requires_grad,
                container_path=schema.container_path,
                alias_group=alias_group,
                role=schema.role,
                forward_required=True,
                backward_required=backward,
            )
        )
    return BoundaryLayout(
        graph_identity=profile.graph_identity,
        split_id=plan.split_id,
        batch_size=profile.batch_size,
        training=training,
        tensors=tuple(rows),
        unique_tensor_count=len(rows),
        forward_payload_bytes=_sum_known(row.bytes for row in rows),
        backward_payload_bytes=None if any(row.backward_required is None for row in rows) else 0,
        unknown_cost_diagnostics=tuple(unknown),
    )
