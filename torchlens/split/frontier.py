"""Boundary frontier construction for split plans."""

from __future__ import annotations

from typing import Any

from .ir import BoundaryRole, BoundarySchema


def boundary_key_for_node(node_id: str, container_path: tuple[Any, ...] = ()) -> str:
    """Return a stable boundary key for a node/container path pair."""

    if not container_path:
        return node_id
    path_text = ".".join(repr(part) for part in container_path)
    return f"{node_id}@{path_text}"


def classify_boundary_role(
    *,
    node_id: str,
    target_node_id: str,
    boundary_kind: str,
    direct_target_parent_ids: set[str],
    is_source: bool,
    dtype: str | None,
    shape: tuple[int, ...] | None,
    spatial_shapes: set[tuple[int, ...]],
) -> BoundaryRole:
    """Classify a crossing dependency into a public boundary role."""

    dtype_text = dtype or ""
    if is_source:
        return "passthrough"
    if node_id == target_node_id and boundary_kind == "after":
        return "primary"
    if boundary_kind == "before" and node_id in direct_target_parent_ids:
        return "primary"
    if "int" in dtype_text or "bool" in dtype_text:
        return "index"
    if shape is not None and len(shape) <= 1 and ("int" in dtype_text or dtype_text == ""):
        return "shape_value"
    if shape is not None and len(shape) >= 4 and len(spatial_shapes) > 1:
        return "multi_scale_feature"
    return "skip"


def make_boundary_schema(
    key: str,
    *,
    node: Any,
    role: BoundaryRole,
    device_policy: str,
) -> BoundarySchema:
    """Create a public boundary schema from a split graph node."""

    return BoundarySchema(
        value_id=key,
        container_path=node.output_container_path,
        role=role,
        shape=node.symbolic_output_shape,
        dtype=node.dtype,
        device=None,
        requires_grad=node.requires_grad,
        alias_group=None,
        source_kind="boundary",
        label=node.label,
        backend=node.backend,
        module_path=node.module_path,
        op_type=node.op_type,
        output_index=getattr(node.op, "multi_output_index", None),
        device_policy=device_policy,
    )


__all__ = [
    "boundary_key_for_node",
    "classify_boundary_role",
    "make_boundary_schema",
]
