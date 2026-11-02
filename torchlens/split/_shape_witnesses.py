"""Pure finite-witness axis evidence shared by batch shape compilation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import SplitTraceGraph


def _witness_scaled_axes(
    node_id: str,
    traced_shape: tuple[int, ...],
    traced_batch_size: int,
    witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
) -> set[int]:
    """Return axes observed to be a fixed multiple of batch at every sample."""

    if not witnesses:
        return set()
    witnessed_shapes: list[tuple[int, tuple[int, ...]]] = []
    for batch_size, shapes in witnesses.items():
        shape = shapes.get(node_id)
        if shape is None or len(shape) != len(traced_shape):
            return set()
        witnessed_shapes.append((batch_size, shape))
    return {
        axis
        for axis, traced_dim in enumerate(traced_shape)
        if traced_dim > 0
        and traced_dim % traced_batch_size == 0
        and all(
            shape[axis] * traced_batch_size == traced_dim * batch_size
            for batch_size, shape in witnessed_shapes
        )
    }


def _witness_axis_diagnostics(
    graph: SplitTraceGraph,
    traced_batch_size: int,
    witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Describe finite axis candidates accepted or eliminated by witnesses."""

    if not witnesses:
        return {}
    diagnostics: dict[str, dict[str, tuple[int, ...]]] = {}
    for node in graph.nodes:
        if node.output_shape is None:
            continue
        candidates = tuple(
            axis
            for axis, dim in enumerate(node.output_shape)
            if traced_batch_size > 0 and dim > 0 and dim % traced_batch_size == 0
        )
        if not candidates:
            continue
        accepted = tuple(
            sorted(
                _witness_scaled_axes(
                    node.canonical_id,
                    node.output_shape,
                    traced_batch_size,
                    witnesses,
                )
            )
        )
        diagnostics[node.canonical_id] = {
            "candidate_axes": candidates,
            "accepted_axes": accepted,
            "eliminated_axes": tuple(axis for axis in candidates if axis not in accepted),
        }
    return diagnostics
