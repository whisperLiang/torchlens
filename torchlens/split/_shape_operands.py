"""Lower witnessed split extents at their exact call-argument locations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

from .graph import ReplayValueRef, SplitTraceGraph, SplitTraceNode

if TYPE_CHECKING:
    from .shape_program import DimExpr, ShapeRecipe, TensorShapeIR


@dataclass(frozen=True)
class _SplitOperand:
    """An exact native operand location and its captured extent descriptor."""

    receiver: str
    axis: int
    slot: int | str
    captured: tuple[int, ...]
    scalar: bool


@dataclass(frozen=True)
class _SplitEvidence:
    """One batch symbol and its independent captured/witnessed shape observations."""

    symbol: str
    traced_batch: int
    shapes: Mapping[int, Mapping[str, tuple[int, ...] | None]]


def _native_split(node: SplitTraceNode) -> bool:
    """Recognize audited native identities without authorizing custom name lookalikes."""
    key = getattr(node.args_template, "func_id", None)
    return getattr(key, "qualname", "") in {"split", "split_with_sizes"} and (
        getattr(key, "namespace", None) in {"torch", "torch.Tensor"}
        or getattr(key, "import_path", None)
        in {
            "torch._C._VariableFunctionsClass:split",
            "torch._C._VariableFunctionsClass:split_with_sizes",
        }
    )


def _split_operand(node: SplitTraceNode) -> _SplitOperand | None:
    """Parse scalar/vector extents separately from the receiver and axis arguments."""
    from .shape_program import _captured_call_parts, _literal_component

    if not _native_split(node):
        return None
    args, kwargs = _captured_call_parts(node)
    receiver = args[0] if args else kwargs.get("tensor", kwargs.get("input"))
    axis = _literal_component(args[2] if len(args) > 2 else kwargs.get("dim", 0))
    if (
        not isinstance(receiver, ReplayValueRef)
        or not isinstance(axis, Integral)
        or isinstance(axis, bool)
    ):
        return None
    slot: int | str = (
        1
        if len(args) > 1
        else next(
            (k for k in ("split_size", "split_size_or_sections", "split_sizes") if k in kwargs), ""
        )
    )
    sizes = _literal_component(args[slot] if isinstance(slot, int) else kwargs.get(slot))
    scalar = isinstance(sizes, Integral) and not isinstance(sizes, bool)
    captured = (sizes,) if scalar else sizes
    if not isinstance(captured, (tuple, list)) or not captured:
        return None
    values = tuple(_literal_component(item) for item in captured)
    if not all(isinstance(x, Integral) and not isinstance(x, bool) and int(x) > 0 for x in values):
        return None
    return _SplitOperand(receiver.value_id, int(axis), slot, tuple(int(x) for x in values), scalar)


def compile_split_operands(
    graph: SplitTraceGraph,
    shapes: Mapping[str, TensorShapeIR],
    symbol: str,
    traced_batch: int,
    witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
) -> dict[str, tuple[ShapeRecipe, ...]]:
    """Prove split extents from complete call outputs and independent shape witnesses."""
    if not witnesses:
        return {}
    evidence = _SplitEvidence(
        symbol,
        traced_batch,
        {traced_batch: {n.canonical_id: n.output_shape for n in graph.nodes}, **witnesses},
    )
    result: dict[str, tuple[ShapeRecipe, ...]] = {}
    for call in graph.replay_calls:
        node = graph.node_by_id[call.output_node_ids[0]]
        operand = _split_operand(node)
        parent = shapes.get(operand.receiver) if operand is not None else None
        if operand is None or parent is None:
            continue
        recipe = _prove_split_operand(
            node,
            operand,
            parent,
            call.output_node_ids,
            evidence,
        )
        if recipe is not None:
            result.update((output, (recipe,)) for output in call.output_node_ids)
    return result


def _prove_split_operand(
    node: SplitTraceNode,
    operand: _SplitOperand,
    parent: TensorShapeIR,
    outputs: tuple[str, ...],
    evidence: _SplitEvidence,
) -> ShapeRecipe | None:
    """Require symbolic receiver provenance and complete witnessed partitioning."""
    from .shape_program import DimExpr, ShapeRecipe

    axis = operand.axis % len(parent.dims)
    extent = parent.dims[axis]
    if not extent.contains(evidence.symbol):
        return None
    captured, scalar = operand.captured, operand.scalar
    base = extent.evaluate({evidence.symbol: evidence.traced_batch})
    complete = (
        base == captured[0] * len(outputs)
        if scalar
        else len(captured) == len(outputs) and sum(captured) == base
    )
    if base <= 0 or not complete:
        return None
    dims = tuple(
        DimExpr(
            "floordiv", args=(DimExpr("mul", args=(extent, DimExpr.const(x))), DimExpr.const(base))
        ).simplify()
        for x in captured
    )
    if not _split_shapes_match(evidence, operand, outputs, axis, dims):
        return None
    return ShapeRecipe(
        node_id=node.canonical_id,
        recipe_id=f"split:{node.canonical_id}",
        captured=captured,
        dims=dims,
        argument=operand.slot,
    )


def _split_shapes_match(
    evidence: _SplitEvidence,
    operand: _SplitOperand,
    outputs: tuple[str, ...],
    axis: int,
    dims: tuple[DimExpr, ...],
) -> bool:
    """Require complete partitioning and the same operand rule at every observed batch."""
    for batch, shapes in evidence.shapes.items():
        parent = shapes.get(operand.receiver)
        children = [shapes.get(output) for output in outputs]
        if parent is None or any(shape is None or len(shape) != len(parent) for shape in children):
            return False
        extents = [shape[axis] for shape in children if shape is not None]
        expected = [dim.evaluate({evidence.symbol: batch}) for dim in dims]
        if operand.scalar:
            expected *= len(outputs)
        if sum(extents) != parent[axis] or extents != expected:
            return False
    return True
