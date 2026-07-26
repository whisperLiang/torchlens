"""Backend-neutral symbolic batch constraint compilation and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import prod
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

from ..intervention.types import CapturedArgTemplate, LiteralValue
from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError
from .graph import iter_replay_value_refs

if TYPE_CHECKING:
    from .adapters.base import SplitBackendAdapter
    from .graph import SplitTraceGraph, SplitTraceNode
    from .ir import SplitRequest


DimOp = Literal["const", "symbol", "add", "mul", "floordiv", "ceildiv", "min", "max"]
ShapeSemantic = Literal[
    "reshape",
    "repeat",
    "permute",
    "squeeze",
    "unsqueeze",
    "expand",
    "concat",
    "stack",
    "split",
    "unbind",
    "slice",
    "reduction",
    "matmul",
    "factory",
]

_SHAPE_RULE_VERSION = 3
_WITNESS_DISAMBIGUATED_SEMANTICS = frozenset(
    {
        "reshape",
        "squeeze",
        "unsqueeze",
        "concat",
        "stack",
        "split",
        "unbind",
        "slice",
        "reduction",
        "matmul",
        "factory",
    }
)
_TORCH_SHAPE_SEMANTICS: dict[tuple[str, str], ShapeSemantic] = {
    ("torch.Tensor", "view"): "reshape",
    ("torch.Tensor", "reshape"): "reshape",
    ("torch.Tensor", "flatten"): "reshape",
    ("torch", "reshape"): "reshape",
    ("torch", "flatten"): "reshape",
    ("torch.Tensor", "repeat"): "repeat",
    ("torch", "tile"): "repeat",
    ("torch.Tensor", "permute"): "permute",
    ("torch.Tensor", "transpose"): "permute",
    ("torch", "transpose"): "permute",
    ("torch.Tensor", "squeeze"): "squeeze",
    ("torch", "squeeze"): "squeeze",
    ("torch.Tensor", "unsqueeze"): "unsqueeze",
    ("torch", "unsqueeze"): "unsqueeze",
    ("torch.Tensor", "expand"): "expand",
    ("torch", "broadcast_to"): "expand",
    ("torch", "cat"): "concat",
    ("torch", "concat"): "concat",
    ("torch", "stack"): "stack",
    ("torch", "split"): "split",
    ("torch", "chunk"): "split",
    ("torch.Tensor", "split"): "split",
    ("torch.Tensor", "chunk"): "split",
    ("torch", "unbind"): "unbind",
    ("torch.Tensor", "unbind"): "unbind",
    ("torch.Tensor", "__getitem__"): "slice",
    ("torch.Tensor", "select"): "slice",
    ("torch.Tensor", "narrow"): "slice",
    ("torch", "index_select"): "slice",
    ("torch.Tensor", "index_select"): "slice",
    ("torch", "sum"): "reduction",
    ("torch.Tensor", "sum"): "reduction",
    ("torch", "mean"): "reduction",
    ("torch.Tensor", "mean"): "reduction",
    ("torch", "amax"): "reduction",
    ("torch.Tensor", "amax"): "reduction",
    ("torch", "amin"): "reduction",
    ("torch.Tensor", "amin"): "reduction",
    ("torch", "prod"): "reduction",
    ("torch.Tensor", "prod"): "reduction",
    ("torch", "matmul"): "matmul",
    ("torch.Tensor", "matmul"): "matmul",
    ("torch", "zeros"): "factory",
    ("torch", "ones"): "factory",
    ("torch", "empty"): "factory",
    ("torch", "full"): "factory",
    ("torch", "new_zeros"): "factory",
    ("torch", "new_ones"): "factory",
    ("torch.Tensor", "new_tensor"): "factory",
    ("torch.Tensor", "new_zeros"): "factory",
    ("torch.Tensor", "new_ones"): "factory",
}


@dataclass(frozen=True)
class DimExpr:
    """One evaluable symbolic dimension expression."""

    op: DimOp
    value: int | str | None = None
    args: tuple["DimExpr", ...] = ()

    @classmethod
    def const(cls, value: int) -> "DimExpr":
        """Return a constant dimension expression."""

        return cls("const", int(value))

    @classmethod
    def symbol(cls, name: str) -> "DimExpr":
        """Return a symbolic dimension expression."""

        return cls("symbol", str(name))

    def evaluate(self, symbols: Mapping[str, int]) -> int:
        """Evaluate this expression under ``symbols``."""

        if self.op == "const":
            if not isinstance(self.value, int):
                raise ValueError("Constant dimension expression has no integer value.")
            return self.value
        if self.op == "symbol":
            try:
                return int(symbols[str(self.value)])
            except KeyError as exc:
                raise ValueError(f"Missing shape symbol {self.value!r}.") from exc
        values = tuple(arg.evaluate(symbols) for arg in self.args)
        if self.op == "add":
            return sum(values)
        if self.op == "mul":
            product = 1
            for value in values:
                product *= value
            return product
        if self.op == "floordiv":
            return values[0] // values[1]
        if self.op == "ceildiv":
            return -(-values[0] // values[1])
        if self.op == "min":
            return min(values)
        if self.op == "max":
            return max(values)
        raise ValueError(f"Unsupported dimension expression op {self.op!r}.")

    def as_dict(self) -> dict[str, Any]:
        """Return portable expression metadata."""

        return {
            "op": self.op,
            "value": self.value,
            "args": [item.as_dict() for item in self.args],
        }

    def contains(self, symbol: str) -> bool:
        """Return whether this expression references ``symbol``."""

        return (self.op == "symbol" and self.value == symbol) or any(
            item.contains(symbol) for item in self.args
        )


@dataclass(frozen=True)
class TensorShapeIR:
    """Symbolic shape assigned to one graph value."""

    value_id: str
    dims: tuple[DimExpr, ...]

    def evaluate(self, binding: "ShapeBinding") -> tuple[int, ...]:
        """Evaluate this shape under a runtime binding."""

        return tuple(dim.evaluate(binding.symbols) for dim in self.dims)


@dataclass(frozen=True)
class ShapeConstraintIR:
    """Structured relation checked by the dynamic-batch runtime."""

    constraint_id: str
    kind: Literal["equal", "broadcast", "product", "range", "axis"]
    value_ids: tuple[str, ...]
    lhs: DimExpr | None = None
    rhs: DimExpr | None = None
    description: str | None = None


@dataclass(frozen=True)
class ShapeRecipe:
    """Exact captured shape descriptor and its symbolic replacement."""

    node_id: str
    recipe_id: str
    captured: tuple[int, ...]
    dims: tuple[DimExpr, ...]
    target: str = "shape_literal"

    def evaluate(self, binding: "ShapeBinding") -> tuple[int, ...]:
        """Evaluate the replacement descriptor."""

        return tuple(dim.evaluate(binding.symbols) for dim in self.dims)


@dataclass(frozen=True)
class ShapeBinding:
    """Runtime symbol values for one prefix/suffix execution."""

    symbols: Mapping[str, int]
    input_shapes: Mapping[str, tuple[int, ...]]

    @property
    def batch_size(self) -> int:
        """Return the sole batch symbol value."""

        if len(self.symbols) != 1:
            raise ValueError("ShapeBinding.batch_size requires exactly one symbol.")
        return int(next(iter(self.symbols.values())))


@dataclass(frozen=True)
class InputLeaf:
    """One tensor leaf and its canonical JSON Pointer path."""

    path: str
    value: Any


@dataclass(frozen=True)
class ShapeProgram:
    """Compiled symbolic batch program shared by every backend adapter."""

    batch_symbol: str
    dynamic_batch: tuple[int, int]
    traced_batch_size: int
    input_paths: tuple[str, ...]
    input_node_ids: tuple[str, ...]
    input_batch_axes: Mapping[str, int]
    traced_input_shapes: Mapping[str, tuple[int, ...]]
    value_shapes: Mapping[str, TensorShapeIR]
    constraints: tuple[ShapeConstraintIR, ...]
    recipes: Mapping[str, tuple[ShapeRecipe, ...]]
    unresolved: Mapping[str, str]
    fingerprint: str
    inference_mode: Literal["explicit", "conservative_auto"]
    witness_batch_sizes: tuple[int, ...] = ()
    proof_sources: Mapping[str, str] = field(default_factory=dict)
    witness_axis_diagnostics: Mapping[str, Mapping[str, tuple[int, ...]]] = field(
        default_factory=dict
    )

    def bind_flat_values(
        self,
        values: Sequence[Any],
        *,
        shape_of: Any,
        backend: str,
        split_point: str,
    ) -> ShapeBinding:
        """Bind symbols from tensor leaves in captured graph-input order."""

        if len(values) != len(self.input_paths):
            raise SplitBoundaryError(
                f"Runtime input leaf count {len(values)} does not match shape program "
                f"input count {len(self.input_paths)}."
            )
        by_path = dict(zip(self.input_paths, values))
        missing = tuple(path for path in self.input_batch_axes if path not in by_path)
        if missing:
            raise SplitBoundaryError(
                f"Dynamic-batch input paths are missing at runtime: {missing!r}.",
                context=SplitErrorContext(
                    backend=backend,
                    split_point=split_point,
                    reason="missing batch input path",
                ),
            )
        batch_values: dict[str, int] = {}
        input_shapes: dict[str, tuple[int, ...]] = {}
        for path, axis in self.input_batch_axes.items():
            shape = shape_of(by_path[path])
            if shape is None:
                raise SplitBoundaryError(f"Dynamic-batch input {path!r} is not tensor-like.")
            normalized_axis = axis if axis >= 0 else len(shape) + axis
            if not 0 <= normalized_axis < len(shape):
                raise SplitBoundaryError(
                    f"Batch axis {axis} is invalid for runtime input {path!r} shape {shape}."
                )
            batch_values[path] = int(shape[normalized_axis])
            input_shapes[path] = tuple(shape)
            traced_shape = self.traced_input_shapes[path]
            if len(shape) != len(traced_shape):
                raise SplitBoundaryError(
                    f"Runtime input {path!r} rank changed from {len(traced_shape)} to {len(shape)}."
                )
            for index, (runtime_dim, traced_dim) in enumerate(zip(shape, traced_shape)):
                if index != normalized_axis and int(runtime_dim) != int(traced_dim):
                    raise SplitBoundaryError(
                        f"Runtime input {path!r} non-batch dimension {index} changed from "
                        f"{traced_dim} to {runtime_dim}."
                    )
        unique = set(batch_values.values())
        if len(unique) != 1:
            raise SplitBoundaryError(
                f"Dynamic-batch inputs disagree: {batch_values!r}.",
                context=SplitErrorContext(
                    backend=backend,
                    split_point=split_point,
                    reason="inconsistent runtime batch",
                ),
            )
        batch_size = next(iter(unique))
        low, high = self.dynamic_batch
        if not low <= batch_size <= high:
            raise SplitBoundaryError(
                f"Runtime batch {batch_size} is outside {self.dynamic_batch}.",
                context=SplitErrorContext(
                    backend=backend,
                    split_point=split_point,
                    reason="dynamic batch outside allowed range",
                ),
            )
        return ShapeBinding(
            symbols={self.batch_symbol: batch_size},
            input_shapes=input_shapes,
        )

    def binding_from_batch(self, batch_size: int) -> ShapeBinding:
        """Create a suffix binding from trusted boundary metadata."""

        low, high = self.dynamic_batch
        if not low <= int(batch_size) <= high:
            raise SplitBoundaryError(
                f"Boundary batch {batch_size} is outside {self.dynamic_batch}."
            )
        return ShapeBinding(symbols={self.batch_symbol: int(batch_size)}, input_shapes={})

    def rewrite(self, node_id: str, value: Any, binding: ShapeBinding) -> Any:
        """Apply an exact lowered recipe to a captured literal tree."""

        node_recipes = self.recipes.get(node_id, ())
        if not node_recipes:
            return value
        return _rewrite_recipe_tree(value, node_recipes, binding)

    def value_shape(self, node_id: str, binding: ShapeBinding) -> tuple[int, ...] | None:
        """Evaluate one graph value shape for the current runtime symbols."""

        shape = self.value_shapes.get(node_id)
        return None if shape is None else shape.evaluate(binding)


def flatten_input_leaves(
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    *,
    adapter: "SplitBackendAdapter",
) -> tuple[InputLeaf, ...]:
    """Flatten tensor inputs with stable JSON Pointer paths."""

    leaves: list[InputLeaf] = []
    for index, value in enumerate(inputs):
        _walk_input(value, f"/args/{index}", adapter, leaves)
    for key in sorted((input_kwargs or {}), key=repr):
        _walk_input(
            (input_kwargs or {})[key], f"/kwargs/{_escape_pointer(str(key))}", adapter, leaves
        )
    return tuple(leaves)


def compile_shape_program(
    graph: "SplitTraceGraph",
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    request: "SplitRequest",
    *,
    adapter: "SplitBackendAdapter",
    shape_witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]] | None = None,
) -> ShapeProgram | None:
    """Compile a backend-neutral dynamic-batch shape program."""

    if request.dynamic_batch is None:
        return None
    leaves = flatten_input_leaves(inputs, input_kwargs, adapter=adapter)
    if len(leaves) != len(graph.input_node_ids):
        raise SplitUnsupportedError(
            "Cannot align runtime input paths with captured graph inputs: "
            f"paths={len(leaves)}, graph_inputs={len(graph.input_node_ids)}.",
            context=SplitErrorContext(
                backend=graph.backend,
                split_point=request.boundary,
                reason="input path alignment failed",
            ),
        )
    input_by_path = {leaf.path: leaf for leaf in leaves}
    explicit = dict(request.features.batch_axes)
    if explicit:
        unknown = tuple(path for path in explicit if path not in input_by_path)
        if unknown:
            raise SplitUnsupportedError(
                f"Unknown dynamic-batch input paths: {unknown!r}; available paths are "
                f"{tuple(input_by_path)!r}."
            )
        axes = {
            path: _normalize_axis(int(axis), adapter.shape(input_by_path[path].value), path)
            for path, axis in explicit.items()
        }
        declared_batch_values = {
            tuple(adapter.shape(input_by_path[path].value) or ())[axis]
            for path, axis in axes.items()
        }
        if len(declared_batch_values) != 1:
            raise SplitUnsupportedError(
                "Declared batch inputs disagree during capture: "
                f"{ {path: adapter.shape(input_by_path[path].value) for path in axes}!r}."
            )
        traced_batch = int(next(iter(declared_batch_values)))
        undeclared_candidates: dict[str, tuple[int, ...]] = {}
        for path, leaf in input_by_path.items():
            if path in axes:
                continue
            shape = tuple(adapter.shape(leaf.value) or ())
            candidates = tuple(index for index, dim in enumerate(shape) if dim == traced_batch)
            if candidates:
                undeclared_candidates[path] = candidates
        if undeclared_candidates:
            raise SplitUnsupportedError(
                "Explicit dynamic-batch inputs are incomplete: undeclared tensor paths contain "
                f"the traced batch dimension {traced_batch}: {undeclared_candidates!r}. "
                "Declare these paths in SplitFeatures.batch_axes."
            )
        inference_mode: Literal["explicit", "conservative_auto"] = "explicit"
    else:
        axes = _infer_batch_axes(leaves, graph, adapter)
        inference_mode = "conservative_auto"

    traced_shapes = {path: tuple(adapter.shape(input_by_path[path].value) or ()) for path in axes}
    traced_batch_values = {shape[axes[path]] for path, shape in traced_shapes.items()}
    if len(traced_batch_values) != 1:
        raise SplitUnsupportedError(
            f"Declared batch inputs disagree during capture: {traced_shapes!r}."
        )
    traced_batch_size = int(next(iter(traced_batch_values)))
    node_axis: dict[str, int] = {}
    input_paths = tuple(leaf.path for leaf in leaves)
    for node_id, path in zip(graph.input_node_ids, input_paths):
        if path in axes:
            node_axis[node_id] = axes[path]
    value_shapes = _propagate_shapes(
        graph,
        node_axis,
        request.batch_symbol,
        traced_batch_size=traced_batch_size,
        shape_witnesses=shape_witnesses or {},
    )
    constraints = tuple(
        ShapeConstraintIR(
            constraint_id=f"axis:{node_id}",
            kind="axis",
            value_ids=(node_id,),
            lhs=shape.dims[
                next(i for i, dim in enumerate(shape.dims) if dim.contains(request.batch_symbol))
            ],
            rhs=DimExpr.symbol(request.batch_symbol),
            description="value carries the declared batch symbol",
        )
        for node_id, shape in value_shapes.items()
        if any(dim.contains(request.batch_symbol) for dim in shape.dims)
    )
    recipes = _compile_recipes(graph, value_shapes, request.batch_symbol)
    unresolved = _unresolved_dynamic_nodes(
        graph,
        value_shapes,
        request.batch_symbol,
        traced_batch_size=traced_batch_size,
        shape_witnesses=shape_witnesses or {},
        dynamic_batch=request.dynamic_batch,
    )
    payload = {
        "rule_version": _SHAPE_RULE_VERSION,
        "batch_symbol": request.batch_symbol,
        "range": request.dynamic_batch,
        "axes": sorted(axes.items()),
        "values": {
            key: [dim.as_dict() for dim in value.dims]
            for key, value in sorted(value_shapes.items())
        },
        "recipes": {
            key: [recipe.captured for recipe in value] for key, value in sorted(recipes.items())
        },
        "witness_batch_sizes": sorted(shape_witnesses or {}),
    }
    fingerprint = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    witness_axis_diagnostics = _witness_axis_diagnostics(
        graph,
        traced_batch_size,
        shape_witnesses or {},
    )
    return ShapeProgram(
        batch_symbol=request.batch_symbol,
        dynamic_batch=request.dynamic_batch,
        traced_batch_size=traced_batch_size,
        input_paths=input_paths,
        input_node_ids=graph.input_node_ids,
        input_batch_axes=axes,
        traced_input_shapes=traced_shapes,
        value_shapes=value_shapes,
        constraints=constraints,
        recipes=recipes,
        unresolved=unresolved,
        fingerprint=fingerprint,
        inference_mode=inference_mode,
        witness_batch_sizes=tuple(sorted(shape_witnesses or {})),
        proof_sources=_shape_proof_sources(
            graph,
            value_shapes,
            request.batch_symbol,
            witness_axis_diagnostics,
        ),
        witness_axis_diagnostics=witness_axis_diagnostics,
    )


def _walk_input(
    value: Any,
    path: str,
    adapter: "SplitBackendAdapter",
    leaves: list[InputLeaf],
) -> None:
    """Walk one public input tree."""

    if adapter.is_tensor(value):
        leaves.append(InputLeaf(path, value))
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=repr):
            _walk_input(value[key], f"{path}/{_escape_pointer(str(key))}", adapter, leaves)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_input(item, f"{path}/{index}", adapter, leaves)
        return
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        for name in fields:
            _walk_input(getattr(value, name), f"{path}/{_escape_pointer(name)}", adapter, leaves)


def _escape_pointer(value: str) -> str:
    """Escape one RFC 6901 JSON Pointer component."""

    return value.replace("~", "~0").replace("/", "~1")


def _normalize_axis(axis: int, shape: tuple[int, ...] | None, path: str) -> int:
    """Normalize one declared input axis."""

    if shape is None:
        raise SplitUnsupportedError(f"Batch input {path!r} is not tensor-like.")
    normalized = axis if axis >= 0 else len(shape) + axis
    if not 0 <= normalized < len(shape):
        raise SplitUnsupportedError(f"Batch axis {axis} is invalid for {path!r} shape {shape}.")
    return normalized


def _infer_batch_axes(
    leaves: Sequence[InputLeaf],
    graph: "SplitTraceGraph",
    adapter: "SplitBackendAdapter",
) -> dict[str, int]:
    """Infer only unambiguous top-level tensor batch inputs."""

    traced_batch = graph.traced_batch_size
    candidates = {
        leaf.path: 0
        for leaf in leaves
        if leaf.path.count("/") == 2
        and (shape := adapter.shape(leaf.value)) is not None
        and shape
        and traced_batch is not None
        and int(shape[0]) == traced_batch
    }
    if not candidates:
        available = {
            leaf.path: adapter.shape(leaf.value)
            for leaf in leaves
            if adapter.shape(leaf.value) is not None
        }
        raise SplitUnsupportedError(
            "Dynamic batch could not be inferred conservatively. Declare "
            f"SplitFeatures.batch_axes using one of {available!r}."
        )
    return candidates


def _propagate_shapes(
    graph: "SplitTraceGraph",
    input_axes: Mapping[str, int],
    batch_symbol: str,
    *,
    traced_batch_size: int,
    shape_witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
) -> dict[str, TensorShapeIR]:
    """Propagate batch provenance through concrete graph shapes."""

    concrete = {node.canonical_id: node.output_shape for node in graph.nodes}
    dynamic_axes: dict[str, set[int]] = {node_id: {axis} for node_id, axis in input_axes.items()}
    for _iteration in range(max(1, len(graph.nodes) * 2)):
        changed = False
        for node in graph.nodes:
            output_shape = concrete.get(node.canonical_id)
            if output_shape is None:
                continue
            parent_nodes: list[SplitTraceNode] = []
            for parent_label in node.parents:
                parent_node = graph.node_for_label(parent_label)
                if parent_node is not None:
                    parent_nodes.extend(_shape_bearing_parents(graph, parent_node, concrete))
            op = node.op_type.lower()
            semantic = shape_semantic_for_node(node)
            inferred = set(dynamic_axes.get(node.canonical_id, set()))
            inferred.update(
                _witness_scaled_axes(
                    node.canonical_id,
                    output_shape,
                    traced_batch_size,
                    shape_witnesses,
                )
            )
            permutation = _node_permutation(node, len(output_shape))
            for parent in parent_nodes:
                parent_shape = concrete.get(parent.canonical_id)
                for axis in dynamic_axes.get(parent.canonical_id, set()):
                    if parent_shape is None:
                        continue
                    if permutation is not None and axis in permutation:
                        inferred.add(permutation.index(axis))
                        continue
                    if semantic == "repeat":
                        aligned = axis + len(output_shape) - len(parent_shape)
                        if 0 <= aligned < len(output_shape):
                            inferred.add(aligned)
                        continue
                    if semantic in _WITNESS_DISAMBIGUATED_SEMANTICS:
                        # Equality or divisibility at one traced batch is not a proof
                        # that a shape-changing output axis carries batch provenance.
                        # Finite semantic candidates are resolved by declared-range witnesses.
                        continue
                    if (
                        axis == 0
                        and parent_shape
                        and output_shape
                        and output_shape[0] == parent_shape[0]
                    ):
                        inferred.add(0)
                        continue
                    aligned = axis + len(output_shape) - len(parent_shape)
                    if 0 <= aligned < len(output_shape):
                        if output_shape[aligned] == parent_shape[axis] or _preserves_batch(op):
                            inferred.add(aligned)
            if (
                _is_reshape_like(op)
                and parent_nodes
                and any(dynamic_axes.get(parent.canonical_id) for parent in parent_nodes)
            ):
                first_parent = parent_nodes[0]
                first_parent_shape = concrete.get(first_parent.canonical_id)
                witnessed_axes = _witness_scaled_axes(
                    node.canonical_id,
                    output_shape,
                    traced_batch_size,
                    shape_witnesses,
                )
                if semantic == "reshape":
                    inferred.update(witnessed_axes)
                elif (
                    first_parent_shape
                    and 0 in dynamic_axes.get(first_parent.canonical_id, set())
                    and output_shape
                    and output_shape[0] == first_parent_shape[0]
                ):
                    inferred.add(0)
            if inferred != dynamic_axes.get(node.canonical_id, set()):
                dynamic_axes[node.canonical_id] = inferred
                changed = True
            if _is_elementwise(op) and inferred:
                for parent in parent_nodes:
                    parent_shape = concrete.get(parent.canonical_id)
                    if parent_shape is None or not parent_shape:
                        continue
                    parent_op = parent.op_type.lower()
                    if _is_parameter_lineage(graph, parent) and not any(
                        token in parent_op for token in ("expand", "broadcast")
                    ):
                        continue
                    if not parent.parents and not parent.is_input:
                        # Parentless backend nodes are parameter/constant
                        # transforms.  Their singleton leading dimensions are
                        # broadcast ABI, not model batch provenance.
                        continue
                    if _is_parameter_only_reshape(
                        graph,
                        parent,
                        concrete,
                        dynamic_axes,
                    ):
                        continue
                    parent_axes = set(dynamic_axes.get(parent.canonical_id, set()))
                    for axis in inferred:
                        aligned = axis - len(output_shape) + len(parent_shape)
                        if (
                            0 <= aligned < len(parent_shape)
                            and parent_shape[aligned] == output_shape[axis]
                            and len(parent_shape) > 1
                        ):
                            parent_axes.add(aligned)
                    if parent_axes != dynamic_axes.get(parent.canonical_id, set()):
                        dynamic_axes[parent.canonical_id] = parent_axes
                        changed = True
        if not changed:
            break

    result: dict[str, TensorShapeIR] = {}
    for node in graph.nodes:
        if node.output_shape is None:
            continue
        dims = [DimExpr.const(dim) for dim in node.output_shape]
        for axis in dynamic_axes.get(node.canonical_id, set()):
            if 0 <= axis < len(dims):
                concrete_dim = node.output_shape[axis]
                if traced_batch_size and concrete_dim % traced_batch_size == 0:
                    factor = concrete_dim // traced_batch_size
                    dims[axis] = (
                        DimExpr.symbol(batch_symbol)
                        if factor == 1
                        else DimExpr(
                            "mul",
                            args=(DimExpr.symbol(batch_symbol), DimExpr.const(factor)),
                        )
                    )
        result[node.canonical_id] = TensorShapeIR(node.canonical_id, tuple(dims))
    return _apply_semantic_shape_expressions(
        graph,
        result,
        batch_symbol=batch_symbol,
        traced_batch_size=traced_batch_size,
        shape_witnesses=shape_witnesses,
    )


def _apply_semantic_shape_expressions(
    graph: "SplitTraceGraph",
    value_shapes: Mapping[str, TensorShapeIR],
    *,
    batch_symbol: str,
    traced_batch_size: int,
    shape_witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
) -> dict[str, TensorShapeIR]:
    """Propagate non-proportional expressions using exact operation semantics."""

    resolved = dict(value_shapes)
    for _iteration in range(max(1, len(graph.nodes))):
        changed = False
        for node in graph.nodes:
            candidate = _semantic_shape_candidate(graph, node, resolved)
            if candidate is None:
                candidate = _passthrough_shape_candidate(graph, node, resolved)
            if candidate is None or not _shape_candidate_matches_evidence(
                node.canonical_id,
                candidate,
                node.output_shape,
                batch_symbol=batch_symbol,
                traced_batch_size=traced_batch_size,
                shape_witnesses=shape_witnesses,
            ):
                continue
            if resolved.get(node.canonical_id) != candidate:
                resolved[node.canonical_id] = candidate
                changed = True
        if not changed:
            break
    return resolved


def _semantic_shape_candidate(
    graph: "SplitTraceGraph",
    node: "SplitTraceNode",
    value_shapes: Mapping[str, TensorShapeIR],
) -> TensorShapeIR | None:
    """Return an expression candidate derived from one registered operation rule."""

    semantic = shape_semantic_for_node(node)
    if semantic == "concat":
        return _concat_shape_candidate(node, value_shapes)
    if semantic == "stack":
        return _stack_shape_candidate(node, value_shapes)
    if semantic == "slice":
        return _slice_shape_candidate(node, value_shapes)
    if semantic == "repeat":
        return _repeat_shape_candidate(node, value_shapes)
    if semantic == "permute":
        return _permuted_shape_candidate(graph, node, value_shapes)
    return None


def _captured_call_parts(node: "SplitTraceNode") -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return normalized positional and keyword capture components for one call."""

    template = node.args_template
    if not isinstance(template, CapturedArgTemplate):
        return (), {}
    return template.args, dict(template.kwargs)


def _literal_component(value: Any) -> Any:
    """Unwrap a captured immutable literal component."""

    while isinstance(value, LiteralValue):
        value = value.value
    if isinstance(value, tuple):
        return tuple(_literal_component(item) for item in value)
    if isinstance(value, list):
        return [_literal_component(item) for item in value]
    return value


def _template_value_refs(component: Any) -> tuple[str, ...]:
    """Return canonical replay value references while preserving multiplicity."""

    return tuple(ref.value_id for ref in iter_replay_value_refs(component))


def _captured_int_argument(
    node: "SplitTraceNode",
    position: int,
    keyword: str,
    default: int,
) -> int | None:
    """Return one captured integer argument without resolving tensor references."""

    args, kwargs = _captured_call_parts(node)
    component = kwargs.get(keyword, args[position] if position < len(args) else default)
    value = _literal_component(component)
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return None


def _normalize_dim(axis: int, rank: int) -> int | None:
    """Normalize an operation dimension, returning ``None`` when invalid."""

    normalized = axis if axis >= 0 else rank + axis
    return normalized if 0 <= normalized < rank else None


def _dim_add(*items: DimExpr) -> DimExpr:
    """Build a simplified additive dimension expression."""

    flattened: list[DimExpr] = []
    constant = 0
    for item in items:
        children = item.args if item.op == "add" else (item,)
        for child in children:
            if child.op == "const" and isinstance(child.value, int):
                constant += child.value
            else:
                flattened.append(child)
    if constant or not flattened:
        flattened.append(DimExpr.const(constant))
    return flattened[0] if len(flattened) == 1 else DimExpr("add", args=tuple(flattened))


def _dim_mul(left: DimExpr, factor: int) -> DimExpr:
    """Build a simplified dimension multiplied by an integer factor."""

    if factor == 1:
        return left
    if factor == 0:
        return DimExpr.const(0)
    return DimExpr("mul", args=(left, DimExpr.const(factor)))


def _concat_shape_candidate(
    node: "SplitTraceNode",
    value_shapes: Mapping[str, TensorShapeIR],
) -> TensorShapeIR | None:
    """Derive concat output dimensions by summing ordered input dimensions."""

    args, _kwargs = _captured_call_parts(node)
    if not args:
        return None
    input_shapes = [
        value_shapes[value_id]
        for value_id in _template_value_refs(args[0])
        if value_id in value_shapes
    ]
    if not input_shapes:
        return None
    rank = len(input_shapes[0].dims)
    if any(len(shape.dims) != rank for shape in input_shapes):
        return None
    axis_value = _captured_int_argument(node, 1, "dim", 0)
    axis = None if axis_value is None else _normalize_dim(axis_value, rank)
    if axis is None:
        return None
    dims = list(input_shapes[0].dims)
    dims[axis] = _dim_add(*(shape.dims[axis] for shape in input_shapes))
    return TensorShapeIR(node.canonical_id, tuple(dims))


def _stack_shape_candidate(
    node: "SplitTraceNode",
    value_shapes: Mapping[str, TensorShapeIR],
) -> TensorShapeIR | None:
    """Derive stack output dimensions from its first tensor and stack count."""

    args, _kwargs = _captured_call_parts(node)
    if not args:
        return None
    refs = _template_value_refs(args[0])
    input_shape = next(
        (value_shapes[value_id] for value_id in refs if value_id in value_shapes),
        None,
    )
    if input_shape is None:
        return None
    output_rank = len(input_shape.dims) + 1
    axis_value = _captured_int_argument(node, 1, "dim", 0)
    if axis_value is None:
        return None
    axis = axis_value if axis_value >= 0 else output_rank + axis_value
    if not 0 <= axis < output_rank:
        return None
    dims = list(input_shape.dims)
    dims.insert(axis, DimExpr.const(len(refs)))
    return TensorShapeIR(node.canonical_id, tuple(dims))


def _slice_shape_candidate(
    node: "SplitTraceNode",
    value_shapes: Mapping[str, TensorShapeIR],
) -> TensorShapeIR | None:
    """Derive dimensions for Python basic slicing with positive strides."""

    args, _kwargs = _captured_call_parts(node)
    if len(args) < 2:
        return None
    receiver_refs = _template_value_refs(args[0])
    parent_shape = next(
        (value_shapes[value_id] for value_id in receiver_refs if value_id in value_shapes),
        None,
    )
    if parent_shape is None:
        return None
    index = _literal_component(args[1])
    indexes = index if isinstance(index, tuple) else (index,)
    ellipsis_index = next(
        (position for position, item in enumerate(indexes) if item is Ellipsis),
        None,
    )
    if ellipsis_index is not None:
        consumed = sum(item is not None and item is not Ellipsis for item in indexes)
        fill = max(0, len(parent_shape.dims) - consumed)
        indexes = (
            *indexes[:ellipsis_index],
            *(slice(None) for _ in range(fill)),
            *indexes[ellipsis_index + 1 :],
        )
    dims: list[DimExpr] = []
    parent_axis = 0
    for item in indexes:
        if item is None:
            dims.append(DimExpr.const(1))
            continue
        if parent_axis >= len(parent_shape.dims):
            return None
        parent_dim = parent_shape.dims[parent_axis]
        parent_axis += 1
        if isinstance(item, Integral) and not isinstance(item, bool):
            continue
        if not isinstance(item, slice):
            return None
        step = 1 if item.step is None else item.step
        start = 0 if item.start is None else item.start
        stop = item.stop
        if (
            not isinstance(step, Integral)
            or isinstance(step, bool)
            or int(step) <= 0
            or not isinstance(start, Integral)
            or isinstance(start, bool)
            or int(start) < 0
            or (
                stop is not None
                and (
                    not isinstance(stop, Integral)
                    or isinstance(stop, bool)
                    or int(stop) < 0
                )
            )
        ):
            return None
        bounded = (
            parent_dim
            if stop is None
            else DimExpr("min", args=(parent_dim, DimExpr.const(int(stop))))
        )
        remaining = _dim_add(bounded, DimExpr.const(-int(start)))
        nonnegative = DimExpr("max", args=(remaining, DimExpr.const(0)))
        dims.append(
            nonnegative
            if int(step) == 1
            else DimExpr("ceildiv", args=(nonnegative, DimExpr.const(int(step))))
        )
    dims.extend(parent_shape.dims[parent_axis:])
    return TensorShapeIR(node.canonical_id, tuple(dims))


def _repeat_shape_candidate(
    node: "SplitTraceNode",
    value_shapes: Mapping[str, TensorShapeIR],
) -> TensorShapeIR | None:
    """Derive repeat/tile dimensions directly from captured repeat factors."""

    args, _kwargs = _captured_call_parts(node)
    if len(args) < 2:
        return None
    receiver_refs = _template_value_refs(args[0])
    parent_shape = next(
        (value_shapes[value_id] for value_id in receiver_refs if value_id in value_shapes),
        None,
    )
    if parent_shape is None:
        return None
    raw_repeats: Any = args[1:] if len(args) > 2 else _literal_component(args[1])
    repeats = raw_repeats if isinstance(raw_repeats, (tuple, list)) else (raw_repeats,)
    repeat_values = tuple(_literal_component(item) for item in repeats)
    if not repeat_values or not all(
        isinstance(item, Integral) and not isinstance(item, bool) for item in repeat_values
    ):
        return None
    factors = tuple(int(item) for item in repeat_values)
    padded_dims = (
        (DimExpr.const(1),) * max(0, len(factors) - len(parent_shape.dims))
        + parent_shape.dims
    )
    padded_factors = (1,) * max(0, len(padded_dims) - len(factors)) + factors
    return TensorShapeIR(
        node.canonical_id,
        tuple(_dim_mul(dim, factor) for dim, factor in zip(padded_dims, padded_factors)),
    )


def _permuted_shape_candidate(
    graph: "SplitTraceGraph",
    node: "SplitTraceNode",
    value_shapes: Mapping[str, TensorShapeIR],
) -> TensorShapeIR | None:
    """Derive permute/transpose dimensions from the captured axis permutation."""

    parent = next(
        (
            candidate
            for label in node.parents
            if (candidate := graph.node_for_label(label)) is not None
        ),
        None,
    )
    if parent is None or parent.canonical_id not in value_shapes:
        return None
    parent_shape = value_shapes[parent.canonical_id]
    permutation = _node_permutation(node, len(parent_shape.dims))
    if permutation is None:
        return None
    return TensorShapeIR(node.canonical_id, tuple(parent_shape.dims[axis] for axis in permutation))


def _passthrough_shape_candidate(
    graph: "SplitTraceGraph",
    node: "SplitTraceNode",
    value_shapes: Mapping[str, TensorShapeIR],
) -> TensorShapeIR | None:
    """Propagate arbitrary expressions through aligned shape-preserving operations."""

    if node.output_shape is None or not _is_elementwise(node.op_type.lower()):
        return None
    dims = [DimExpr.const(dim) for dim in node.output_shape]
    found_dynamic = False
    for parent_label in node.parents:
        parent = graph.node_for_label(parent_label)
        if parent is None or parent.output_shape is None:
            continue
        parent_shape = value_shapes.get(parent.canonical_id)
        if parent_shape is None:
            continue
        offset = len(node.output_shape) - len(parent.output_shape)
        for parent_axis, expression in enumerate(parent_shape.dims):
            output_axis = parent_axis + offset
            if (
                _symbols_in_expr(expression)
                and 0 <= output_axis < len(dims)
                and node.output_shape[output_axis] == parent.output_shape[parent_axis]
            ):
                dims[output_axis] = expression
                found_dynamic = True
    return TensorShapeIR(node.canonical_id, tuple(dims)) if found_dynamic else None


def _symbols_in_expr(expression: DimExpr) -> set[str]:
    """Return symbol names referenced by one dimension expression."""

    symbols = {str(expression.value)} if expression.op == "symbol" else set()
    for child in expression.args:
        symbols.update(_symbols_in_expr(child))
    return symbols


def _shape_candidate_matches_evidence(
    node_id: str,
    candidate: TensorShapeIR,
    traced_shape: tuple[int, ...] | None,
    *,
    batch_symbol: str,
    traced_batch_size: int,
    shape_witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
) -> bool:
    """Return whether a semantic candidate matches trace and all captured witnesses."""

    if traced_shape is None:
        return False
    trace_binding = ShapeBinding({batch_symbol: traced_batch_size}, {})
    if candidate.evaluate(trace_binding) != traced_shape:
        return False
    return all(
        (witness_shape := shapes.get(node_id)) is not None
        and candidate.evaluate(ShapeBinding({batch_symbol: batch_size}, {})) == witness_shape
        for batch_size, shapes in shape_witnesses.items()
    )


def _compile_recipes(
    graph: "SplitTraceGraph",
    value_shapes: Mapping[str, TensorShapeIR],
    batch_symbol: str,
) -> dict[str, tuple[ShapeRecipe, ...]]:
    """Lower symbolic output descriptors to exact per-node recipes."""

    recipes: dict[str, tuple[ShapeRecipe, ...]] = {}
    for node in graph.nodes:
        semantic = shape_semantic_for_node(node)
        if graph.backend == "torch" and semantic not in {"reshape", "expand", "factory"}:
            continue
        shape = value_shapes.get(node.canonical_id)
        if shape is None or node.output_shape is None:
            continue
        relations: list[tuple[tuple[int, ...], tuple[DimExpr, ...]]] = [
            (tuple(node.output_shape), shape.dims)
        ]
        for parent_label in node.parents:
            parent = graph.node_for_label(parent_label)
            if parent is None or parent.output_shape is None:
                continue
            parent_shape = value_shapes.get(parent.canonical_id)
            if parent_shape is not None:
                relations.append((tuple(parent.output_shape), parent_shape.dims))
        if not any(
            dim.contains(batch_symbol) for _concrete_shape, dims in relations for dim in dims
        ):
            continue
        descriptors: set[tuple[int, ...]] = set()
        if _shape_sensitive(node.op_type.lower()):
            descriptors.add(tuple(node.output_shape))
        descriptors.update(_captured_shape_descriptors(node))
        node_recipes: list[ShapeRecipe] = []
        for index, descriptor in enumerate(sorted(descriptors)):
            dims = next(
                (
                    resolved
                    for concrete_shape, relation_dims in relations
                    if (
                        resolved := _descriptor_exprs(
                            descriptor,
                            concrete_shape,
                            relation_dims,
                            batch_symbol,
                        )
                    )
                    is not None
                ),
                None,
            )
            if dims is None or not any(dim.contains(batch_symbol) for dim in dims):
                continue
            node_recipes.append(
                ShapeRecipe(
                    node_id=node.canonical_id,
                    recipe_id=f"shape:{node.canonical_id}:{index}",
                    captured=descriptor,
                    dims=dims,
                )
            )
        if node_recipes:
            recipes[node.canonical_id] = tuple(node_recipes)
    return recipes


def _descriptor_exprs(
    descriptor: tuple[int, ...],
    output_shape: tuple[int, ...],
    output_dims: tuple[DimExpr, ...],
    batch_symbol: str,
) -> tuple[DimExpr, ...] | None:
    """Derive a descriptor expression from its node output relation."""

    if descriptor == output_shape:
        return output_dims
    if (
        len(descriptor) == 1
        and descriptor[0] == prod(output_shape)
        and any(dim.contains(batch_symbol) for dim in output_dims)
    ):
        product = DimExpr.const(1)
        for dim in output_dims:
            product = DimExpr("mul", args=(product, dim))
        return (product,)
    dynamic_axes = [index for index, dim in enumerate(output_dims) if dim.contains(batch_symbol)]
    if (
        len(dynamic_axes) == 1
        and dynamic_axes[0] == 0
        and descriptor
        and output_shape
        and descriptor[0] == output_shape[0]
        and (len(descriptor) == len(output_shape) or -1 in descriptor)
    ):
        return (output_dims[0], *(DimExpr.const(dim) for dim in descriptor[1:]))
    return None


def _captured_shape_descriptors(node: "SplitTraceNode") -> set[tuple[int, ...]]:
    """Collect backend-native shape literals attached to one audited node."""

    descriptors: set[tuple[int, ...]] = set()
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or id(value) in seen:
            return
        if not isinstance(value, (int, str, bytes, bool, float)):
            seen.add(id(value))
        if isinstance(value, (tuple, list)):
            descriptor = _flat_int_shape(value)
            if descriptor is not None:
                descriptors.add(descriptor)
                return
            unwrapped = [getattr(item, "value", item) for item in value]
            for start in range(len(unwrapped)):
                descriptor = _flat_int_shape(unwrapped[start:])
                if descriptor is not None:
                    descriptors.add(descriptor)
                    break
            for item in value:
                visit(item)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        literal_value = getattr(value, "value", None)
        if literal_value is not None and literal_value is not value:
            visit(literal_value)
        params = getattr(value, "params", None)
        if isinstance(params, Mapping):
            for key in ("shape", "new_sizes", "sizes", "limit_indices", "slice_sizes"):
                if key in params:
                    visit(params[key])
        args = getattr(value, "args", None)
        if isinstance(args, (tuple, list)):
            visit(args)
        kwargs = getattr(value, "kwargs", None)
        if isinstance(kwargs, (tuple, list, Mapping)):
            visit(kwargs)
        inputs = getattr(value, "inputs", None)
        if inputs is not None:
            for item in inputs:
                tensor = getattr(item, "tensor", None)
                if tensor is not None:
                    try:
                        visit(tensor.numpy().tolist())
                    except Exception:
                        pass
        native_uop = getattr(value, "uop", None)
        if native_uop is not None and native_uop is not value:
            visit(native_uop)
        native_arg = getattr(value, "arg", None)
        if isinstance(native_arg, (tuple, list)):
            visit(native_arg)
        src = getattr(value, "src", None)
        if src:
            op_name = str(getattr(value, "op", "")).lower()
            if "stack" in op_name:
                scalar_values = [getattr(item, "arg", None) for item in src]
                if all(isinstance(item, Integral) for item in scalar_values):
                    descriptors.add(
                        tuple(int(item) for item in scalar_values if isinstance(item, Integral))
                    )
            for item in src:
                visit(item)

    visit(node.target)
    visit(node.args_template)
    visit(node.kwargs_template)
    return descriptors


def _unresolved_dynamic_nodes(
    graph: "SplitTraceGraph",
    value_shapes: Mapping[str, TensorShapeIR],
    batch_symbol: str,
    *,
    traced_batch_size: int,
    shape_witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
    dynamic_batch: tuple[int, int],
) -> dict[str, str]:
    """Return shape-sensitive nodes whose dynamic relation was not solved."""

    unresolved: dict[str, str] = {}
    for node in graph.nodes:
        semantic = shape_semantic_for_node(node)
        if not _shape_sensitive(node.op_type.lower()) and semantic is None:
            continue
        parent_dynamic = any(
            parent_node is not None
            and (shape := value_shapes.get(parent_node.canonical_id)) is not None
            and any(dim.contains(batch_symbol) for dim in shape.dims)
            for parent in node.parents
            if (parent_node := graph.node_for_label(parent)) is not None
        )
        output = value_shapes.get(node.canonical_id)
        output_dynamic = output is not None and any(
            dim.contains(batch_symbol) for dim in output.dims
        )
        expected_witnesses = set(range(dynamic_batch[0], dynamic_batch[1] + 1)) - {
            traced_batch_size
        }
        witnesses_complete = expected_witnesses == set(shape_witnesses)
        witness_shapes = tuple(shapes.get(node.canonical_id) for shapes in shape_witnesses.values())
        witnessed_unchanged = witnesses_complete and all(
            shape == node.output_shape for shape in witness_shapes
        )
        if parent_dynamic and not output_dynamic and not witnessed_unchanged:
            unresolved[node.canonical_id] = "dynamic shape relation could not be proven"
            continue
        descriptor_mentions_batch = any(
            traced_batch_size in descriptor for descriptor in _captured_shape_descriptors(node)
        )
        if (
            not parent_dynamic
            and not output_dynamic
            and not witnessed_unchanged
            and descriptor_mentions_batch
            and node.output_shape is not None
            and traced_batch_size in node.output_shape
        ):
            unresolved[node.canonical_id] = (
                "captured shape literal may depend on batch and requires proof"
            )
    return unresolved


def _rewrite_recipe_tree(
    value: Any,
    recipes: Sequence[ShapeRecipe],
    binding: ShapeBinding,
) -> Any:
    """Rewrite exact flat shape descriptors inside a literal tree."""

    if isinstance(value, tuple):
        flat = _flat_int_shape(value)
        if flat is not None:
            for recipe in recipes:
                if flat == recipe.captured:
                    return recipe.evaluate(binding)
        for recipe in recipes:
            width = len(recipe.captured)
            if width <= len(value):
                suffix = _flat_int_shape(value[-width:])
                if suffix == recipe.captured:
                    return (*value[:-width], *recipe.evaluate(binding))
        return tuple(_rewrite_recipe_tree(item, recipes, binding) for item in value)
    if isinstance(value, list):
        flat = _flat_int_shape(value)
        if flat is not None:
            for recipe in recipes:
                if flat == recipe.captured:
                    return list(recipe.evaluate(binding))
        for recipe in recipes:
            width = len(recipe.captured)
            if width <= len(value):
                suffix = _flat_int_shape(value[-width:])
                if suffix == recipe.captured:
                    return [*value[:-width], *recipe.evaluate(binding)]
        return [_rewrite_recipe_tree(item, recipes, binding) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_recipe_tree(item, recipes, binding) for key, item in value.items()}
    return value


def _flat_int_shape(value: Sequence[Any]) -> tuple[int, ...] | None:
    """Return a flat integer descriptor or ``None``."""

    if not value or not all(
        isinstance(item, Integral) and not isinstance(item, bool) for item in value
    ):
        return None
    return tuple(int(item) for item in value)


def _preserves_batch(op: str) -> bool:
    """Return whether an operation normally preserves aligned batch axes."""

    return any(
        token in op
        for token in (
            "conv",
            "pool",
            "norm",
            "relu",
            "gelu",
            "softmax",
            "dropout",
            "slice",
            "gather",
            "index",
            "pad",
            "shrink",
            "copy",
            "cast",
            "identity",
        )
    )


def _shape_bearing_parents(
    graph: "SplitTraceGraph",
    node: "SplitTraceNode",
    concrete: Mapping[str, tuple[int, ...] | None],
    seen: set[str] | None = None,
) -> list["SplitTraceNode"]:
    """Resolve transparent shapeless region nodes to shape-bearing parents."""

    if concrete.get(node.canonical_id) is not None:
        return [node]
    visited = set() if seen is None else seen
    if node.canonical_id in visited:
        return []
    visited.add(node.canonical_id)
    result: list[SplitTraceNode] = []
    for parent_label in node.parents:
        parent = graph.node_for_label(parent_label)
        if parent is not None:
            result.extend(_shape_bearing_parents(graph, parent, concrete, visited))
    return result


def _is_parameter_only_reshape(
    graph: "SplitTraceGraph",
    node: "SplitTraceNode",
    concrete: Mapping[str, tuple[int, ...] | None],
    dynamic_axes: Mapping[str, set[int]],
) -> bool:
    """Return whether a reshape only lays out a fixed parameter tensor."""

    if "reshape" not in node.op_type.lower() or not node.parents or node.output_shape is None:
        return False
    parent = graph.node_for_label(node.parents[0])
    if parent is None or dynamic_axes.get(parent.canonical_id):
        return False
    parent_shape = concrete.get(parent.canonical_id)
    if parent_shape is None or not (
        parent.is_param_source or parent.is_buffer or parent.param_refs
    ):
        return False
    parent_numel = prod(parent_shape)
    output_numel = 1
    for dim in node.output_shape:
        output_numel *= dim
    return parent_numel == output_numel


def _is_parameter_lineage(
    graph: "SplitTraceGraph",
    node: "SplitTraceNode",
    seen: set[str] | None = None,
) -> bool:
    """Return whether a node is derived exclusively from parameter or buffer state."""

    if node.is_param_source or node.is_buffer or node.param_refs:
        return True
    if node.is_input:
        return False
    if not node.parents:
        return True
    visited = set() if seen is None else seen
    if node.canonical_id in visited:
        return False
    visited.add(node.canonical_id)
    parents = [graph.node_for_label(label) for label in node.parents]
    resolved = [parent for parent in parents if parent is not None]
    return bool(resolved) and all(
        _is_parameter_lineage(graph, parent, visited.copy()) for parent in resolved
    )


def _node_permutation(node: "SplitTraceNode", rank: int) -> tuple[int, ...] | None:
    """Return an audited output-to-input axis permutation for one node."""

    op = node.op_type.lower()
    if "transpose" not in op and "permute" not in op and "swapaxes" not in op:
        return None
    params = getattr(node.target, "params", None)
    if isinstance(params, Mapping):
        for key in ("permutation", "axes"):
            value = params.get(key)
            if isinstance(value, (tuple, list)) and len(value) == rank:
                return tuple(int(item) for item in value)
    template = getattr(node.args_template, "args", None)
    if isinstance(template, tuple):
        literal_args: list[int] = []
        for item in template[1:]:
            value = getattr(item, "value", item)
            if isinstance(value, Integral):
                literal_args.append(int(value))
        if "permute" in op and len(literal_args) == rank:
            return tuple(int(item) % rank for item in literal_args)
        if ("transpose" in op or "swapaxes" in op) and len(literal_args) >= 2:
            first, second = (int(item) % rank for item in literal_args[-2:])
            permutation = list(range(rank))
            permutation[first], permutation[second] = permutation[second], permutation[first]
            return tuple(permutation)
    return None


def _is_elementwise(op: str) -> bool:
    """Return whether an op participates in broadcast shape unification."""

    return _preserves_batch(op) or any(
        token in op
        for token in (
            "add",
            "sub",
            "mul",
            "div",
            "reciprocal",
            "maximum",
            "minimum",
            "where",
            "exp",
        )
    )


def _is_reshape_like(op: str) -> bool:
    """Return whether an op changes shape while preserving element count."""

    return any(token in op for token in ("reshape", "view", "flatten", "expand", "broadcast"))


def _shape_sensitive(op: str) -> bool:
    """Return whether an op may carry a lowered shape descriptor."""

    return _is_reshape_like(op) or any(
        token in op
        for token in (
            "concat",
            "stack",
            "slice",
            "full",
            "zeros",
            "ones",
            "empty",
            "pad",
            "shrink",
            "repeat",
            "tile",
        )
    )


def shape_semantic_for_node(node: "SplitTraceNode") -> ShapeSemantic | None:
    """Return an exact semantic category from the captured function identity."""

    template = node.args_template
    func_id = getattr(template, "func_id", None)
    namespace = getattr(func_id, "namespace", None)
    qualname = getattr(func_id, "qualname", None)
    if isinstance(namespace, str) and isinstance(qualname, str):
        return _TORCH_SHAPE_SEMANTICS.get((namespace, qualname.rsplit(".", 1)[-1]))
    return None


def _witness_scaled_axes(
    node_id: str,
    traced_shape: tuple[int, ...],
    traced_batch_size: int,
    witnesses: Mapping[int, Mapping[str, tuple[int, ...] | None]],
) -> set[int]:
    """Return axes proven to be a fixed multiple of batch by all witnesses."""

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
        if traced_dim % traced_batch_size == 0
        and all(
            shape[axis] * traced_batch_size == traced_dim * batch_size
            for batch_size, shape in witnessed_shapes
        )
    }


def _witness_axis_diagnostics(
    graph: "SplitTraceGraph",
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
            if traced_batch_size > 0 and dim % traced_batch_size == 0
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


def _shape_proof_sources(
    graph: "SplitTraceGraph",
    value_shapes: Mapping[str, TensorShapeIR],
    batch_symbol: str,
    witness_diagnostics: Mapping[str, Mapping[str, tuple[int, ...]]],
) -> dict[str, str]:
    """Return the strongest proof source for each batch-dependent graph value."""

    sources: dict[str, str] = {}
    input_ids = set(graph.input_node_ids)
    for node in graph.nodes:
        shape = value_shapes.get(node.canonical_id)
        if shape is None or not any(dim.contains(batch_symbol) for dim in shape.dims):
            continue
        semantic = shape_semantic_for_node(node)
        witness_axes = witness_diagnostics.get(node.canonical_id, {}).get("accepted_axes", ())
        if node.canonical_id in input_ids:
            source = "input_batch_axis"
        elif semantic == "repeat":
            source = "semantic:repeat"
        elif witness_axes:
            source = "shape_witness"
        elif semantic is not None:
            source = f"semantic:{semantic}"
        else:
            source = "symbolic_propagation"
        sources[node.canonical_id] = source
    return sources


__all__ = [
    "DimExpr",
    "InputLeaf",
    "ShapeBinding",
    "ShapeConstraintIR",
    "ShapeProgram",
    "ShapeRecipe",
    "TensorShapeIR",
    "compile_shape_program",
    "flatten_input_leaves",
    "shape_semantic_for_node",
]
