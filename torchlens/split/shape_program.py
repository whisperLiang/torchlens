"""Backend-neutral symbolic batch constraint compilation and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError

if TYPE_CHECKING:
    from .adapters.base import SplitBackendAdapter
    from .graph import SplitTraceGraph, SplitTraceNode
    from .ir import SplitRequest


DimOp = Literal["const", "symbol", "add", "mul", "floordiv", "ceildiv", "min", "max"]


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

    def bind(
        self,
        inputs: tuple[Any, ...],
        input_kwargs: Mapping[str, Any] | None,
        *,
        adapter: "SplitBackendAdapter",
        backend: str,
        split_point: str,
    ) -> ShapeBinding:
        """Bind and validate the batch symbol from runtime inputs."""

        leaves = flatten_input_leaves(inputs, input_kwargs, adapter=adapter)
        return self.bind_flat_values(
            tuple(leaf.value for leaf in leaves),
            shape_of=adapter.shape,
            backend=backend,
            split_point=split_point,
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
                    module_path=None,
                    op_type=None,
                    layer_label=None,
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
                    module_path=None,
                    op_type=None,
                    layer_label=None,
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
                    module_path=None,
                    op_type=None,
                    layer_label=None,
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
        _walk_input((input_kwargs or {})[key], f"/kwargs/{_escape_pointer(str(key))}", adapter, leaves)
    return tuple(leaves)


def compile_shape_program(
    graph: "SplitTraceGraph",
    inputs: tuple[Any, ...],
    input_kwargs: Mapping[str, Any] | None,
    request: "SplitRequest",
    *,
    adapter: "SplitBackendAdapter",
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
                module_path=None,
                op_type=None,
                layer_label=None,
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

    traced_shapes = {
        path: tuple(adapter.shape(input_by_path[path].value) or ()) for path in axes
    }
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
    )
    constraints = tuple(
        ShapeConstraintIR(
            constraint_id=f"axis:{node_id}",
            kind="axis",
            value_ids=(node_id,),
            lhs=shape.dims[next(i for i, dim in enumerate(shape.dims) if dim.contains(request.batch_symbol))],
            rhs=DimExpr.symbol(request.batch_symbol),
            description="value carries the declared batch symbol",
        )
        for node_id, shape in value_shapes.items()
        if any(dim.contains(request.batch_symbol) for dim in shape.dims)
    )
    recipes = _compile_recipes(graph, value_shapes, request.batch_symbol)
    unresolved = _unresolved_dynamic_nodes(graph, value_shapes, request.batch_symbol)
    payload = {
        "batch_symbol": request.batch_symbol,
        "range": request.dynamic_batch,
        "axes": sorted(axes.items()),
        "values": {
            key: [dim.as_dict() for dim in value.dims] for key, value in sorted(value_shapes.items())
        },
        "recipes": {
            key: [recipe.captured for recipe in value] for key, value in sorted(recipes.items())
        },
    }
    fingerprint = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
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
            inferred = set(dynamic_axes.get(node.canonical_id, set()))
            permutation = _node_permutation(node, len(output_shape))
            for parent in parent_nodes:
                parent_shape = concrete.get(parent.canonical_id)
                for axis in dynamic_axes.get(parent.canonical_id, set()):
                    if parent_shape is None:
                        continue
                    if permutation is not None and axis in permutation:
                        inferred.add(permutation.index(axis))
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
            if _is_reshape_like(op) and parent_nodes and any(
                dynamic_axes.get(parent.canonical_id) for parent in parent_nodes
            ):
                first_parent = parent_nodes[0]
                first_parent_shape = concrete.get(first_parent.canonical_id)
                if (
                    first_parent_shape
                    and 0 in dynamic_axes.get(first_parent.canonical_id, set())
                    and output_shape
                    and output_shape[0] % traced_batch_size == 0
                ):
                    inferred.add(0)
                else:
                    inferred.update(_reshape_batch_axes(output_shape, traced_batch_size))
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
    return result


def _compile_recipes(
    graph: "SplitTraceGraph",
    value_shapes: Mapping[str, TensorShapeIR],
    batch_symbol: str,
) -> dict[str, tuple[ShapeRecipe, ...]]:
    """Lower symbolic output descriptors to exact per-node recipes."""

    recipes: dict[str, tuple[ShapeRecipe, ...]] = {}
    for node in graph.nodes:
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
            dim.contains(batch_symbol)
            for _concrete_shape, dims in relations
            for dim in dims
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
    dynamic_axes = [
        index for index, dim in enumerate(output_dims) if dim.contains(batch_symbol)
    ]
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
) -> dict[str, str]:
    """Return shape-sensitive nodes whose dynamic relation was not solved."""

    unresolved: dict[str, str] = {}
    for node in graph.nodes:
        if not _shape_sensitive(node.op_type.lower()):
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
        if parent_dynamic and not output_dynamic:
            unresolved[node.canonical_id] = "dynamic shape relation could not be proven"
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

    if not value or not all(isinstance(item, Integral) and not isinstance(item, bool) for item in value):
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


def _reshape_batch_axes(
    output_shape: tuple[int, ...],
    traced_batch_size: int | None,
) -> set[int]:
    """Return candidate output dimensions carrying a batch product."""

    if traced_batch_size is None or traced_batch_size <= 0:
        return set()
    exact = {index for index, dim in enumerate(output_shape) if dim == traced_batch_size}
    if len(exact) == 1:
        return exact
    if exact:
        return set()
    divisible = {
        index
        for index, dim in enumerate(output_shape)
        if dim > traced_batch_size and dim % traced_batch_size == 0
    }
    return divisible if len(divisible) == 1 else set()


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
]
