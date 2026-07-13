"""JAX native-IR split replay adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...backends.jax.jaxpr import (
    JaxEquationCapture,
    JaxRegionCapture,
    _evaluate_closed_jaxpr_no_capture,
    replay_equation,
)
from ...ir.container import rebuild_container_from_spec, reorder_container_leaves
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import SplitTraceGraph, SplitTraceNode
from ..planner import SplitPlan
from ..shape_program import ShapeBinding
from ..ir import SplitRequest
from .base import SegmentBundle, SplitPolicyMixin, boundary_overlay


def _jax() -> Any:
    """Import JAX lazily for split adapter operations."""

    import jax

    return jax


def _jnp() -> Any:
    """Import jax.numpy lazily for split adapter operations."""

    import jax.numpy as jnp

    return jnp


def _slice_output_by_path(output: Any, path: tuple[Any, ...]) -> Any:
    """Return one output leaf addressed by a container path."""

    current = output
    for component in path:
        component = _jax_tree_key(component)
        if isinstance(current, dict):
            current = current[component]
        elif isinstance(component, str) and hasattr(current, component):
            current = getattr(current, component)
        else:
            current = current[component]
    return current


def _jax_tree_key(component: Any) -> Any:
    """Convert a JAX tree path entry to a native container key."""

    if hasattr(component, "key"):
        return component.key
    if hasattr(component, "idx"):
        return component.idx
    return component


def _rebuild_output_container(
    leaves: list[tuple[tuple[Any, ...], Any]],
) -> Any:
    """Rebuild a JAX output pytree from its recorded leaf paths."""

    def build(items: list[tuple[tuple[Any, ...], Any]]) -> Any:
        if len(items) == 1 and not items[0][0]:
            return items[0][1]
        keys: list[Any] = []
        for path, _value in items:
            if path:
                key = _jax_tree_key(path[0])
                if key not in keys:
                    keys.append(key)
        if not keys:
            return tuple(value for _path, value in items)
        children = {
            key: build(
                [
                    (tuple(_jax_tree_key(item) for item in path[1:]), value)
                    for path, value in items
                    if path and _jax_tree_key(path[0]) == key
                ]
            )
            for key in keys
        }
        if all(isinstance(key, str) for key in keys):
            return children
        if all(isinstance(key, int) for key in keys):
            return tuple(children[index] for index in sorted(children))
        return children

    return build(leaves)


def _is_jax_tensor(value: Any) -> bool:
    """Return whether ``value`` is a JAX array-like tensor."""

    jax = _jax()
    array_type = getattr(jax, "Array", ())
    if array_type and isinstance(value, array_type):
        return True
    module = type(value).__module__
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (module.startswith("jax") or module.startswith("jaxlib"))
    )


def _flatten_tensor_leaves(value: Any) -> list[Any]:
    """Collect JAX tensor leaves in deterministic traversal order."""

    if _is_jax_tensor(value):
        return [value]
    if isinstance(value, dict):
        leaves: list[Any] = []
        for key in sorted(value, key=repr):
            leaves.extend(_flatten_tensor_leaves(value[key]))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_flatten_tensor_leaves(item))
        return leaves
    try:
        leaves = _jax().tree_util.tree_leaves(value)
    except Exception:
        return []
    if len(leaves) == 1 and leaves[0] is value:
        return []
    flattened: list[Any] = []
    for item in leaves:
        if _is_jax_tensor(item):
            flattened.append(item)
    return flattened


class _JaxGeneratedSegmentBase:
    """Shared JAX native-IR replay helpers."""

    def __init__(
        self,
        *,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
        node_ids: frozenset[str],
    ) -> None:
        """Create a generated JAX replay segment."""

        self.graph = graph
        self.plan = plan
        self.spec = spec
        self.node_ids = node_ids
        self._node_by_id = graph.node_by_id
        self._label_to_id = graph.node_id_by_alias
        self._shape_binding: ShapeBinding | None = None

    def _context(self, node: SplitTraceNode, reason: str) -> SplitErrorContext:
        """Build an error context for ``node``."""

        return SplitErrorContext(
            backend="jax",
            split_point=self.spec.boundary,
            module_path=node.module_path,
            op_type=node.op_type,
            layer_label=node.label,
            reason=reason,
            traced_shape=node.output_shape,
            dtype=node.dtype,
        )

    @staticmethod
    def _is_replay_source_node(node: SplitTraceNode) -> bool:
        """Return whether a target-less node is safe to seed from trace payload."""

        return not node.parents and not node.is_output

    def _source_value(self, node: SplitTraceNode) -> Any:
        """Return a replay value for source-like nodes."""

        value = getattr(node.op, "out", None)
        if value is None:
            raise SplitUnsupportedError(
                f"{node.label!r} source value is unavailable.",
                context=self._context(node, "missing source value"),
            )
        return value

    def _resolve_parent_value(
        self,
        parent_label: str,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> Any:
        """Resolve a raw/display/canonical parent label from the replay overlay."""

        parent_id = self._label_to_id.get(parent_label)
        if parent_id is None or parent_id not in overlay:
            raise SplitUnsupportedError(
                f"{node.label!r} references unavailable JAX parent {parent_label!r}.",
                context=self._context(node, "missing parent value"),
            )
        return overlay[parent_id]

    def _params_override(
        self,
        capture: JaxEquationCapture,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return dynamic-batch rewritten primitive params when needed."""

        shape_program = self.graph.shape_program
        binding = self._shape_binding
        if shape_program is None or binding is None:
            return None
        rewritten = {
            key: shape_program.rewrite(node.canonical_id, value, binding)
            for key, value in capture.params.items()
        }
        return None if rewritten == dict(capture.params) else rewritten

    def _equation_inputs(
        self,
        capture: JaxEquationCapture,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Rebuild one JAX equation input tuple from replay overlay parents."""

        inputs = list(capture.input_values)
        graph_positions = getattr(node.op, "parent_arg_positions", {}).get("args", {})
        for position, parent_label in graph_positions.items():
            if not isinstance(position, int) or position < 0 or position >= len(inputs):
                raise SplitUnsupportedError(
                    f"{node.label!r} has invalid JAX parent arg position {position!r}.",
                    context=self._context(node, "invalid parent position"),
                )
            inputs[position] = self._resolve_parent_value(str(parent_label), node, overlay)
        return tuple(inputs)

    def _execute_jax_capture(self, node: SplitTraceNode, overlay: dict[str, Any]) -> Any:
        """Execute one captured JAX primitive equation."""

        capture = node.target
        if isinstance(capture, JaxRegionCapture):
            if capture.kind == "region_output":
                for parent in node.parents:
                    parent_id = self._label_to_id.get(parent)
                    if parent_id in overlay:
                        return overlay[parent_id]
                return capture.input_values[0]
            inputs = list(capture.input_values)
            graph_positions = getattr(node.op, "parent_arg_positions", {}).get("args", {})
            for position, parent_label in graph_positions.items():
                if not isinstance(position, int) or position < 0 or position >= len(inputs):
                    raise SplitUnsupportedError(
                        f"{node.label!r} has invalid JAX region parent position {position!r}.",
                        context=self._context(node, "invalid region parent position"),
                    )
                inputs[position] = self._resolve_parent_value(str(parent_label), node, overlay)
            try:
                if capture.primitive in {"custom_jvp_call", "custom_vjp_call"}:
                    outputs = _evaluate_closed_jaxpr_no_capture(
                        capture.params["call_jaxpr"], tuple(inputs)
                    )
                else:
                    result = capture.primitive_obj.bind(*tuple(inputs), **capture.params)
                    outputs = tuple(result if capture.primitive_obj.multiple_results else (result,))
            except Exception as exc:
                raise SplitUnsupportedError(
                    f"JAX region replay failed for {node.label!r}: {exc}",
                    context=self._context(node, "JAX region replay failed"),
                ) from exc
            return outputs[0] if len(outputs) == 1 else tuple(outputs)
        if not isinstance(capture, JaxEquationCapture):
            raise SplitUnsupportedError(
                f"{node.label!r} has no JAX equation capture.",
                context=self._context(node, "missing JAX equation capture"),
            )
        try:
            outputs = replay_equation(
                capture,
                self._equation_inputs(capture, node, overlay),
                params_override=self._params_override(capture, node, overlay),
            )
        except Exception as exc:
            raise SplitUnsupportedError(
                f"JAX split replay failed for {node.label!r}: {exc}",
                context=self._context(node, "JAX primitive replay failed"),
            ) from exc
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _execute_nodes(self, overlay: dict[str, Any]) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids:
                continue
            if node.is_input:
                continue
            if node.target is None and self._is_replay_source_node(node):
                if node.canonical_id not in overlay and not node.is_output:
                    overlay[node.canonical_id] = self._source_value(node)
                continue
            if node.is_output and node.target is None:
                continue
            output = self._execute_jax_capture(node, overlay)
            # Final jaxpr outputs are captured as individual leaf equations;
            # their DictKey/SequenceKey path describes the public container,
            # not an additional index into the equation result.
            overlay[node.canonical_id] = (
                output
                if node.is_output
                else _slice_output_by_path(output, node.output_container_path)
            )
        return overlay


class JaxGeneratedPrefix(_JaxGeneratedSegmentBase):
    """Generated-eager JAX prefix segment."""

    def __call__(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
        detach_boundary: bool,
    ) -> ReplayBoundary:
        """Run the prefix and return a replay boundary."""

        input_leaves = _flatten_tensor_leaves(inputs)
        input_leaves.extend(_flatten_tensor_leaves(input_kwargs or {}))
        if len(input_leaves) != len(self.graph.input_node_ids):
            raise SplitUnsupportedError(
                "Runtime inputs do not match traced JAX tensor input count.",
                context=SplitErrorContext(
                    backend="jax",
                    split_point=self.spec.boundary,
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="input count mismatch",
                ),
            )
        overlay = {
            node_id: value for node_id, value in zip(self.graph.input_node_ids, input_leaves)
        }
        if self.graph.shape_program is not None:
            self._shape_binding = self.graph.shape_program.bind_flat_values(
                input_leaves,
                shape_of=lambda value: tuple(int(dim) for dim in value.shape),
                backend="jax",
                split_point=self.spec.boundary,
            )
        runtime_batch_size = None if self._shape_binding is None else self._shape_binding.batch_size
        self._execute_nodes(overlay)
        boundary_tensors: dict[str, Any] = {}
        prefix_tensors: dict[str, Any] = {}
        for node_id in self.plan.boundary_node_ids:
            node = self._node_by_id[node_id]
            key = boundary_key_for_node(node.canonical_id, node.output_container_path)
            value = overlay[node_id]
            prefix_tensors[key] = value
            boundary_tensors[key] = value
        metadata: dict[str, Any] = {
            "split_id": self.plan.split_id,
            "graph_shape_hash": self.graph.graph_shape_hash,
            "batch_symbol": self.spec.batch_symbol,
            "dynamic_batch": self.spec.dynamic_batch,
            "runtime_batch_size": runtime_batch_size,
            "shape_program_hash": (
                None if self.graph.shape_program is None else self.graph.shape_program.fingerprint
            ),
            "device_policy": self.spec.device_policy,
            "supports_prefix_backward": not detach_boundary,
        }
        if not detach_boundary:
            metadata["prefix_boundary_tensors"] = prefix_tensors
            metadata["prefix_inputs"] = inputs
        return ReplayBoundary(
            backend="jax",
            tensors=boundary_tensors,
            spec=self.plan.boundary_spec,
            metadata=metadata,
        )


class JaxGeneratedSuffix(_JaxGeneratedSegmentBase):
    """Generated-eager JAX suffix segment."""

    def __call__(self, boundary: ReplayBoundary) -> Any:
        """Run the suffix from ``boundary`` and reconstruct final output."""

        overlay = boundary_overlay(boundary, self.plan)
        runtime_batch_size = boundary.metadata.get("runtime_batch_size")
        if self.graph.shape_program is not None and runtime_batch_size is not None:
            self._shape_binding = self.graph.shape_program.binding_from_batch(
                int(runtime_batch_size)
            )
        self._execute_nodes(overlay)
        return self._reconstruct_output(overlay)

    def _output_leaf(self, node: SplitTraceNode, overlay: dict[str, Any]) -> Any:
        """Return one final-output leaf value."""

        if node.canonical_id in overlay:
            return overlay[node.canonical_id]
        for parent in node.parents:
            parent_id = self._label_to_id.get(parent)
            if parent_id in overlay:
                return overlay[parent_id]
        return getattr(node.op, "out", None)

    def _reconstruct_output(self, overlay: dict[str, Any]) -> Any:
        """Reconstruct the traced model output value."""

        output_nodes = [self._node_by_id[node_id] for node_id in self.graph.output_node_ids]
        if not output_nodes:
            if not overlay:
                return None
            return overlay[next(reversed(overlay))]
        leaves = [
            (node.output_container_path, self._output_leaf(node, overlay)) for node in output_nodes
        ]
        spec = next(
            (node.output_container_spec for node in output_nodes if node.output_container_spec),
            None,
        )
        if spec is not None:
            return rebuild_container_from_spec(
                spec,
                reorder_container_leaves(spec, leaves),
            )
        if len(leaves) == 1 and not leaves[0][0]:
            return leaves[0][1]
        return _rebuild_output_container(leaves)


class JaxSplitAdapter(SplitPolicyMixin):
    """JAX split backend adapter."""

    name = "jax"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_dynamic_batch = True
    native_target_types = frozenset({"JaxEquationCapture", "JaxRegionCapture"})

    def is_tensor(self, value: Any) -> bool:
        """Return whether ``value`` is a JAX tensor."""

        return _is_jax_tensor(value)

    def shape(self, value: Any) -> tuple[int, ...] | None:
        """Return tensor shape."""

        if not self.is_tensor(value):
            return None
        return tuple(int(dim) for dim in value.shape)

    def dtype_name(self, value: Any) -> str | None:
        """Return tensor dtype name."""

        dtype = getattr(value, "dtype", None)
        return None if dtype is None else str(dtype)

    def requires_grad(self, value: Any) -> bool | None:
        """Return no eager autograd flag for JAX arrays."""

        del value
        return None

    def detach(self, value: Any) -> Any:
        """Detach tensor values; JAX arrays are immutable values."""

        return value

    def clone(self, value: Any) -> Any:
        """Clone tensor values."""

        return _jnp().array(value) if self.is_tensor(value) else value

    def to_device(self, value: Any, device: Any) -> Any:
        """Move tensor values to a device when JAX accepts the device object."""

        if not self.is_tensor(value):
            return value
        try:
            return _jax().device_put(value, device)
        except Exception:
            return value

    def collate(self, values: list[Any]) -> Any:
        """Stack JAX tensor values."""

        if values and self.is_tensor(values[0]):
            return _jnp().stack(values)
        return list(values)

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like a tensor."""

        return _jnp().zeros_like(value)

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two tensors are numerically close."""

        if self.is_tensor(left) and self.is_tensor(right):
            return bool(_jnp().allclose(left, right, atol=atol, rtol=rtol))
        return left == right

    def build_segments(
        self,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
    ) -> SegmentBundle:
        """Build JAX native-IR replay prefix/suffix segments."""

        prefix = JaxGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.prefix_node_ids,
        )
        training_prefix = JaxGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=replace(spec, features=replace(spec.features, training=True)),
            node_ids=plan.prefix_node_ids,
        )
        suffix = JaxGeneratedSuffix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.suffix_node_ids,
        )
        return SegmentBundle(prefix=prefix, training_prefix=training_prefix, suffix=suffix)


__all__ = ["JaxGeneratedPrefix", "JaxGeneratedSuffix", "JaxSplitAdapter"]
