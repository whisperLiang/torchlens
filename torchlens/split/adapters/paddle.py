"""Paddle generated-eager split replay adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ... import _state
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import SplitTraceGraph, SplitTraceNode
from ..planner import SplitPlan
from ..shape_program import ShapeBinding
from ..ir import SplitRequest
from .base import SegmentBundle, SplitPolicyMixin


def _paddle() -> Any:
    """Import Paddle lazily for split adapter operations."""

    import paddle

    return paddle


def _is_tensor_marker(value: Any) -> bool:
    """Return whether a Paddle template leaf marks a tensor producer."""

    return isinstance(value, dict) and value.get("kind") == "tensor" and "label" in value


def _slice_output_by_path(output: Any, path: tuple[Any, ...]) -> Any:
    """Return one output leaf addressed by a Paddle capture path."""

    current = output
    for component in path:
        if isinstance(current, dict):
            current = current[component]
        elif isinstance(component, str) and hasattr(current, component):
            current = getattr(current, component)
        else:
            current = current[component]
    return current


def _flatten_tensor_leaves(value: Any, paddle: Any) -> list[Any]:
    """Collect Paddle tensor leaves in deterministic traversal order."""

    if isinstance(value, paddle.Tensor):
        return [value]
    if isinstance(value, dict):
        leaves: list[Any] = []
        for key in sorted(value, key=repr):
            leaves.extend(_flatten_tensor_leaves(value[key], paddle))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_flatten_tensor_leaves(item, paddle))
        return leaves
    return []


def _param_ref_handle(param: Any) -> Any:
    """Return the live backend handle for a captured Paddle parameter."""

    handle = getattr(param, "_param_ref", None)
    if handle is None:
        handle = getattr(param, "handle", None)
    return handle


def _numel_from_shape(shape: Any) -> int | None:
    """Return the product of a concrete shape-like value."""

    try:
        dims = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None
    product = 1
    for dim in dims:
        product *= dim
    return product


class _LiveParamCursor:
    """Resolve unlabeled positional Paddle parameter template leaves in order."""

    def __init__(self, node: SplitTraceNode) -> None:
        """Create a cursor over live parameter handles for ``node``."""

        self._handles = [
            handle for param in node.param_refs if (handle := _param_ref_handle(param)) is not None
        ]
        self._index = 0

    def next(self) -> Any | None:
        """Return the next live parameter handle, if any."""

        if self._index >= len(self._handles):
            return None
        handle = self._handles[self._index]
        self._index += 1
        return handle


class _PaddleGeneratedSegmentBase:
    """Shared generated-eager replay helpers for Paddle."""

    def __init__(
        self,
        *,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
        node_ids: frozenset[str],
    ) -> None:
        """Create a generated replay segment."""

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
            backend="paddle",
            split_point=self.spec.boundary,
            module_path=node.module_path,
            op_type=node.op_type,
            layer_label=node.label,
            reason=reason,
            traced_shape=node.output_shape,
            dtype=node.dtype,
        )

    def _param_component_value(self, key: Any, node: SplitTraceNode) -> Any:
        """Resolve an unlabeled Paddle tensor template leaf from ``node.param_refs``."""

        keyed_name = self._param_name_for_template_key(key, node)
        key_text = str(keyed_name if keyed_name is not None else key)
        for param in node.param_refs:
            name = getattr(param, "name", None)
            address = getattr(param, "address", None)
            if name == key_text or (
                isinstance(address, str) and address.rsplit(".", 1)[-1] == key_text
            ):
                handle = _param_ref_handle(param)
                if handle is not None:
                    return handle
        raise SplitUnsupportedError(
            f"{node.label!r} has an unlabeled Paddle tensor template leaf.",
            context=self._context(node, "unlabeled tensor template"),
        )

    @staticmethod
    def _param_name_for_template_key(key: Any, node: SplitTraceNode) -> str | None:
        """Return the Paddle parameter name implied by a positional op argument."""

        if not isinstance(key, int):
            return None
        if node.op_type in {"c_ops.conv2d", "c_ops.depthwise_conv2d"} and key == 1:
            return "weight"
        if node.op_type in {"c_ops.depthwise_conv2d_bias"}:
            return {1: "weight", 2: "bias"}.get(key)
        if node.op_type == "c_ops.batch_norm":
            return {1: "_mean", 2: "_variance", 3: "weight", 4: "bias"}.get(key)
        return None

    def _shape_matched_param_component_value(self, node: SplitTraceNode) -> Any | None:
        """Resolve an unlabeled parameter by matching this node's output shape."""

        if node.output_shape is None:
            return None
        output_numel = _numel_from_shape(node.output_shape)
        if output_numel is None:
            return None
        matches: list[Any] = []
        for param in node.param_refs:
            handle = _param_ref_handle(param)
            if handle is None:
                continue
            if _numel_from_shape(getattr(handle, "shape", None)) == output_numel:
                matches.append(handle)
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_component(
        self,
        component: Any,
        node: SplitTraceNode,
        overlay: dict[str, Any],
        *,
        param_cursor: _LiveParamCursor,
        template_key: Any | None = None,
    ) -> Any:
        """Resolve one captured Paddle template component."""

        if _is_tensor_marker(component):
            label = component.get("label")
            if label is None:
                keyed_name = (
                    template_key
                    if isinstance(template_key, str)
                    else self._param_name_for_template_key(template_key, node)
                )
                if keyed_name is not None:
                    return self._param_component_value(keyed_name, node)
                value = self._shape_matched_param_component_value(node)
                if value is not None:
                    return value
                value = param_cursor.next()
                if value is not None:
                    return value
            if not isinstance(label, str):
                raise SplitUnsupportedError(
                    f"{node.label!r} has an unlabeled Paddle tensor template leaf.",
                    context=self._context(node, "unlabeled tensor template"),
                )
            node_id = self._label_to_id.get(label)
            if node_id is None or node_id not in overlay:
                raise SplitUnsupportedError(
                    f"{node.label!r} references unavailable Paddle parent {label!r}.",
                    context=self._context(node, "missing parent value"),
                )
            return overlay[node_id]
        if isinstance(component, tuple):
            return tuple(
                self._resolve_component(
                    item,
                    node,
                    overlay,
                    param_cursor=param_cursor,
                    template_key=index,
                )
                for index, item in enumerate(component)
            )
        if isinstance(component, list):
            return [
                self._resolve_component(
                    item,
                    node,
                    overlay,
                    param_cursor=param_cursor,
                    template_key=index,
                )
                for index, item in enumerate(component)
            ]
        if isinstance(component, dict):
            return {
                key: self._resolve_component(
                    value,
                    node,
                    overlay,
                    param_cursor=param_cursor,
                    template_key=key,
                )
                for key, value in component.items()
            }
        return component

    @staticmethod
    def _is_paddle_tensor(value: Any) -> bool:
        """Return whether ``value`` is a Paddle tensor."""

        return isinstance(value, _paddle().Tensor)

    def _rewrite_dynamic_args(
        self,
        node: SplitTraceNode,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        overlay: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Rewrite captured batch literals in Paddle shape-sensitive calls."""

        if self.graph.shape_program is None or self._shape_binding is None:
            return args, kwargs
        return (
            self.graph.shape_program.rewrite(node.canonical_id, args, self._shape_binding),
            self.graph.shape_program.rewrite(node.canonical_id, kwargs, self._shape_binding),
        )

    def _reconstruct_args(
        self,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Reconstruct concrete function args for ``node``."""

        if node.args_template is None:
            raise SplitUnsupportedError(
                f"{node.label!r} has no captured Paddle args_template.",
                context=self._context(node, "missing args_template"),
            )
        param_cursor = _LiveParamCursor(node)
        args = tuple(
            self._resolve_component(
                component,
                node,
                overlay,
                param_cursor=param_cursor,
                template_key=index,
            )
            for index, component in enumerate(node.args_template)
        )
        kwargs = {
            str(key): self._resolve_component(
                component,
                node,
                overlay,
                param_cursor=param_cursor,
                template_key=key,
            )
            for key, component in (node.kwargs_template or {}).items()
        }
        return self._rewrite_dynamic_args(node, args, kwargs, overlay)

    def _source_value(self, node: SplitTraceNode) -> Any:
        """Return a replay value for source-like nodes."""

        value = getattr(node.op, "out", None)
        if value is None:
            raise SplitUnsupportedError(
                f"{node.label!r} source value is unavailable.",
                context=self._context(node, "missing source value"),
            )
        return value

    @staticmethod
    def _is_replay_source_node(node: SplitTraceNode) -> bool:
        """Return whether a target-less node is safe to seed from trace payload."""

        return not node.parents and not node.is_output

    def _execute_func(
        self,
        node: SplitTraceNode,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute one captured Paddle operation function."""

        if node.target is None:
            raise SplitUnsupportedError(
                f"{node.label!r} has no callable target for Paddle split replay.",
                context=self._context(node, "missing callable target"),
            )
        paddle = _paddle()
        if self.spec.trainable:
            with _state.pause_logging():
                return node.target(*args, **kwargs)
        with _state.pause_logging(), paddle.no_grad():
            return node.target(*args, **kwargs)

    def _execute_nodes(self, overlay: dict[str, Any]) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        executed_call_ids: set[int] = set()
        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids:
                continue
            if node.is_input:
                continue
            if node.target is None and self._is_replay_source_node(node):
                if node.canonical_id not in overlay and not node.is_output:
                    overlay[node.canonical_id] = self._source_value(node)
                continue
            if node.func_call_id is not None and node.func_call_id in executed_call_ids:
                continue
            if node.func_call_id is None:
                group = [node]
            else:
                group = [
                    candidate
                    for candidate in self.graph.nodes
                    if candidate.canonical_id in self.node_ids
                    and candidate.func_call_id == node.func_call_id
                    and not candidate.is_input
                    and not candidate.is_output
                ]
                if not group:
                    group = [node]
            executor = next((member for member in group if member.target is not None), None)
            if executor is None:
                raise SplitUnsupportedError(
                    f"{node.label!r} has no callable target for Paddle split replay.",
                    context=self._context(node, "missing callable target"),
                )
            if node.func_call_id is not None:
                executed_call_ids.add(node.func_call_id)
            args, kwargs = self._reconstruct_args(executor, overlay)
            try:
                output = self._execute_func(executor, args, kwargs)
            except Exception as exc:
                raise SplitUnsupportedError(
                    f"Paddle replay failed at {executor.label!r} ({executor.op_type}): {exc}",
                    context=self._context(executor, "backend replay execution failed"),
                ) from exc
            for member in group:
                value = _slice_output_by_path(output, member.output_container_path)
                overlay[member.canonical_id] = value
        return overlay


class PaddleGeneratedPrefix(_PaddleGeneratedSegmentBase):
    """Generated-eager Paddle prefix segment."""

    def __call__(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
        detach_boundary: bool,
    ) -> ReplayBoundary:
        """Run the prefix and return a replay boundary."""

        paddle = _paddle()
        input_leaves = _flatten_tensor_leaves(inputs, paddle)
        input_leaves.extend(_flatten_tensor_leaves(input_kwargs or {}, paddle))
        if len(input_leaves) != len(self.graph.input_node_ids):
            raise SplitUnsupportedError(
                "Runtime inputs do not match traced Paddle tensor input count.",
                context=SplitErrorContext(
                    backend="paddle",
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
                backend="paddle",
                split_point=self.spec.boundary,
            )
        self._execute_nodes(overlay)
        boundary_tensors: dict[str, Any] = {}
        prefix_tensors: dict[str, Any] = {}
        for node_id in self.plan.boundary_node_ids:
            node = self._node_by_id[node_id]
            key = boundary_key_for_node(node.canonical_id, node.output_container_path)
            value = overlay[node_id]
            prefix_tensors[key] = value
            boundary_tensors[key] = value.detach() if detach_boundary else value
        metadata: dict[str, Any] = {
            "split_id": self.plan.split_id,
            "graph_shape_hash": self.graph.graph_shape_hash,
            "batch_symbol": self.spec.batch_symbol,
            "dynamic_batch": self.spec.dynamic_batch,
            "runtime_batch_size": (
                None if self._shape_binding is None else self._shape_binding.batch_size
            ),
            "shape_program_hash": (
                None if self.graph.shape_program is None else self.graph.shape_program.fingerprint
            ),
            "device_policy": self.spec.device_policy,
            "supports_prefix_backward": not detach_boundary,
        }
        if not detach_boundary:
            metadata["prefix_boundary_tensors"] = prefix_tensors
        return ReplayBoundary(
            backend="paddle",
            tensors=boundary_tensors,
            spec=self.plan.boundary_spec,
            metadata=metadata,
        )


class PaddleGeneratedSuffix(_PaddleGeneratedSegmentBase):
    """Generated-eager Paddle suffix segment."""

    def __call__(self, boundary: ReplayBoundary) -> Any:
        """Run the suffix from ``boundary`` and reconstruct final output."""

        overlay = dict(boundary.tensors)
        runtime_batch_size = boundary.metadata.get("runtime_batch_size")
        if self.graph.shape_program is not None and runtime_batch_size is not None:
            self._shape_binding = self.graph.shape_program.binding_from_batch(
                int(runtime_batch_size)
            )
        for key, item in boundary.spec.items():
            node_id = self._label_to_id.get(item.label)
            if node_id is not None and key in boundary.tensors:
                overlay[node_id] = boundary.tensors[key]
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
        leaves = [self._output_leaf(node, overlay) for node in output_nodes]
        if len(leaves) == 1:
            return leaves[0]
        return tuple(leaves)


class PaddleSplitAdapter(SplitPolicyMixin):
    """Paddle split backend adapter."""

    name = "paddle"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_dynamic_batch = True
    allow_callable_target = True

    def is_tensor(self, value: Any) -> bool:
        """Return whether ``value`` is a Paddle tensor."""

        return isinstance(value, _paddle().Tensor)

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
        """Return Paddle autograd flag."""

        stop_gradient = getattr(value, "stop_gradient", None)
        return None if stop_gradient is None else not bool(stop_gradient)

    def detach(self, value: Any) -> Any:
        """Detach tensor values."""

        return value.detach() if hasattr(value, "detach") else value

    def clone(self, value: Any) -> Any:
        """Clone tensor values."""

        paddle = _paddle()
        return paddle.clone(value) if self.is_tensor(value) else value

    def to_device(self, value: Any, device: Any) -> Any:
        """Move tensor values to a device when Paddle exposes the method."""

        if not self.is_tensor(value):
            return value
        device_text = str(device)
        if device_text == "cpu" and hasattr(value, "cpu"):
            return value.cpu()
        if device_text.startswith(("gpu", "cuda")) and hasattr(value, "cuda"):
            return value.cuda()
        return value

    def collate(self, values: list[Any]) -> Any:
        """Stack Paddle tensor values."""

        paddle = _paddle()
        if values and isinstance(values[0], paddle.Tensor):
            return paddle.stack(values)
        return list(values)

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like a tensor."""

        return _paddle().zeros_like(value)

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two tensors are numerically close."""

        paddle = _paddle()
        if isinstance(left, paddle.Tensor) and isinstance(right, paddle.Tensor):
            return bool(paddle.allclose(left, right, atol=atol, rtol=rtol).item())
        return left == right

    def build_segments(
        self,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
    ) -> SegmentBundle:
        """Build Paddle generated-eager prefix/suffix segments."""

        prefix = PaddleGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.prefix_node_ids,
        )
        training_prefix = PaddleGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=(
                spec
                if spec.trainable
                else replace(spec, features=replace(spec.features, training=True))
            ),
            node_ids=plan.prefix_node_ids,
        )
        suffix = PaddleGeneratedSuffix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.suffix_node_ids,
        )
        return SegmentBundle(prefix=prefix, training_prefix=training_prefix, suffix=suffix)


__all__ = ["PaddleGeneratedPrefix", "PaddleGeneratedSuffix", "PaddleSplitAdapter"]
