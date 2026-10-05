"""Paddle generated-eager split replay adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ... import _state
from ...backends.paddle._cuda import paddle_cuda_scope
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import SplitTraceGraph, SplitTraceNode
from ..ir import SplitRequest
from ..planner import SplitPlan
from ..shape_program import ShapeBinding
from ..validation import nested_allclose
from .base import SegmentBundle, SplitPolicyMixin, boundary_overlay


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

    def _resolve_component(
        self,
        component: Any,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> Any:
        """Resolve one captured Paddle template component."""

        if _is_tensor_marker(component):
            label = component.get("label")
            if label is None:
                value = component.get("value")
                if self._is_paddle_tensor(value):
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
            return tuple(self._resolve_component(item, node, overlay) for item in component)
        if isinstance(component, list):
            return [self._resolve_component(item, node, overlay) for item in component]
        if isinstance(component, dict):
            return {
                key: self._resolve_component(value, node, overlay)
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
        args = tuple(
            self._resolve_component(component, node, overlay) for component in node.args_template
        )
        kwargs = {
            str(key): self._resolve_component(component, node, overlay)
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

        templates = tuple(
            component
            for node in self.graph.nodes
            if node.canonical_id in self.node_ids
            for component in (node.args_template, node.kwargs_template)
        )
        with paddle_cuda_scope(overlay, templates):
            return self._execute_nodes_in_context(overlay)

    def _execute_nodes_in_context(self, overlay: dict[str, Any]) -> dict[str, Any]:
        """Execute segment calls with their tensors' owning CUDA context current."""

        executed_call_ids: set[str] = set()
        call_by_output = self.graph.replay_call_by_output_id
        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids:
                continue
            if node.is_input:
                continue
            if node.target is None and self._is_replay_source_node(node):
                if node.canonical_id not in overlay and not node.is_output:
                    overlay[node.canonical_id] = self._source_value(node)
                continue
            call = call_by_output.get(node.canonical_id)
            call_id = node.canonical_id if call is None else call.call_id
            if call_id in executed_call_ids:
                continue
            group = (
                [node]
                if call is None
                else [
                    self._node_by_id[node_id]
                    for node_id in call.output_node_ids
                    if node_id in self.node_ids
                ]
            )
            if not group:
                group = [node]
            executor = next((member for member in group if member.target is not None), None)
            if executor is None:
                raise SplitUnsupportedError(
                    f"{node.label!r} has no callable target for Paddle split replay.",
                    context=self._context(node, "missing callable target"),
                )
            executed_call_ids.add(call_id)
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
                    reason="input count mismatch",
                ),
            )
        overlay = dict(zip(self.graph.input_node_ids, input_leaves))
        if self.graph.shape_program is not None:
            self._shape_binding = self.graph.shape_program.bind_flat_values(
                input_leaves,
                shape_of=lambda value: tuple(int(dim) for dim in value.shape),
                backend="paddle",
                split_point=self.spec.boundary,
            )
            self.graph.shape_program.require_batch_resolvable(
                self._shape_binding.batch_size,
                self.node_ids,
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

        overlay = boundary_overlay(boundary, self.plan)
        runtime_batch_size = boundary.metadata.get("runtime_batch_size")
        if self.graph.shape_program is not None and runtime_batch_size is not None:
            self._shape_binding = self.graph.shape_program.binding_from_batch(
                int(runtime_batch_size)
            )
            self.graph.shape_program.require_batch_resolvable(
                int(runtime_batch_size),
                self.node_ids,
                backend="paddle",
                split_point=self.spec.boundary,
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
    supports_state_placement = False
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
        with paddle_cuda_scope(value):
            return paddle.clone(value) if self.is_tensor(value) else value

    def resize_batch(self, value: Any, axis: int, batch_size: int) -> Any:
        """Select cyclic batch rows on the source Paddle device."""

        if not self.is_tensor(value):
            return value
        current = int(value.shape[axis])
        if current <= 0:
            raise ValueError("Cannot resize an empty batch axis.")
        paddle = _paddle()
        with paddle_cuda_scope(value):
            indexes = paddle.to_tensor(
                [index % current for index in range(batch_size)], dtype="int64", place=value.place
            )
            return paddle.index_select(value, indexes, axis=axis)

    def to_device(self, value: Any, device: Any) -> Any:
        """Move tensor values to a device when Paddle exposes the method."""

        if not self.is_tensor(value):
            return value
        device_text = str(device)
        with paddle_cuda_scope(value):
            if device_text == "cpu" and hasattr(value, "cpu"):
                return value.cpu()
            if device_text.startswith(("gpu", "cuda")) and hasattr(value, "cuda"):
                return value.cuda()
        return value

    def collate(self, values: list[Any]) -> Any:
        """Stack Paddle tensor values."""

        paddle = _paddle()
        if values and isinstance(values[0], paddle.Tensor):
            with paddle_cuda_scope(values):
                return paddle.stack(values)
        return list(values)

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like a tensor."""

        with paddle_cuda_scope(value):
            return _paddle().zeros_like(value)

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two tensors are numerically close."""

        paddle = _paddle()
        if isinstance(left, paddle.Tensor) and isinstance(right, paddle.Tensor):
            with paddle_cuda_scope(left, right):
                return bool(paddle.allclose(left, right, atol=atol, rtol=rtol).item())
        return left == right

    def validate_equivalence(
        self,
        runtime: Any,
        model: Any,
        inputs: tuple[Any, ...],
        **options: Any,
    ) -> bool:
        """Compare native and split outputs while restoring the caller's CUDA context."""

        input_kwargs = options.get("input_kwargs")
        state = model.state_dict() if isinstance(model, _paddle().nn.Layer) else None
        with paddle_cuda_scope(inputs, input_kwargs, state):
            full_output = model(*inputs, **(input_kwargs or {}))
            replay_output = runtime.replay(*inputs, input_kwargs=input_kwargs)
            return nested_allclose(
                self,
                full_output,
                replay_output,
                atol=options.get("atol", 1e-5),
                rtol=options.get("rtol", 1e-4),
            )

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
