"""Torch generated-eager split replay adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...intervention.types import LiteralTensor, LiteralValue, ParentRef, Unsupported
from ...ir.container import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    TupleIndex,
    rebuild_container_from_spec,
)
from ...utils.rng import execute_with_restored_rng_autocast
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import ReplayValueRef, SplitTraceGraph, SplitTraceNode
from ..planner import SplitPlan
from ..shape_program import ShapeBinding, shape_semantic_for_node
from ..ir import SplitRequest
from .base import SegmentBundle, SplitPolicyMixin, boundary_overlay


def _torch() -> Any:
    """Import torch lazily for split adapter operations."""

    import torch

    return torch


def _component_key(component: Any) -> Any:
    """Return an index/key value for a TorchLens container path component."""

    if isinstance(component, TupleIndex):
        return component.index
    if isinstance(component, DictKey):
        return component.key
    if isinstance(component, (NamedField, DataclassField)):
        return component.name
    if isinstance(component, HFKey):
        return component.key
    return component


def _slice_output_by_path(output: Any, path: tuple[Any, ...]) -> Any:
    """Return the output leaf addressed by a captured container path."""

    current = output
    for component in path:
        key = _component_key(component)
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(key, str) and hasattr(current, key):
            current = getattr(current, key)
        else:
            current = current[key]
    return current


def _is_template_dict(component: tuple[Any, ...]) -> bool:
    """Return whether a tuple template component encodes a dictionary."""

    return bool(component) and all(isinstance(item, tuple) and len(item) == 2 for item in component)


def _flatten_tensor_leaves(value: Any, torch: Any) -> list[Any]:
    """Collect tensor leaves in deterministic traversal order."""

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        leaves: list[Any] = []
        for key in sorted(value, key=repr):
            leaves.extend(_flatten_tensor_leaves(value[key], torch))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_flatten_tensor_leaves(item, torch))
        return leaves
    return []


def _dtype_name(value: Any) -> str | None:
    """Return a dtype string for a torch value."""

    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


class _LiveParamCursor:
    """Sequential matcher from literal tensor template leaves to live params."""

    def __init__(self, handles: list[Any]) -> None:
        """Create a cursor over live parameter handles."""

        self.handles = handles
        self.index = 0

    def maybe_replace(self, value: Any) -> Any:
        """Return a matching live parameter/buffer handle for ``value`` if present."""

        torch = _torch()
        if self.index >= len(self.handles) or not isinstance(value, torch.Tensor):
            return value
        for offset, handle in enumerate(self.handles[self.index :], start=self.index):
            if not isinstance(handle, torch.Tensor):
                continue
            if tuple(handle.shape) == tuple(value.shape) and str(handle.dtype) == str(value.dtype):
                self.index = offset + 1
                return handle
        return value


class _GeneratedSegmentBase:
    """Shared generated-eager replay helpers."""

    def __init__(
        self,
        *,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
        node_ids: frozenset[str],
        use_live_param_sources: bool,
    ) -> None:
        """Create a generated replay segment."""

        self.graph = graph
        self.plan = plan
        self.spec = spec
        self.node_ids = node_ids
        self.use_live_param_sources = use_live_param_sources
        self._node_by_id = graph.node_by_id
        self._label_to_id = self._build_label_lookup(graph)
        self._shape_binding: ShapeBinding | None = None

    @staticmethod
    def _build_label_lookup(graph: SplitTraceGraph) -> dict[str, str]:
        """Build raw/final/canonical label lookup table."""

        return graph.node_id_by_alias

    def _context(self, node: SplitTraceNode, reason: str) -> SplitErrorContext:
        """Build an error context for ``node``."""

        return SplitErrorContext(
            backend=self.graph.backend,
            split_point=self.spec.boundary,
            module_path=node.module_path,
            op_type=node.op_type,
            layer_label=node.label,
            reason=reason,
            traced_shape=node.output_shape,
            dtype=node.dtype,
        )

    def _param_handles_for_node(self, node: SplitTraceNode) -> list[Any]:
        """Resolve live parameter handles for a replay node."""

        handles: list[Any] = []
        if not self.use_live_param_sources:
            return handles
        seen_handles: set[int] = set()
        for param_ref in node.param_refs:
            handle = getattr(param_ref, "handle", None)
            if handle is None:
                raise SplitUnsupportedError(
                    f"Cannot resolve live parameter source for {node.label!r}.",
                    context=self._context(node, "missing live parameter source"),
                )
            if id(handle) not in seen_handles:
                handles.append(handle)
                seen_handles.add(id(handle))
        for param_ref in node.param_refs:
            module = getattr(param_ref, "module", None)
            buffers = getattr(module, "buffers", None)
            if buffers is None:
                continue
            for buffer in buffers.values():
                buffer_handle = getattr(buffer, "handle", None)
                if buffer_handle is not None and id(buffer_handle) not in seen_handles:
                    handles.append(buffer_handle)
                    seen_handles.add(id(buffer_handle))
        return handles

    def _resolve_parent_ref(
        self,
        ref: ParentRef | ReplayValueRef,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> Any:
        """Resolve a captured parent reference against the replay overlay."""

        if isinstance(ref, ReplayValueRef):
            parent_id = ref.value_id
            reference = ref.value_id
        else:
            parent_id = None
            reference = ref.parent_label
        if parent_id is None or parent_id not in overlay:
            raise SplitUnsupportedError(
                f"{node.label!r} references unavailable parent {reference!r}.",
                context=self._context(node, "missing parent replay value"),
            )
        return overlay[parent_id]

    def _resolve_component(
        self,
        component: Any,
        node: SplitTraceNode,
        overlay: dict[str, Any],
        *,
        param_cursor: _LiveParamCursor,
    ) -> Any:
        """Resolve one captured argument-template component."""

        if isinstance(component, (ParentRef, ReplayValueRef)):
            return self._resolve_parent_ref(component, node, overlay)
        if isinstance(component, LiteralTensor):
            return param_cursor.maybe_replace(component.value)
        if isinstance(component, LiteralValue):
            return self._rewrite_literal_value(
                component.value,
                node=node,
            )
        if isinstance(component, Unsupported):
            raise SplitUnsupportedError(
                f"{node.label!r} has unsupported replay template component: "
                f"{component.reason} ({component.value_type}).",
                context=self._context(node, "unsupported replay template component"),
            )
        if isinstance(component, tuple):
            if _is_template_dict(component):
                return {
                    key: self._resolve_component(
                        value,
                        node,
                        overlay,
                        param_cursor=param_cursor,
                    )
                    for key, value in component
                }
            return tuple(
                self._resolve_component(
                    value,
                    node,
                    overlay,
                    param_cursor=param_cursor,
                )
                for value in component
            )
        return component

    def _rewrite_literal_value(
        self,
        value: Any,
        *,
        node: SplitTraceNode,
    ) -> Any:
        """Rewrite dynamic-batch shape literals for known shape-sensitive ops."""

        if self.graph.shape_program is None or self._shape_binding is None:
            return value
        return self.graph.shape_program.rewrite(node.canonical_id, value, self._shape_binding)

    def _rewrite_dynamic_call_args(
        self,
        node: SplitTraceNode,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Rewrite Torch calls whose shape is represented as scalar varargs."""

        if self.graph.shape_program is None or self._shape_binding is None:
            return args, kwargs
        rewritten_args = self.graph.shape_program.rewrite(
            node.canonical_id,
            args,
            self._shape_binding,
        )
        rewritten_kwargs = self.graph.shape_program.rewrite(
            node.canonical_id,
            kwargs,
            self._shape_binding,
        )
        if shape_semantic_for_node(node) != "reshape":
            return rewritten_args, rewritten_kwargs
        runtime_shape = self.graph.shape_program.value_shape(
            node.canonical_id,
            self._shape_binding,
        )
        if runtime_shape is None or not rewritten_args:
            return rewritten_args, rewritten_kwargs
        func_id = getattr(node.args_template, "func_id", None)
        namespace = getattr(func_id, "namespace", None)
        qualname = getattr(func_id, "qualname", "").rsplit(".", 1)[-1]
        if qualname == "flatten":
            return rewritten_args, rewritten_kwargs
        if namespace == "torch.Tensor":
            if len(rewritten_args) == 2 and isinstance(rewritten_args[1], (tuple, list)):
                shape_type = type(rewritten_args[1])
                return (rewritten_args[0], shape_type(runtime_shape)), rewritten_kwargs
            return (rewritten_args[0], *runtime_shape), rewritten_kwargs
        if namespace == "torch" and len(rewritten_args) >= 2:
            shape_type = type(rewritten_args[1])
            shape_value = (
                shape_type(runtime_shape)
                if isinstance(rewritten_args[1], (tuple, list))
                else runtime_shape
            )
            return (rewritten_args[0], shape_value, *rewritten_args[2:]), rewritten_kwargs
        return rewritten_args, rewritten_kwargs

    def _reconstruct_args(
        self,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Reconstruct concrete function args for ``node``."""

        template = node.args_template
        if template is None:
            raise SplitUnsupportedError(
                f"{node.label!r} has no captured args_template.",
                context=self._context(node, "missing args_template"),
            )
        param_cursor = _LiveParamCursor(self._param_handles_for_node(node))
        args = tuple(
            self._resolve_component(
                component,
                node,
                overlay,
                param_cursor=param_cursor,
            )
            for component in template.args
        )
        kwargs = {
            key: self._resolve_component(
                component,
                node,
                overlay,
                param_cursor=param_cursor,
            )
            for key, component in template.kwargs
        }
        return self._rewrite_dynamic_call_args(node, args, kwargs)

    def _execute_func(
        self,
        node: SplitTraceNode,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute one captured operation function."""

        if node.target is None:
            raise SplitUnsupportedError(
                f"{node.label!r} has no callable target for split replay.",
                context=self._context(node, "missing callable target"),
            )
        try:
            output = execute_with_restored_rng_autocast(
                node.target,
                args,
                kwargs,
                rng_states=getattr(node.op, "func_rng_states", None),
                autocast_state=getattr(node.op, "func_autocast_state", None),
            )
        except Exception as exc:
            raise SplitUnsupportedError(
                f"Torch replay failed at {node.canonical_id!r} ({node.op_type}): {exc}",
                context=self._context(node, "backend replay execution failed"),
            ) from exc
        if output is None and args:
            return args[0]
        return output

    def _source_value(self, node: SplitTraceNode) -> Any:
        """Return a replay value for an input/buffer/source node."""

        if self.use_live_param_sources and node.buffer_refs:
            return getattr(node.buffer_refs[0], "handle", node.buffer_refs[0])
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

        return node.is_buffer or (not node.parents and not node.is_output)

    def _execute_nodes(self, overlay: dict[str, Any]) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        executed_call_ids: set[str] = set()
        call_by_output = self.graph.replay_call_by_output_id
        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids:
                continue
            if node.is_input:
                continue
            if node.is_output:
                continue
            if node.is_buffer or (node.target is None and self._is_replay_source_node(node)):
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
                    f"{node.label!r} has no callable target for split replay.",
                    context=self._context(node, "missing callable target"),
                )
            executed_call_ids.add(call_id)
            args, kwargs = self._reconstruct_args(executor, overlay)
            output = self._execute_func(executor, args, kwargs)
            for member in group:
                value = _slice_output_by_path(output, member.output_container_path)
                overlay[member.canonical_id] = value
        return overlay


class GeneratedPrefix(_GeneratedSegmentBase):
    """Generated-eager Torch prefix segment."""

    def __call__(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
        detach_boundary: bool,
    ) -> ReplayBoundary:
        """Run the prefix and return a replay boundary."""

        torch = _torch()
        input_leaves = _flatten_tensor_leaves(inputs, torch)
        input_leaves.extend(_flatten_tensor_leaves(input_kwargs or {}, torch))
        if len(input_leaves) != len(self.graph.input_node_ids):
            raise SplitUnsupportedError(
                "Runtime inputs do not match traced tensor input count.",
                context=SplitErrorContext(
                    backend="torch",
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
                backend="torch",
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
            boundary_tensors[key] = (
                value.detach() if detach_boundary and hasattr(value, "detach") else value
            )
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
            for value in prefix_tensors.values():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    value.retain_grad()
            metadata["prefix_boundary_tensors"] = prefix_tensors
        return ReplayBoundary(
            backend="torch",
            tensors=boundary_tensors,
            spec=self.plan.boundary_spec,
            metadata=metadata,
        )


class GeneratedSuffix(_GeneratedSegmentBase):
    """Generated-eager Torch suffix segment."""

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
        """Reconstruct the traced model output container."""

        output_nodes = [self._node_by_id[node_id] for node_id in self.graph.output_node_ids]
        if not output_nodes:
            if not overlay:
                return None
            last_id = next(reversed(overlay))
            return overlay[last_id]
        leaves = [self._output_leaf(node, overlay) for node in output_nodes]
        spec = next(
            (node.output_container_spec for node in output_nodes if node.output_container_spec),
            None,
        )
        if spec is not None:
            return rebuild_container_from_spec(spec, leaves)
        if len(leaves) == 1:
            return leaves[0]
        return tuple(leaves)


class TorchSplitAdapter(SplitPolicyMixin):
    """Torch split backend adapter."""

    name = "torch"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_dynamic_batch = True
    allow_callable_target = True
    native_state_replay = True

    def is_tensor(self, value: Any) -> bool:
        """Return whether ``value`` is a torch tensor."""

        return isinstance(value, _torch().Tensor)

    def shape(self, value: Any) -> tuple[int, ...] | None:
        """Return tensor shape."""

        if not self.is_tensor(value):
            return None
        return tuple(int(dim) for dim in value.shape)

    def dtype_name(self, value: Any) -> str | None:
        """Return tensor dtype name."""

        return _dtype_name(value)

    def requires_grad(self, value: Any) -> bool | None:
        """Return tensor autograd flag."""

        requires_grad = getattr(value, "requires_grad", None)
        return None if requires_grad is None else bool(requires_grad)

    def detach(self, value: Any) -> Any:
        """Detach tensor values."""

        return value.detach() if hasattr(value, "detach") else value

    def clone(self, value: Any) -> Any:
        """Clone tensor values."""

        return value.clone() if hasattr(value, "clone") else value

    def to_device(self, value: Any, device: Any) -> Any:
        """Move tensor values to a device."""

        return value.to(device) if hasattr(value, "to") else value

    def collate(self, values: list[Any]) -> Any:
        """Stack torch tensor values."""

        torch = _torch()
        if values and isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        return list(values)

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like a tensor."""

        return _torch().zeros_like(value)

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two tensors are numerically close."""

        torch = _torch()
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return bool(torch.allclose(left, right, atol=atol, rtol=rtol))
        return left == right

    def resize_batch(self, value: Any, axis: int, batch_size: int) -> Any:
        """Resize a tensor axis by deterministic cyclic selection."""

        torch = _torch()
        if not isinstance(value, torch.Tensor):
            return value
        normalized_axis = axis if axis >= 0 else value.ndim + axis
        current = int(value.shape[normalized_axis])
        if current <= 0:
            raise ValueError("Cannot resize an empty batch axis for shape witnessing.")
        indexes = torch.arange(batch_size, device=value.device) % current
        return value.index_select(normalized_axis, indexes)

    def build_segments(
        self,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
    ) -> SegmentBundle:
        """Build Torch generated-eager prefix/suffix segments."""

        use_live = (
            spec.trainable if spec.use_live_param_sources is None else spec.use_live_param_sources
        )
        prefix = GeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.prefix_node_ids,
            use_live_param_sources=use_live,
        )
        training_spec = replace(
            spec,
            features=replace(spec.features, training=True, live_param_sources=True),
        )
        training_prefix = GeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=training_spec,
            node_ids=plan.prefix_node_ids,
            use_live_param_sources=True,
        )
        suffix = GeneratedSuffix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.suffix_node_ids,
            use_live_param_sources=use_live,
        )
        return SegmentBundle(prefix=prefix, training_prefix=training_prefix, suffix=suffix)


__all__ = [
    "GeneratedPrefix",
    "GeneratedSuffix",
    "SegmentBundle",
    "TorchSplitAdapter",
]
