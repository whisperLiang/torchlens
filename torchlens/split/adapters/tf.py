"""TensorFlow raw-op split replay adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...backends.tf.op_callback_capture import TFOpCapture
from ...backends.tf.validation import _replay_raw_op
from ...ir.container import rebuild_container_from_spec, reorder_container_leaves
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import SplitTraceGraph, SplitTraceNode
from ..planner import SplitPlan
from ..shape_program import ShapeBinding
from ..ir import SplitRequest
from .base import SegmentBundle, SplitPolicyMixin


def _tf() -> Any:
    """Import TensorFlow lazily for split adapter operations."""

    import tensorflow as tf

    return tf


def _flatten_tensor_leaves(value: Any, tf: Any) -> list[Any]:
    """Collect TensorFlow tensor leaves in deterministic traversal order."""

    if isinstance(value, (tf.Tensor, tf.Variable)):
        return [value]
    if isinstance(value, dict):
        leaves: list[Any] = []
        for key in sorted(value, key=repr):
            leaves.extend(_flatten_tensor_leaves(value[key], tf))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_flatten_tensor_leaves(item, tf))
        return leaves
    return []


def _is_diff_tf_tensor(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a differentiable TensorFlow tensor."""

    if not isinstance(value, (tf.Tensor, tf.Variable)):
        return False
    dtype = getattr(value, "dtype", None)
    return bool(getattr(dtype, "is_floating", False) or getattr(dtype, "is_complex", False))


def _param_ref_handle(param: Any) -> Any:
    """Return the live backend handle for a captured TensorFlow parameter."""

    handle = getattr(param, "_param_ref", None)
    if handle is None:
        handle = getattr(param, "handle", None)
    return handle


def _tf_gradient_source(value: Any) -> Any:
    """Return a TensorFlow object suitable for GradientTape watch/gradient."""

    tf = _tf()
    if isinstance(value, (tf.Tensor, tf.Variable)):
        return value
    keras_value = getattr(value, "value", None)
    if isinstance(keras_value, (tf.Tensor, tf.Variable)):
        return keras_value
    return value


class _TfGeneratedSegmentBase:
    """Shared TensorFlow raw-op replay helpers."""

    def __init__(
        self,
        *,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
        node_ids: frozenset[str],
    ) -> None:
        """Create a generated TensorFlow replay segment."""

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
            backend="tf",
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
        return _tf().convert_to_tensor(value)

    def _trainable_param_handles(self, node_ids: frozenset[str]) -> list[Any]:
        """Return unique live trainable parameter handles used by ``node_ids``."""

        handles: list[Any] = []
        seen: set[int] = set()
        for node_id in node_ids:
            node = self._node_by_id.get(node_id)
            if node is None:
                continue
            for param in node.param_refs:
                if not getattr(param, "is_trainable", False):
                    continue
                handle = _param_ref_handle(param)
                if handle is None or id(handle) in seen:
                    continue
                seen.add(id(handle))
                handles.append(handle)
        return handles

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
                f"{node.label!r} references unavailable TensorFlow parent {parent_label!r}.",
                context=self._context(node, "missing parent value"),
            )
        return overlay[parent_id]

    def _rewrite_tf_literal_tensor(
        self,
        value: Any,
        *,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> Any:
        """Rewrite captured TF shape tensors for dynamic-batch replay."""

        tf = _tf()
        tensor = tf.convert_to_tensor(value)
        capture = node.target
        if isinstance(capture, TFOpCapture) and capture.inputs:
            first_input = min(capture.inputs, key=lambda item: item.input_index)
            parent_label = first_input.producer_label_raw or first_input.source_label_raw
            parent_id = self._label_to_id.get(parent_label) if parent_label is not None else None
            parent = self._node_by_id.get(parent_id) if parent_id is not None else None
            if parent is not None and (
                parent.is_buffer
                or parent.is_param_source
                or parent.replay_source_policy.startswith("live_param")
                or parent.op_type.lower() in {"readvariableop", "varhandleop"}
            ):
                return tensor
        if self.graph.shape_program is None or self._shape_binding is None:
            return tensor
        try:
            payload = tensor.numpy().tolist()
        except Exception:
            return tensor
        rewritten = self.graph.shape_program.rewrite(
            node.canonical_id,
            payload,
            self._shape_binding,
        )
        if rewritten == payload:
            return tensor
        return tf.convert_to_tensor(rewritten, dtype=tensor.dtype)

    def _tf_inputs(
        self,
        capture: TFOpCapture,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> list[Any]:
        """Rebuild TensorFlow raw-op inputs from replay overlay parents."""

        tf = _tf()
        inputs: list[Any] = []
        for input_record in sorted(capture.inputs, key=lambda item: item.input_index):
            label = input_record.producer_label_raw or input_record.source_label_raw
            if label is not None:
                inputs.append(
                    tf.convert_to_tensor(self._resolve_parent_value(label, node, overlay))
                )
                continue
            if input_record.tensor is not None:
                inputs.append(
                    self._rewrite_tf_literal_tensor(
                        input_record.tensor,
                        node=node,
                        overlay=overlay,
                    )
                )
                continue
            raise SplitUnsupportedError(
                f"{node.label!r} has unresolved TensorFlow input slot {input_record.input_index}.",
                context=self._context(node, "unresolved TF input"),
            )
        return inputs

    def _execute_tf_capture(self, node: SplitTraceNode, overlay: dict[str, Any]) -> Any:
        """Execute one captured TensorFlow raw op."""

        capture = node.target
        if not isinstance(capture, TFOpCapture):
            raise SplitUnsupportedError(
                f"{node.label!r} has no TensorFlow op capture.",
                context=self._context(node, "missing TF op capture"),
            )
        try:
            output = _replay_raw_op(capture, self._tf_inputs(capture, node, overlay))
        except Exception as exc:
            raise SplitUnsupportedError(
                f"TensorFlow split replay failed for {node.label!r}: {exc}",
                context=self._context(node, "TF raw-op replay failed"),
            ) from exc
        if isinstance(output, (tuple, list)):
            return output[capture.output_index]
        return output

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
            overlay[node.canonical_id] = self._execute_tf_capture(node, overlay)
        return overlay


class TfGeneratedPrefix(_TfGeneratedSegmentBase):
    """Generated-eager TensorFlow prefix segment."""

    def __call__(
        self,
        *inputs: Any,
        input_kwargs: dict[str, Any] | None = None,
        detach_boundary: bool,
    ) -> ReplayBoundary:
        """Run the prefix and return a replay boundary."""

        tf = _tf()
        input_leaves = _flatten_tensor_leaves(inputs, tf)
        input_leaves.extend(_flatten_tensor_leaves(input_kwargs or {}, tf))
        if len(input_leaves) != len(self.graph.input_node_ids):
            raise SplitUnsupportedError(
                "Runtime inputs do not match traced TensorFlow tensor input count.",
                context=SplitErrorContext(
                    backend="tf",
                    split_point=self.spec.boundary,
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="input count mismatch",
                ),
            )
        if self.graph.shape_program is not None:
            self._shape_binding = self.graph.shape_program.bind_flat_values(
                input_leaves,
                shape_of=lambda value: tuple(int(dim) for dim in value.shape),
                backend="tf",
                split_point=self.spec.boundary,
            )
        tape = None
        if detach_boundary:
            overlay = {
                node_id: value for node_id, value in zip(self.graph.input_node_ids, input_leaves)
            }
            self._execute_nodes(overlay)
        else:
            tape = tf.GradientTape(persistent=True)
            with tape:
                for value in input_leaves:
                    if _is_diff_tf_tensor(value, tf):
                        tape.watch(value)
                for value in self._trainable_param_handles(self.plan.prefix_node_ids):
                    tape.watch(_tf_gradient_source(value))
                overlay = {
                    node_id: value
                    for node_id, value in zip(self.graph.input_node_ids, input_leaves)
                }
                self._execute_nodes(overlay)
        boundary_tensors: dict[str, Any] = {}
        prefix_tensors: dict[str, Any] = {}
        for node_id in self.plan.boundary_node_ids:
            node = self._node_by_id[node_id]
            key = boundary_key_for_node(node.canonical_id, node.output_container_path)
            value = overlay[node_id]
            prefix_tensors[key] = value
            boundary_tensors[key] = tf.stop_gradient(value) if detach_boundary else value
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
            metadata["tf_tape"] = tape
        return ReplayBoundary(
            backend="tf",
            tensors=boundary_tensors,
            spec=self.plan.boundary_spec,
            metadata=metadata,
        )


class TfGeneratedSuffix(_TfGeneratedSegmentBase):
    """Generated-eager TensorFlow suffix segment."""

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
        value = getattr(node.op, "out", None)
        return _tf().convert_to_tensor(value) if value is not None else None

    def _reconstruct_output(self, overlay: dict[str, Any]) -> Any:
        """Reconstruct the traced model output value."""

        output_nodes = [self._node_by_id[node_id] for node_id in self.graph.output_node_ids]
        if not output_nodes:
            if not overlay:
                return None
            return overlay[next(reversed(overlay))]
        leaves = [(node.output_container_path, self._output_leaf(node, overlay)) for node in output_nodes]
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
        return tuple(value for _path, value in leaves)


class TfSplitAdapter(SplitPolicyMixin):
    """TensorFlow split backend adapter."""

    name = "tf"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_dynamic_batch = True
    native_target_types = frozenset({"TFOpCapture"})

    def is_tensor(self, value: Any) -> bool:
        """Return whether ``value`` is a TensorFlow tensor."""

        tf = _tf()
        return isinstance(value, (tf.Tensor, tf.Variable))

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
        """Return no eager autograd flag for TensorFlow tensors."""

        del value
        return None

    def detach(self, value: Any) -> Any:
        """Detach tensor values with ``tf.stop_gradient``."""

        return _tf().stop_gradient(value) if self.is_tensor(value) else value

    def clone(self, value: Any) -> Any:
        """Clone tensor values."""

        return _tf().identity(value) if self.is_tensor(value) else value

    def to_device(self, value: Any, device: Any) -> Any:
        """Move tensor values to a TensorFlow device when possible."""

        if not self.is_tensor(value) or device is None:
            return value
        with _tf().device(str(device)):
            return _tf().identity(value)

    def collate(self, values: list[Any]) -> Any:
        """Stack TensorFlow tensor values."""

        if values and self.is_tensor(values[0]):
            return _tf().stack(values)
        return list(values)

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like a tensor."""

        return _tf().zeros_like(value)

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two tensors are numerically close."""

        tf = _tf()
        if self.is_tensor(left) and self.is_tensor(right):
            return bool(tf.reduce_all(tf.abs(left - right) <= atol + rtol * tf.abs(right)).numpy())
        return left == right

    def build_segments(
        self,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
    ) -> SegmentBundle:
        """Build TensorFlow raw-op replay prefix/suffix segments."""

        prefix = TfGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.prefix_node_ids,
        )
        training_prefix = TfGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=(
                spec
                if spec.trainable
                else replace(spec, features=replace(spec.features, training=True))
            ),
            node_ids=plan.prefix_node_ids,
        )
        suffix = TfGeneratedSuffix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.suffix_node_ids,
        )
        return SegmentBundle(prefix=prefix, training_prefix=training_prefix, suffix=suffix)


__all__ = ["TfGeneratedPrefix", "TfGeneratedSuffix", "TfSplitAdapter"]
