"""TensorFlow raw-op split replay adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple

from torchlens.split.errors import SplitRequestError

from ..._ir.container import rebuild_container_from_spec
from ...backends.tf._raw_op_replay import (
    _raw_op_dispatch,
    _replay_outputs,
    _replay_raw_op,
    _select_replay_output,
)
from ...backends.tf._tensor_probe import _tensor_ref_key
from ...backends.tf.op_callback_capture import TFInputCapture, TFOpCapture
from .._capture_projection import reorder_container_leaves
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import SplitTraceGraph, SplitTraceNode
from ..ir import SplitRequest
from ..planner import SplitPlan
from ..shape_program import ShapeBinding, _tf_shape_literal_input
from .base import SegmentBundle, SplitPolicyMixin, boundary_overlay


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


class _TfInputBinding(NamedTuple):
    """An ordered native operand and its resolved source."""

    record: TFInputCapture
    parent_label: str | None
    parent_id: str | None
    rewrite_shape: bool


_TfReplayStep = tuple[SplitTraceNode, tuple[_TfInputBinding, ...], tuple[str, ...]]


class _TfReplayBlueprint(NamedTuple):
    """Immutable operand bindings and the captured resource identities they read."""

    steps: tuple[_TfReplayStep, ...]
    resources: tuple[Any, ...]
    resource_keys: frozenset[object]
    model_resource_keys: frozenset[object]


def _tf_variable_resource(variable: Any) -> Any:
    """Get the native resource tensor behind a TensorFlow or Keras variable."""
    return getattr(_tf_gradient_source(variable), "handle", None)


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
        self._blueprint = graph._segment_blueprint(("tf", node_ids), self._prepare_blueprint)

    def _prepare_blueprint(self) -> _TfReplayBlueprint:
        """Prepare execution and resource membership for this segment."""
        steps = self._prepare_execution_steps()
        resources = {
            binding.record.ref_key: binding.record.tensor
            for node, bindings, _outputs in steps
            if isinstance(node.target, TFOpCapture) and node.target.op_type == "ReadVariableOp"
            for binding in bindings
            if binding.record.source_kind == "resource" and binding.record.ref_key is not None
        }
        model_resources = self._captured_model_resources()
        resources.update(model_resources)
        return _TfReplayBlueprint(
            steps, tuple(resources.values()), frozenset(resources), frozenset(model_resources)
        )

    def _prepare_execution_steps(self) -> tuple[_TfReplayStep, ...]:
        """Lower calls in capture order, with only their leader in the schedule."""

        grouped: dict[str, tuple[str, ...]] = {}
        siblings: set[str] = set()
        for call in self.graph._multi_output_calls:
            outputs = call.output_node_ids
            if self.node_ids.isdisjoint(outputs):
                continue
            leader, *followers = outputs
            grouped[leader] = self._prepare_call_outputs(outputs)
            siblings.update(followers)
        scheduled = self.node_ids - siblings if siblings else self.node_ids
        return tuple(
            (node, self._prepare_tf_inputs(node.target), grouped.get(node.canonical_id, ()))
            for node in self.graph.nodes
            if node.canonical_id in scheduled
            and not node.is_input
            and not (node.is_output and node.target is None)
        )

    def _prepare_call_outputs(self, members: tuple[str, ...]) -> tuple[str, ...]:
        """Validate one group's membership and bind its canonical values by slot."""

        leader = self._node_by_id[members[0]]
        captures = [self._node_by_id[member].target for member in members]
        slots = {
            capture.output_index: member
            for capture, member in zip(captures, members, strict=True)
            if isinstance(capture, TFOpCapture)
            and capture.op_type == getattr(leader.target, "op_type", None)
            and capture._output_count == len(members)
        }
        if not self.node_ids.issuperset(members) or slots.keys() != set(range(len(members))):
            raise SplitUnsupportedError(
                f"{leader.label!r} has incomplete or invalid TensorFlow call outputs.",
                context=self._context(leader, "invalid TF call outputs"),
                code=SplitUnsupportedError.code,
            )
        return tuple(slots[index] for index in range(len(members)))

    def _captured_model_resources(self) -> dict[object, Any]:
        """Index native resources from this segment's captured parameter records."""
        model_resources: dict[object, Any] = {}
        for node_id in self.node_ids:
            for param in self._node_by_id[node_id].param_refs:
                resource = _tf_variable_resource(_param_ref_handle(param))
                key = _tensor_ref_key(resource)
                if key is not None:
                    model_resources[key] = resource
        return model_resources

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
                code=SplitUnsupportedError.code,
            )
        return _tf().convert_to_tensor(value)

    def _trainable_variables(self, model: Any) -> list[Any]:
        """Bind segment resources against the model's current trainability."""
        blueprint = self._blueprint
        if not blueprint.resource_keys:
            return []
        bindings: dict[object, Any] = {}
        for variable in getattr(model, "trainable_variables", ()) or ():
            key = _tensor_ref_key(_tf_variable_resource(variable))
            if key in blueprint.resource_keys:
                bindings.setdefault(key, variable)
        missing = blueprint.model_resource_keys - bindings.keys()
        if missing:
            attached = {
                _tensor_ref_key(_tf_variable_resource(variable))
                for variable in getattr(model, "variables", ()) or ()
            }
            if missing - attached:
                raise SplitUnsupportedError(
                    "TensorFlow model resources changed since capture; prepare a new split graph.",
                    context=SplitErrorContext(
                        backend="tf",
                        split_point=self.spec.boundary,
                        reason="captured TensorFlow variable replaced",
                    ),
                    code=SplitUnsupportedError.code,
                )
        return list(bindings.values())

    def _prepare_tf_inputs(self, capture: Any) -> tuple[_TfInputBinding, ...]:
        """Resolve callback operand order, parents and shape-literal eligibility."""
        if not isinstance(capture, TFOpCapture) or not capture.inputs:
            return ()
        records = sorted(capture.inputs, key=lambda item: item.input_index)
        bindings: list[_TfInputBinding] = []
        for record in records:
            label = record.producer_label_raw or record.source_label_raw
            parent_id = self._label_to_id.get(label) if label is not None else None
            rewrite_shape = (
                label is None
                and self.graph.shape_program is not None
                and _tf_shape_literal_input(capture, record.input_index, record.tensor)
                and not self._tf_shape_source_is_state(records[0])
            )
            bindings.append(_TfInputBinding(record, label, parent_id, rewrite_shape))
        return tuple(bindings)

    def _tf_shape_source_is_state(self, first_input: TFInputCapture) -> bool:
        """Keep dimension operands of state-derived values fixed."""
        label = first_input.producer_label_raw or first_input.source_label_raw
        first_id = self._label_to_id.get(label) if label is not None else None
        first = self._node_by_id.get(first_id) if first_id is not None else None
        return first is not None and (
            first.is_buffer
            or first.is_param_source
            or first.replay_source_policy.startswith("live_param")
            or first.op_type.lower() in {"readvariableop", "varhandleop"}
        )

    def _rewrite_tf_literal_tensor(self, tensor: Any, node: SplitTraceNode) -> Any:
        """Rewrite a dimension literal using the current batch binding."""
        shape_program = self.graph.shape_program
        if shape_program is None or self._shape_binding is None:
            return tensor
        try:
            payload = tensor.numpy().tolist()
        except Exception:
            return tensor
        rewritten = shape_program.rewrite(node.canonical_id, payload, self._shape_binding)
        if rewritten == payload:
            return tensor
        return _tf().convert_to_tensor(rewritten, dtype=tensor.dtype)

    def _tf_inputs(
        self,
        bindings: tuple[_TfInputBinding, ...],
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> list[Any]:
        """Rebuild TensorFlow raw-op inputs from replay overlay parents."""

        tf = _tf()
        inputs: list[Any] = []
        for record, label, parent_id, rewrite_shape in bindings:
            if label is not None:
                if parent_id is None or parent_id not in overlay:
                    raise SplitUnsupportedError(
                        f"{node.label!r} references unavailable TensorFlow parent {label!r}.",
                        context=self._context(node, "missing parent value"),
                        code=SplitUnsupportedError.code,
                    )
                value = overlay[parent_id]
            elif record.tensor is not None:
                value = record.tensor
            else:
                raise SplitUnsupportedError(
                    f"{node.label!r} has unresolved TensorFlow input slot {record.input_index}.",
                    context=self._context(node, "unresolved TF input"),
                    code=SplitUnsupportedError.code,
                )
            tensor = value if isinstance(value, tf.Tensor) else tf.convert_to_tensor(value)
            inputs.append(
                self._rewrite_tf_literal_tensor(tensor, node) if rewrite_shape else tensor
            )
        return inputs

    def _execute_tf_capture(
        self,
        node: SplitTraceNode,
        bindings: tuple[_TfInputBinding, ...],
        output_ids: tuple[str, ...],
        overlay: dict[str, Any],
    ) -> None:
        """Execute one native op and publish its outputs to the current overlay."""

        capture = node.target
        if not isinstance(capture, TFOpCapture):
            raise SplitUnsupportedError(
                f"{node.label!r} has no TensorFlow op capture.",
                context=self._context(node, "missing TF op capture"),
                code=SplitUnsupportedError.code,
            )
        try:
            output = _replay_raw_op(capture, self._tf_inputs(bindings, node, overlay))
            if output_ids:
                overlay.update(zip(output_ids, _replay_outputs(capture, output), strict=True))
            else:
                if capture._output_count != 1:
                    raise ValueError(f"{capture.op_type} is missing sibling outputs")
                overlay[node.canonical_id] = _select_replay_output(capture, output)
        except Exception as exc:
            raise SplitUnsupportedError(
                f"TensorFlow split replay failed for {node.label!r}: {exc}",
                context=self._context(node, "TF raw-op replay failed"),
                code=SplitUnsupportedError.code,
            ) from exc

    def _execute_nodes(self, overlay: dict[str, Any]) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        for node, bindings, output_ids in self._blueprint.steps:
            if node.target is None and self._is_replay_source_node(node):
                if node.canonical_id not in overlay:
                    overlay[node.canonical_id] = self._source_value(node)
                continue
            self._execute_tf_capture(node, bindings, output_ids, overlay)
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
                    reason="input count mismatch",
                ),
                code=SplitUnsupportedError.code,
            )
        if self.graph.shape_program is not None:
            self._shape_binding = self.graph.shape_program.bind_flat_values(
                input_leaves,
                shape_of=lambda value: tuple(int(dim) for dim in value.shape),
                backend="tf",
                split_point=self.spec.boundary,
            )
            self.graph.shape_program.require_batch_resolvable(
                self._shape_binding.batch_size,
                self.node_ids,
                backend="tf",
                split_point=self.spec.boundary,
            )
        tape = None
        if detach_boundary:
            overlay = dict(zip(self.graph.input_node_ids, input_leaves))
            self._execute_nodes(overlay)
        else:
            tape = tf.GradientTape(persistent=True)
            with tape:
                for value in input_leaves:
                    if _is_diff_tf_tensor(value, tf):
                        tape.watch(value)
                tape.watch(self._blueprint.resources)
                overlay = dict(zip(self.graph.input_node_ids, input_leaves))
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

        overlay = boundary_overlay(boundary, self.plan)
        runtime_batch_size = boundary.metadata.get("runtime_batch_size")
        if self.graph.shape_program is not None and runtime_batch_size is not None:
            self._shape_binding = self.graph.shape_program.binding_from_batch(
                int(runtime_batch_size)
            )
            self.graph.shape_program.require_batch_resolvable(
                int(runtime_batch_size),
                self.node_ids,
                backend="tf",
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
        value = getattr(node.op, "out", None)
        return _tf().convert_to_tensor(value) if value is not None else None

    def _reconstruct_output(self, overlay: dict[str, Any]) -> Any:
        """Reconstruct the traced model output value."""

        output_nodes = [self._node_by_id[node_id] for node_id in self.graph.output_node_ids]
        if not output_nodes:
            if not overlay:
                return None
            return overlay[next(reversed(overlay))]
        leaves = [
            (path, self._output_leaf(node, overlay))
            for node in output_nodes
            for path in (node.output_container_paths or (node.output_container_path,))
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
        return tuple(value for _path, value in leaves)


class TfSplitAdapter(SplitPolicyMixin):
    """TensorFlow split backend adapter."""

    name = "tf"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_state_placement = False
    native_target_types = frozenset({"TFOpCapture"})

    def target_support_reasons(
        self, node: SplitTraceNode, graph: SplitTraceGraph
    ) -> tuple[str, ...]:
        """Require a registered raw-op executor for a captured TensorFlow call."""

        capture = node.target
        if isinstance(capture, TFOpCapture) and capture.op_type not in _raw_op_dispatch():
            return (f"TensorFlow raw op {capture.op_type!r} has no replay implementation",)
        return super().target_support_reasons(node, graph)

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

    def resize_batch(self, value: Any, axis: int, batch_size: int) -> Any:
        """Select cyclic batch rows on the source TensorFlow device."""

        if not self.is_tensor(value):
            return value
        current = int(value.shape[axis])
        if current <= 0:
            raise SplitRequestError(
                "Cannot resize an empty batch axis.",
                code=SplitRequestError.code,
            )
        tf = _tf()
        with tf.device(value.device):
            indexes = tf.range(batch_size) % current
            return tf.gather(value, indexes, axis=axis)

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


__all__ = ("TfGeneratedPrefix", "TfGeneratedSuffix", "TfSplitAdapter")
