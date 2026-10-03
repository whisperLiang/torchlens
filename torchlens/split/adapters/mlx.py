"""Generated-eager MLX split replay adapter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from ... import _state
from ...backends.mlx.containers import iter_arrays_with_paths, rebuild_mlx_module
from ...backends.mlx.validation import REPLAY_SLOT, MLXOpCapture
from ...ir.container import rebuild_container_from_spec, reorder_container_leaves
from ..boundary import ReplayBoundary
from ..errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import SplitTraceGraph, SplitTraceNode
from ..ir import SplitRequest
from ..placement import SegmentName
from ..planner import SplitPlan
from ..shape_program import ShapeBinding
from ._mlx_coordination import MlxStateCoordinator
from ._mlx_devices import mlx_execution_context, resolve_mlx_device
from ._mlx_shapes import mlx_shape_templates
from ._mlx_state import MlxStateBinding
from .base import SegmentBundle, SplitPolicyMixin, boundary_overlay


def _mx() -> Any:
    """Import MLX lazily for split replay operations."""

    import mlx.core as mx

    return mx


def _is_mlx_array(value: Any) -> bool:
    """Return whether ``value`` is an MLX array."""

    return isinstance(value, _mx().array)


def _flatten_tensor_leaves(value: Any) -> list[Any]:
    """Collect MLX array leaves in deterministic traversal order."""

    if _is_mlx_array(value):
        return [value]
    if isinstance(value, dict):
        dict_leaves: list[Any] = []
        for key in sorted(value, key=repr):
            dict_leaves.extend(_flatten_tensor_leaves(value[key]))
        return dict_leaves
    if isinstance(value, (list, tuple)):
        sequence_leaves: list[Any] = []
        for item in value:
            sequence_leaves.extend(_flatten_tensor_leaves(item))
        return sequence_leaves
    return []


def _snapshot_inputs(value: Any, memo: dict[int, Any]) -> Any:
    """Retain detached input values for functional prefix recomputation, preserving aliases."""

    if _is_mlx_array(value):
        if id(value) not in memo:
            memo[id(value)] = _mx().stop_gradient(value).astype(value.dtype)
        return memo[id(value)]
    if isinstance(value, dict):
        return {key: _snapshot_inputs(item, memo) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot_inputs(item, memo) for item in value)
    return value


class _MlxGeneratedSegmentBase:
    """Shared generated-eager MLX replay helpers."""

    def __init__(
        self,
        *,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
        segment: SegmentName,
        adapter: Any,
    ) -> None:
        """Create a generated MLX replay segment."""

        self.graph = graph
        self.plan = plan
        self.spec = spec
        self.node_ids = plan.prefix_node_ids if segment == "prefix" else plan.suffix_node_ids
        self.placement = spec.placement.for_segment(segment)
        self._binding = MlxStateBinding(
            graph=graph,
            node_ids=self.node_ids,
            adapter=adapter,
            placement=self.placement,
            own_parameters=spec.trainable,
        )
        self._state = self._binding.state
        self._backward_token = object()
        self._node_by_id = graph.node_by_id
        self._label_to_id = graph.node_id_by_alias
        self._shape_binding: ShapeBinding | None = None
        input_index_by_identity: dict[int, int] = {}
        self._input_aliases: list[tuple[int, int]] = []
        for index, node_id in enumerate(graph.input_node_ids):
            value = getattr(self._node_by_id[node_id].op, "out", None)
            if value is not None:
                previous = input_index_by_identity.setdefault(id(value), index)
                if previous != index:
                    self._input_aliases.append((index, previous))

    def _context(self, node: SplitTraceNode, reason: str) -> SplitErrorContext:
        """Build an error context for ``node``."""

        return SplitErrorContext(
            backend="mlx",
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
        """Return a replay value for a source-like node."""

        value = getattr(node.op, "out", None)
        if value is None:
            raise SplitUnsupportedError(
                f"{node.label!r} source value is unavailable.",
                context=self._context(node, "missing source value"),
            )
        return self._binding.resolve(value)

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
                f"{node.label!r} references unavailable MLX parent {parent_label!r}.",
                context=self._context(node, "missing parent value"),
            )
        return overlay[parent_id]

    def _resolve_template_component(
        self,
        component: Any,
        labels: Any,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> Any:
        """Resolve replay slots in one captured MLX argument tree."""

        if component is REPLAY_SLOT or _is_mlx_array(component):
            label = next(labels, None)
            if label is None:
                if component is REPLAY_SLOT:
                    raise SplitUnsupportedError(
                        f"{node.label!r} has an unlabeled MLX replay slot.",
                        context=self._context(node, "unlabeled replay slot"),
                    )
                return self._binding.resolve(component)
            return self._resolve_parent_value(str(label), node, overlay)
        if isinstance(component, tuple):
            return tuple(
                self._resolve_template_component(item, labels, node, overlay) for item in component
            )
        if isinstance(component, list):
            return [
                self._resolve_template_component(item, labels, node, overlay) for item in component
            ]
        if isinstance(component, dict):
            return {
                key: self._resolve_template_component(item, labels, node, overlay)
                for key, item in component.items()
            }
        return component

    def _reconstruct_args(
        self,
        node: SplitTraceNode,
        overlay: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Reconstruct concrete MLX call arguments from captured templates."""

        capture = node.target
        if not isinstance(capture, MLXOpCapture):
            raise SplitUnsupportedError(
                f"{node.label!r} has no MLX operation capture.",
                context=self._context(node, "missing MLX operation capture"),
            )
        args_template, kwargs_template = mlx_shape_templates(
            capture, node.canonical_id, self.graph.shape_program, self._shape_binding
        )
        args = tuple(
            self._resolve_template_component(
                component,
                iter(
                    capture.arg_leaf_labels[index] if index < len(capture.arg_leaf_labels) else ()
                ),
                node,
                overlay,
            )
            for index, component in enumerate(args_template)
        )
        kwargs = {
            key: self._resolve_template_component(
                component,
                iter(capture.kwarg_leaf_labels.get(key, ())),
                node,
                overlay,
            )
            for key, component in kwargs_template.items()
        }
        if capture.module_ref is not None and args and isinstance(args[0], dict):
            args = (rebuild_mlx_module(capture.module_ref, args[0]), *args[1:])
        if self.placement.is_explicit and str(getattr(capture.func, "__module__", "")).startswith(
            "mlx.core"
        ):
            # Explicit core streams in the capture must not override the
            # runtime's declared placement. Native modules inherit the scope.
            kwargs["stream"] = (
                self.placement.device
                if isinstance(self.placement.device, _mx().Stream)
                else resolve_mlx_device(self.placement.device, split_point=self.spec.boundary)
            )
        return args, kwargs

    def _execute_capture(self, node: SplitTraceNode, overlay: dict[str, Any]) -> Any:
        """Execute one captured MLX operation with replayed parent values."""

        capture = node.target
        if not isinstance(capture, MLXOpCapture) or not callable(capture.func):
            raise SplitUnsupportedError(
                f"{node.label!r} has no callable MLX operation capture.",
                context=self._context(node, "missing callable target"),
            )
        args, kwargs = self._reconstruct_args(node, overlay)
        module_arrays = self._binding.module_arrays(capture, args)
        try:
            with (
                _state.pause_logging(),
                mlx_execution_context(self.placement.device, split_point=self.spec.boundary),
                self._binding.random_call(capture),
            ):
                output = capture.func(*args, **kwargs)
            leaves = _flatten_tensor_leaves(output)
            if leaves:
                _mx().eval(*leaves)
            if module_arrays:
                self._binding.commit_module(module_arrays, args[0])
            return output
        except Exception as exc:
            raise SplitUnsupportedError(
                f"MLX split replay failed for {node.label!r}: {exc}",
                context=self._context(node, "MLX operation replay failed"),
            ) from exc

    def _execute_nodes(self, overlay: dict[str, Any]) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        executed_call_ids: set[str] = set()
        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids or node.is_input:
                continue
            if node.target is None and self._is_replay_source_node(node):
                if node.canonical_id not in overlay and not node.is_output:
                    overlay[node.canonical_id] = self._source_value(node)
                continue
            if node.is_output and node.target is None:
                continue
            capture = node.target
            if not isinstance(capture, MLXOpCapture) or not capture.labels_raw:
                raise SplitUnsupportedError(
                    f"{node.label!r} has no executable MLX operation capture.",
                    context=self._context(node, "missing MLX operation capture"),
                )
            # Raw output labels group the entire native call, including leaves
            # returned directly from the model. Model output paths can differ
            # from the native call's paths, so bind by captured leaf order.
            call_id = capture.labels_raw[0]
            if call_id in executed_call_ids:
                continue
            executed_call_ids.add(call_id)
            output = self._execute_capture(node, overlay)
            self._bind_call_outputs(node, capture, output, overlay)
        return overlay

    def _bind_call_outputs(
        self, node: SplitTraceNode, capture: MLXOpCapture, output: Any, overlay: dict[str, Any]
    ) -> None:
        """Validate native output arity before binding captured array leaves by identity."""

        output_leaves = iter_arrays_with_paths(output, _is_mlx_array)
        if len(output_leaves) != len(capture.labels_raw):
            raise SplitUnsupportedError(
                f"{node.label!r} returned {len(output_leaves)} MLX array leaves, "
                f"but the trace records {len(capture.labels_raw)} outputs.",
                context=self._context(node, "output arity mismatch"),
            )
        for label, (value, _path) in zip(capture.labels_raw, output_leaves, strict=True):
            node_id = self._label_to_id.get(label)
            if node_id in self.node_ids:
                overlay[node_id] = value

    def bound_state_values(self) -> dict[str, Any]:
        """Return effective state by captured occurrence for migration and recutting."""

        return self._binding.bound_values()

    def trainable_parameters(self) -> list[Any]:
        """Return the segment's unique, consumed trainable parameter arrays."""

        return list(self._binding.parameters().values())


class MlxGeneratedPrefix(_MlxGeneratedSegmentBase):
    """Generated-eager MLX prefix segment."""

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
                "Runtime inputs do not match traced MLX tensor input count.",
                context=SplitErrorContext(
                    backend="mlx",
                    split_point=self.spec.boundary,
                    reason="input count mismatch",
                ),
            )
        for index, previous in self._input_aliases:
            if input_leaves[index] is not input_leaves[previous]:
                raise SplitBoundaryError(
                    "MLX replay inputs must preserve array identities shared during capture.",
                    context=SplitErrorContext(
                        backend="mlx",
                        split_point=self.spec.boundary,
                        reason="input alias mismatch",
                    ),
                )
        overlay = dict(zip(self.graph.input_node_ids, input_leaves, strict=True))
        if self.graph.shape_program is not None:
            self._shape_binding = self.graph.shape_program.bind_flat_values(
                input_leaves,
                shape_of=lambda value: tuple(int(dim) for dim in value.shape),
                backend="mlx",
                split_point=self.spec.boundary,
            )
            self.graph.shape_program.require_batch_resolvable(
                self._shape_binding.batch_size,
                self.node_ids,
                backend="mlx",
                split_point=self.spec.boundary,
            )
        with mlx_execution_context(self.placement.device):
            saved_before = self._binding.begin_forward() if not detach_boundary else {}
        self._execute_nodes(overlay)
        boundary_tensors: dict[str, Any] = {}
        for node_id in self.plan.boundary_node_ids:
            node = self._node_by_id[node_id]
            key = boundary_key_for_node(node.canonical_id, node.output_container_path)
            value = overlay[node_id]
            boundary_tensors[key] = self._detach(value) if detach_boundary else value
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
        if not detach_boundary and not self._binding.is_replaying:
            snapshots: dict[int, Any] = {}
            with mlx_execution_context(self.placement.device):
                saved_inputs = _snapshot_inputs(inputs, snapshots)
                saved_kwargs = _snapshot_inputs(input_kwargs or {}, snapshots)
                saved_state = {
                    entry.source_id: _snapshot_inputs(entry.value, snapshots)
                    for entry in self._state.entries()
                }
                _mx().eval(*snapshots.values())
            metadata.update(
                prefix_inputs=saved_inputs,
                prefix_input_kwargs=saved_kwargs,
                mlx_prefix_owner=self._backward_token,
                mlx_prefix_state_version=self._state.version,
                mlx_prefix_state_values=saved_state,
                mlx_prefix_replay_state=saved_before,
                mlx_prefix_rng=dict(self._binding.rng_journal),
            )
        return ReplayBoundary(
            backend="mlx",
            tensors=boundary_tensors,
            spec=self.plan.boundary_spec,
            metadata=metadata,
        )

    @staticmethod
    def _detach(value: Any) -> Any:
        """Return an MLX boundary value detached from any AD graph."""

        return _mx().stop_gradient(value) if _is_mlx_array(value) else value


class MlxGeneratedSuffix(_MlxGeneratedSegmentBase):
    """Generated-eager MLX suffix segment."""

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
                backend="mlx",
                split_point=self.spec.boundary,
            )
        self._execute_nodes(overlay)
        return self._reconstruct_output(overlay)

    def _output_leaf(self, node: SplitTraceNode, overlay: dict[str, Any]) -> Any:
        """Return one final-output leaf value."""

        if node.canonical_id in overlay:
            return overlay[node.canonical_id]
        if node.target is None:
            if not node.parents:
                return self._source_value(node)
            if len(node.parents) == 1:
                parent_id = self._label_to_id.get(node.parents[0])
                if parent_id in overlay:
                    return overlay[parent_id]
        raise SplitUnsupportedError(
            f"Final output {node.canonical_id!r} has no available MLX replay value.",
            context=self._context(node, "missing output replay value"),
        )

    def _reconstruct_output(self, overlay: dict[str, Any]) -> Any:
        """Reconstruct the captured MLX model output container."""

        output_nodes = [self._node_by_id[node_id] for node_id in self.graph.output_node_ids]
        if not output_nodes:
            raise SplitUnsupportedError(
                "MLX split replay requires explicitly recorded final outputs.",
                context=SplitErrorContext(
                    backend="mlx",
                    split_point=self.spec.boundary,
                    reason="missing output records",
                ),
            )
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
            try:
                return rebuild_container_from_spec(spec, reorder_container_leaves(spec, leaves))
            except ValueError as exc:
                raise SplitUnsupportedError(
                    "MLX output records do not match the recorded container specification.",
                    context=self._context(output_nodes[0], "invalid output container records"),
                ) from exc
        if len(leaves) == 1 and not leaves[0][0]:
            return leaves[0][1]
        raise SplitUnsupportedError(
            "MLX split replay requires captured metadata to reconstruct output containers.",
            context=self._context(output_nodes[0], "missing output container metadata"),
        )


class MlxSplitAdapter(SplitPolicyMixin):
    """MLX generated-eager split replay adapter."""

    name = "mlx"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_state_placement = True
    native_target_types = frozenset({"MLXOpCapture"})
    native_state_replay = True

    def batch_probe_sizes(self, traced_batch: int) -> tuple[int, ...]:
        """Probe B=3 when native train-mode BatchNorm requires a B=2 capture."""

        if traced_batch == 1:
            return (2,)
        return (3,) if traced_batch == 2 else ()

    def validate_equivalence(self, runtime: Any, model: Any, inputs: Any, **options: Any) -> bool:
        """Use identical native random draws and restore running state after comparison."""

        from .._mlx_validation import validate_mlx_equivalence

        return validate_mlx_equivalence(runtime, model, inputs, **options)

    @contextmanager
    def batch_probe_scope(self, segments: SegmentBundle) -> Iterator[None]:
        """Match each random call to the independent native capture during the probe."""

        with ExitStack() as stack:
            for segment in (segments.prefix, segments.suffix):
                binding = segment._binding
                rng = {
                    capture.labels_raw[0]: capture.rng_state
                    for node in segment.graph.nodes
                    if isinstance(capture := node.target, MLXOpCapture)
                    and capture.rng_state is not None
                }
                stack.enter_context(binding.using_random_journal(rng))
            yield

    def is_tensor(self, value: Any) -> bool:
        """Return whether ``value`` is an MLX array."""

        return _is_mlx_array(value)

    def shape(self, value: Any) -> tuple[int, ...] | None:
        """Return an MLX array shape."""

        if not self.is_tensor(value):
            return None
        return tuple(int(dim) for dim in value.shape)

    def dtype_name(self, value: Any) -> str | None:
        """Return an MLX dtype name."""

        dtype = getattr(value, "dtype", None)
        return None if dtype is None else str(dtype)

    def requires_grad(self, value: Any) -> bool | None:
        """Return no eager autograd flag for MLX arrays."""

        del value
        return None

    def detach(self, value: Any) -> Any:
        """Detach an MLX array from automatic differentiation."""

        return _mx().stop_gradient(value) if self.is_tensor(value) else value

    def clone(self, value: Any) -> Any:
        """Clone an MLX array value."""

        return value.astype(value.dtype) if self.is_tensor(value) else value

    def replicate_state(self, value: Any, device: Any, *, trainable: bool) -> Any:
        """Create independent state scheduled on the requested MLX stream."""

        del trainable
        if not self.is_tensor(value):
            return value
        with mlx_execution_context(device):
            return self.clone(self.detach(value))

    def resize_batch(self, value: Any, axis: int, batch_size: int) -> Any:
        """Select cyclic rows on the source MLX device."""

        if not self.is_tensor(value):
            return value
        current = int(value.shape[axis])
        if current <= 0:
            raise ValueError("Cannot resize an empty batch axis.")
        indexes = _mx().arange(batch_size, dtype=_mx().int32) % current
        return _mx().take(value, indexes, axis=axis)

    def to_device(self, value: Any, device: Any) -> Any:
        """Schedule a differentiable identity on the requested native MLX stream."""

        if device is None or not self.is_tensor(value):
            return value
        resolved = resolve_mlx_device(device)
        stream = device if isinstance(device, _mx().Stream) else resolved
        return _mx().reshape(value, value.shape, stream=stream)

    def collate(self, values: list[Any]) -> Any:
        """Stack MLX boundary values."""

        if values and self.is_tensor(values[0]):
            return _mx().stack(values)
        return list(values)

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like an MLX array."""

        return _mx().zeros_like(value)

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two MLX arrays are numerically close."""

        if self.is_tensor(left) and self.is_tensor(right):
            result = _mx().allclose(left, right, atol=atol, rtol=rtol)
            _mx().eval(result)
            return bool(result.item())
        return left == right

    def build_segments(
        self,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
    ) -> SegmentBundle:
        """Build MLX generated-eager prefix and suffix segments."""

        for placement in (spec.placement.prefix, spec.placement.suffix):
            if placement.is_explicit:
                resolve_mlx_device(placement.device, split_point=spec.boundary)
        prefix = MlxGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            segment="prefix",
            adapter=self,
        )
        suffix = MlxGeneratedSuffix(
            graph=graph,
            plan=plan,
            spec=spec,
            segment="suffix",
            adapter=self,
        )
        MlxStateCoordinator(prefix._binding, suffix._binding)
        return SegmentBundle(
            prefix=prefix, training_prefix=prefix if spec.trainable else None, suffix=suffix
        )


__all__ = ["MlxGeneratedPrefix", "MlxGeneratedSuffix", "MlxSplitAdapter"]
