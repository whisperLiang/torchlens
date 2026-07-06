"""tinygrad UOp split replay adapter."""

from __future__ import annotations

from typing import Any

from ...backends.tinygrad.backend import (
    TinygradBackend,
    TinygradUOpCapture,
    _source_matches_payload,
)
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import SplitTraceGraph, SplitTraceNode
from ..planner import SplitPlan
from ..shape import is_dynamic_batch_shape_sensitive_op
from ..spec import SplitSpec
from .base import SegmentBundle


def _tinygrad_tensor_type() -> Any:
    """Return the tinygrad Tensor class."""

    from tinygrad import Tensor

    return Tensor


def _flatten_tensor_leaves(value: Any) -> list[Any]:
    """Collect tinygrad tensor leaves in deterministic traversal order."""

    tensor_type = _tinygrad_tensor_type()
    if isinstance(value, tensor_type):
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
    return []


def _tinygrad_ops() -> Any:
    """Return tinygrad UOp enum values used by split replay."""

    from tinygrad.uop.ops import Ops

    return Ops


def _is_tinygrad_weakint(dtype: Any) -> bool:
    """Return whether ``dtype`` represents tinygrad weak integer shape data."""

    return "weakint" in str(dtype)


def _is_tinygrad_shape_uop(uop: Any) -> bool:
    """Return whether ``uop`` is a tinygrad shape descriptor node."""

    ops = _tinygrad_ops()
    return getattr(uop, "op", None) in {ops.STACK, ops.CONST} and _is_tinygrad_weakint(
        getattr(uop, "dtype", None)
    )


def _tinygrad_shape_tuple(uop: Any) -> tuple[int, ...] | None:
    """Convert a tinygrad shape descriptor UOp into a concrete shape tuple."""

    ops = _tinygrad_ops()
    if getattr(uop, "op", None) is ops.CONST and _is_tinygrad_weakint(
        getattr(uop, "dtype", None)
    ):
        try:
            return (int(getattr(uop, "arg")),)
        except (TypeError, ValueError):
            return None
    if getattr(uop, "op", None) is ops.STACK and _is_tinygrad_weakint(
        getattr(uop, "dtype", None)
    ):
        dims: list[int] = []
        for source in getattr(uop, "src", ()) or ():
            item = _tinygrad_shape_tuple(source)
            if item is None or len(item) != 1:
                return None
            dims.append(item[0])
        return tuple(dims)
    return None


def _replace_tinygrad_shape_tuple(uop: Any, shape: tuple[int, ...]) -> Any:
    """Return ``uop`` with the same shape descriptor structure and new dims."""

    ops = _tinygrad_ops()
    if getattr(uop, "op", None) is ops.CONST:
        if len(shape) != 1:
            raise ValueError("CONST shape descriptors require one replacement dim.")
        return uop.replace(arg=int(shape[0]))
    if getattr(uop, "op", None) is ops.STACK:
        src = tuple(getattr(uop, "src", ()) or ())
        if len(src) != len(shape):
            raise ValueError("STACK shape descriptor rank changed during dynamic batch rewrite.")
        return uop.replace(
            src=tuple(
                _replace_tinygrad_shape_tuple(item, (dim,))
                for item, dim in zip(src, shape, strict=False)
            )
        )
    raise ValueError("Unsupported tinygrad shape descriptor.")


def _tinygrad_uop_shape(value: Any) -> tuple[int, ...] | None:
    """Return a UOp output shape when tinygrad can infer it."""

    try:
        return tuple(int(dim) for dim in value.shape)
    except Exception:
        return None


class _TinygradGeneratedSegmentBase:
    """Shared tinygrad UOp replay helpers."""

    def __init__(
        self,
        *,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitSpec,
        node_ids: frozenset[str],
    ) -> None:
        """Create a generated tinygrad replay segment."""

        self.graph = graph
        self.plan = plan
        self.spec = spec
        self.node_ids = node_ids
        self._node_by_id = graph.node_by_id
        self._label_to_id = graph.node_id_by_alias
        self._backend = TinygradBackend()

    def _context(self, node: SplitTraceNode, reason: str) -> SplitErrorContext:
        """Build an error context for ``node``."""

        return SplitErrorContext(
            backend="tinygrad",
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

    def _live_param_value(self, node: SplitTraceNode) -> Any | None:
        """Return a live tinygrad parameter handle for a param source node."""

        if not node.is_param_source:
            return None
        for param in node.param_refs:
            handle = getattr(param, "_param_ref", None) or getattr(param, "handle", None)
            if self._backend.is_tensor(handle):
                return handle
        return None

    def _source_value(self, node: SplitTraceNode, *, preserve_autograd: bool = False) -> Any:
        """Return a replay value for source-like nodes."""

        if preserve_autograd:
            live_param = self._live_param_value(node)
            if live_param is not None:
                return live_param
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
                f"{node.label!r} references unavailable tinygrad parent {parent_label!r}.",
                context=self._context(node, "missing parent value"),
            )
        return overlay[parent_id]

    def _runtime_batch_size(self, overlay: dict[str, Any]) -> int | None:
        """Infer runtime batch size from symbolized input nodes."""

        if self.spec.dynamic_batch is None or self.graph.traced_batch_size is None:
            return None
        candidate_node_ids = (*self.graph.input_node_ids, *self.plan.boundary_node_ids)
        for node_id in candidate_node_ids:
            node = self._node_by_id[node_id]
            if not node.output_shape or node.output_shape[0] != self.graph.traced_batch_size:
                continue
            value = overlay.get(node_id)
            shape = getattr(value, "shape", None)
            if self._backend.is_tensor(value) and shape is not None and len(shape) == len(
                node.output_shape
            ):
                return int(shape[0])
        return None

    def _rewrite_shape_descriptor(
        self,
        *,
        node: SplitTraceNode,
        shape_uop: Any,
        runtime_batch_size: int | None,
    ) -> Any:
        """Rewrite an audited tinygrad shape descriptor for dynamic batch replay."""

        if self.spec.dynamic_batch is None:
            return shape_uop
        traced_batch_size = self.graph.traced_batch_size
        if traced_batch_size is None or runtime_batch_size is None:
            return shape_uop
        captured_shape = _tinygrad_shape_tuple(shape_uop)
        if captured_shape is None:
            raise SplitUnsupportedError(
                f"tinygrad dynamic-batch replay cannot inspect {node.label!r} shape literal.",
                context=self._context(node, "uninspectable dynamic shape literal"),
            )
        if not captured_shape or captured_shape[0] != traced_batch_size:
            return shape_uop
        low, high = self.spec.dynamic_batch
        if not low <= runtime_batch_size <= high:
            raise SplitUnsupportedError(
                f"tinygrad runtime batch {runtime_batch_size} is outside {self.spec.dynamic_batch}.",
                context=self._context(node, "dynamic batch outside allowed range"),
            )
        replacement = (runtime_batch_size, *captured_shape[1:])
        try:
            return _replace_tinygrad_shape_tuple(shape_uop, replacement)
        except ValueError as exc:
            raise SplitUnsupportedError(
                f"tinygrad dynamic-batch replay failed for {node.label!r}: {exc}",
                context=self._context(node, "dynamic shape rewrite failed"),
            ) from exc

    def _rewrite_dynamic_uop_src(
        self,
        node: SplitTraceNode,
        src: list[Any],
        overlay: dict[str, Any],
    ) -> list[Any]:
        """Rewrite audited tinygrad shape literal branches for dynamic batch replay."""

        if self.spec.dynamic_batch is None:
            return src
        if not is_dynamic_batch_shape_sensitive_op(node.op_type, getattr(node, "func_name", None)):
            return src
        runtime_batch_size = self._runtime_batch_size(overlay)
        if runtime_batch_size is None:
            return src
        rewritten = list(src)
        for index, item in enumerate(src):
            if index == 0:
                continue
            if _is_tinygrad_shape_uop(item):
                rewritten[index] = self._rewrite_shape_descriptor(
                    node=node,
                    shape_uop=item,
                    runtime_batch_size=runtime_batch_size,
                )
        return rewritten

    def _execute_uop(
        self,
        node: SplitTraceNode,
        overlay: dict[str, Any],
        *,
        preserve_autograd: bool = False,
    ) -> Any:
        """Replay one captured tinygrad UOp using runtime parent values."""

        capture = node.target
        if not isinstance(capture, TinygradUOpCapture):
            raise SplitUnsupportedError(
                f"{node.label!r} has no tinygrad UOp capture.",
                context=self._context(node, "missing tinygrad UOp capture"),
            )
        src = list(getattr(capture.uop, "src", ()) or ())
        if not src and not capture.parent_arg_positions:
            return capture.payload_snapshot
        for position, parent_label in capture.parent_arg_positions:
            if position < 0 or position >= len(src):
                raise SplitUnsupportedError(
                    f"{node.label!r} has invalid tinygrad parent arg position {position!r}.",
                    context=self._context(node, "invalid parent position"),
                )
            parent_value = self._resolve_parent_value(parent_label, node, overlay)
            if (
                preserve_autograd
                or self.spec.dynamic_batch is not None
                or _source_matches_payload(src[position], parent_value)
            ):
                src[position] = parent_value.uop
        src = self._rewrite_dynamic_uop_src(node, src, overlay)
        try:
            replay_uop = capture.uop.replace(src=tuple(src))
            value = self._backend._tensor_from_uop(replay_uop)
            return value if preserve_autograd else self._backend._realized_copy(value)
        except Exception as exc:
            raise SplitUnsupportedError(
                f"tinygrad split replay failed for {node.label!r}: {exc}",
                context=self._context(node, "tinygrad UOp replay failed"),
            ) from exc

    def _execute_nodes(
        self,
        overlay: dict[str, Any],
        *,
        preserve_autograd: bool = False,
    ) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids:
                continue
            if node.is_input:
                continue
            if node.target is None and self._is_replay_source_node(node):
                if node.canonical_id not in overlay and not node.is_output:
                    overlay[node.canonical_id] = self._source_value(
                        node,
                        preserve_autograd=preserve_autograd,
                    )
                continue
            if node.is_output and node.target is None:
                continue
            overlay[node.canonical_id] = self._execute_uop(
                node,
                overlay,
                preserve_autograd=preserve_autograd,
            )
        return overlay


class TinygradGeneratedPrefix(_TinygradGeneratedSegmentBase):
    """Generated-eager tinygrad prefix segment."""

    def __call__(self, *inputs: Any, detach_boundary: bool) -> ReplayBoundary:
        """Run the prefix and return a replay boundary."""

        input_leaves = _flatten_tensor_leaves(inputs)
        if len(input_leaves) != len(self.graph.input_node_ids):
            raise SplitUnsupportedError(
                "Runtime inputs do not match traced tinygrad tensor input count.",
                context=SplitErrorContext(
                    backend="tinygrad",
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
        self._execute_nodes(overlay, preserve_autograd=not detach_boundary)
        boundary_tensors: dict[str, Any] = {}
        for node_id in self.plan.boundary_node_ids:
            node = self._node_by_id[node_id]
            key = boundary_key_for_node(node.canonical_id, node.output_container_path)
            value = overlay[node_id]
            boundary_tensors[key] = self._backend._realized_copy(value) if detach_boundary else value
        return ReplayBoundary(
            backend="tinygrad",
            tensors=boundary_tensors,
            spec=self.plan.boundary_spec,
            metadata={
                "split_id": self.plan.split_id,
                "graph_shape_hash": self.graph.graph_shape_hash,
                "batch_symbol": self.spec.batch_symbol,
                "dynamic_batch": self.spec.dynamic_batch,
                "device_policy": self.spec.device_policy,
                "supports_prefix_backward": not detach_boundary,
                "prefix_inputs": tuple(input_leaves) if not detach_boundary else (),
                "prefix_boundary_tensors": boundary_tensors if not detach_boundary else {},
            },
        )


class TinygradGeneratedSuffix(_TinygradGeneratedSegmentBase):
    """Generated-eager tinygrad suffix segment."""

    def __call__(self, boundary: ReplayBoundary) -> Any:
        """Run the suffix from ``boundary`` and reconstruct final output."""

        overlay = dict(boundary.tensors)
        for key, item in boundary.spec.items():
            node_id = self._label_to_id.get(item.label)
            if node_id is not None and key in boundary.tensors:
                overlay[node_id] = boundary.tensors[key]
        preserve_autograd = bool(boundary.metadata.get("suffix_training_roots"))
        self._execute_nodes(overlay, preserve_autograd=preserve_autograd)
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


class TinygradSplitAdapter:
    """tinygrad split backend adapter."""

    name = "tinygrad"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_dynamic_batch = True

    def __init__(self) -> None:
        """Create a tinygrad split adapter."""

        self._backend = TinygradBackend()

    def is_tensor(self, value: Any) -> bool:
        """Return whether ``value`` is a tinygrad tensor."""

        return isinstance(value, _tinygrad_tensor_type())

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
        """Return tinygrad gradient metadata when available."""

        if not self.is_tensor(value):
            return None
        return getattr(value, "requires_grad", None)

    def detach(self, value: Any) -> Any:
        """Detach tensor values when tinygrad exposes the method."""

        return value.detach() if hasattr(value, "detach") else value

    def clone(self, value: Any) -> Any:
        """Clone tensor values as realized tinygrad tensors."""

        if not self.is_tensor(value):
            return value
        return self._backend._realized_copy(value)

    def to_device(self, value: Any, device: Any) -> Any:
        """Move tensor values to a tinygrad device when requested."""

        if not self.is_tensor(value) or device is None:
            return value
        return value.to(str(device)) if hasattr(value, "to") else value

    def collate(self, values: list[Any]) -> Any:
        """Stack tinygrad tensor values."""

        Tensor = _tinygrad_tensor_type()
        if values and isinstance(values[0], Tensor):
            return Tensor.stack(*values)
        return list(values)

    def zeros_like(self, value: Any) -> Any:
        """Return zeros like a tensor."""

        return value.zeros_like()

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Return whether two tensors are numerically close."""

        if not (self.is_tensor(left) and self.is_tensor(right)):
            return left == right
        left_payload = self._backend._realized_copy(left).tolist()
        right_payload = self._backend._realized_copy(right).tolist()
        return _nested_close(left_payload, right_payload, atol=atol, rtol=rtol)

    def build_segments(
        self,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitSpec,
    ) -> SegmentBundle:
        """Build tinygrad UOp replay prefix/suffix segments."""

        if spec.mode == "compiled":
            raise SplitUnsupportedError(
                "tinygrad compiled split mode is not supported.",
                context=SplitErrorContext(
                    backend="tinygrad",
                    split_point=spec.boundary,
                    module_path=None,
                    op_type=None,
                    layer_label=None,
                    reason="compiled split mode unsupported",
                ),
            )
        prefix = TinygradGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.prefix_node_ids,
        )
        suffix = TinygradGeneratedSuffix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.suffix_node_ids,
        )
        training_prefix = TinygradGeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.prefix_node_ids,
        )
        return SegmentBundle(prefix=prefix, training_prefix=training_prefix, suffix=suffix)


def _nested_close(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    """Return whether nested scalar payloads are numerically close."""

    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(l_item, r_item, atol=atol, rtol=rtol)
            for l_item, r_item in zip(left, right)
        )
    return abs(float(left) - float(right)) <= atol + rtol * abs(float(right))


__all__ = [
    "TinygradGeneratedPrefix",
    "TinygradGeneratedSuffix",
    "TinygradSplitAdapter",
]
