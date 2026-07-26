"""tinygrad UOp split replay adapter."""

from __future__ import annotations

from math import prod
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
from ..shape_program import ShapeBinding
from ..ir import SplitRequest
from .base import SegmentBundle, SplitPolicyMixin, boundary_overlay


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


def _is_tinygrad_literal_uop(uop: Any) -> bool:
    """Return whether ``uop`` is a literal branch that should not be parent-swapped."""

    ops = _tinygrad_ops()
    return getattr(uop, "op", None) in {ops.CONST, ops.STACK}


def _rewrite_tinygrad_uop_device(uop: Any, target_device: str | None) -> Any:
    """Rewrite captured tinygrad DEVICE leaves to the requested replay device."""

    if target_device is None or not hasattr(uop, "replace"):
        return uop
    ops = _tinygrad_ops()
    if getattr(uop, "op", None) is ops.DEVICE:
        if getattr(uop, "arg", None) == target_device:
            return uop
        return uop.replace(arg=target_device)
    src = tuple(getattr(uop, "src", ()) or ())
    if not src:
        return uop
    rewritten_src = tuple(_rewrite_tinygrad_uop_device(item, target_device) for item in src)
    if rewritten_src == src:
        return uop
    return uop.replace(src=rewritten_src)


def _tinygrad_shape_tuple(uop: Any) -> tuple[int, ...] | None:
    """Convert a tinygrad shape descriptor UOp into a concrete shape tuple."""

    ops = _tinygrad_ops()
    if getattr(uop, "op", None) is ops.CONST:
        try:
            return (int(getattr(uop, "arg")),)
        except (TypeError, ValueError):
            return None
    if getattr(uop, "op", None) is ops.STACK:
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
        spec: SplitRequest,
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
        self._shape_binding: ShapeBinding | None = None

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

        for param in node.param_refs:
            handle = getattr(param, "_param_ref", None)
            if handle is None:
                handle = getattr(param, "handle", None)
            if self._backend.is_tensor(handle):
                return handle
        return None

    def _live_child_param_source_value(self, node: SplitTraceNode) -> Any | None:
        """Return a unique live parameter handle represented by a source buffer."""

        if node.op_type != "buffer":
            return None
        capture = node.target
        captured_shape = _tinygrad_uop_shape(getattr(capture, "uop", None))
        handles: list[Any] = []
        seen: set[int] = set()
        captured_uop = getattr(capture, "uop", None)
        captured_uop_id = id(captured_uop)
        for candidate in self.graph.nodes:
            candidate_uop = getattr(getattr(candidate, "target", None), "uop", None)
            if candidate_uop is None or not candidate.param_refs:
                continue
            try:
                contains_buffer = any(
                    id(item) == captured_uop_id for item in candidate_uop.toposort()
                )
            except Exception:
                contains_buffer = False
            if not contains_buffer:
                continue
            handle = self._live_param_value(candidate)
            if handle is None or id(handle) in seen:
                continue
            handle_shape = _tinygrad_uop_shape(handle)
            if captured_shape is not None and handle_shape != captured_shape:
                if (
                    handle_shape is None
                    or prod(handle_shape) != prod(captured_shape)
                    or not hasattr(handle, "reshape")
                ):
                    continue
                handle = handle.reshape(captured_shape)
            seen.add(id(handle))
            handles.append(handle)
        return handles[0] if len(handles) == 1 else None

    def _tensor_device(self, value: Any) -> str | None:
        """Return a tinygrad tensor device name, if available."""

        if not self._backend.is_tensor(value):
            return None
        device = getattr(value, "device", None)
        return str(device) if device is not None else None

    def _move_source_to_device(self, value: Any, target_device: str | None) -> Any:
        """Move a replay source payload to the suffix boundary device."""

        if target_device is None or not self._backend.is_tensor(value):
            return value
        if self._tensor_device(value) == target_device or not hasattr(value, "to"):
            return value
        moved = value.to(target_device)
        realize = getattr(moved, "realize", None)
        return realize() if callable(realize) else moved

    def _overlay_device(self, overlay: dict[str, Any]) -> str | None:
        """Infer the execution device from boundary or parent tensor values."""

        for value in overlay.values():
            device = self._tensor_device(value)
            if device is not None:
                return device
        return None

    def _source_value(
        self,
        node: SplitTraceNode,
        *,
        preserve_autograd: bool = False,
        target_device: str | None = None,
    ) -> Any:
        """Return a replay value for source-like nodes."""

        if preserve_autograd:
            live_param = self._live_param_value(node)
            if live_param is not None:
                return live_param
            live_param = self._live_child_param_source_value(node)
            if live_param is not None:
                return live_param
        value = getattr(node.op, "out", None)
        if value is None:
            raise SplitUnsupportedError(
                f"{node.label!r} source value is unavailable.",
                context=self._context(node, "missing source value"),
            )
        return self._move_source_to_device(value, target_device)

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

    def _rewrite_shape_descriptor(
        self,
        *,
        node: SplitTraceNode,
        shape_uop: Any,
    ) -> Any:
        """Rewrite an audited tinygrad shape descriptor for dynamic batch replay."""

        if self.graph.shape_program is None or self._shape_binding is None:
            return shape_uop
        captured_shape = _tinygrad_shape_tuple(shape_uop)
        if captured_shape is None:
            raise SplitUnsupportedError(
                f"tinygrad dynamic-batch replay cannot inspect {node.label!r} shape literal.",
                context=self._context(node, "uninspectable dynamic shape literal"),
            )
        replacement = self.graph.shape_program.rewrite(
            node.canonical_id,
            captured_shape,
            self._shape_binding,
        )
        if tuple(replacement) == captured_shape:
            return shape_uop
        try:
            return _replace_tinygrad_shape_tuple(shape_uop, tuple(replacement))
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
        """Rewrite audited tinygrad shape literal branches for dynamic batch replay.

        tinygrad 0.13 represents ``SHRINK`` margins as a weak-int ``STACK`` in
        ``src[2]``.  It is not a normal output-shape descriptor, but its first
        dimension still contains the captured batch size and must be rewritten
        together with ``RESHAPE``/``EXPAND`` descriptors.
        """

        del overlay
        if self.graph.shape_program is None or self._shape_binding is None:
            return src
        rewritten = list(src)
        parent_by_position = dict(
            getattr(getattr(node, "target", None), "parent_arg_positions", ()) or ()
        )
        for index, item in enumerate(src):
            parent_label = parent_by_position.get(index)
            parent = self.graph.node_for_label(parent_label) if parent_label is not None else None
            if parent is not None and self._is_parameter_lineage(parent):
                rewritten[index] = self._rewrite_parameter_expand_tree(node, item)
                continue
            rewritten[index] = self._rewrite_shape_uop_tree(node, item)
        return rewritten

    def _rewrite_parameter_expand_tree(self, node: SplitTraceNode, uop: Any) -> Any:
        """Rewrite only broadcast extents inside a fixed model-state UOp branch."""

        src = tuple(getattr(uop, "src", ()) or ())
        if not src or not hasattr(uop, "replace"):
            return uop
        ops = _tinygrad_ops()
        if getattr(uop, "op", None) is ops.EXPAND:
            rewritten_src = (
                self._rewrite_parameter_expand_tree(node, src[0]),
                *(self._rewrite_shape_uop_tree(node, item) for item in src[1:]),
            )
        else:
            rewritten_src = tuple(self._rewrite_parameter_expand_tree(node, item) for item in src)
        if rewritten_src == src:
            return uop
        return uop.replace(src=rewritten_src)

    def _is_parameter_lineage(
        self,
        node: SplitTraceNode,
        seen: set[str] | None = None,
    ) -> bool:
        """Return whether one replay node derives exclusively from model state."""

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
        parents = [self.graph.node_for_label(label) for label in node.parents]
        resolved = [parent for parent in parents if parent is not None]
        return bool(resolved) and all(
            self._is_parameter_lineage(parent, visited.copy()) for parent in resolved
        )

    def _is_constant_lineage(
        self,
        node: SplitTraceNode,
        seen: set[str] | None = None,
    ) -> bool:
        """Return whether one replay node derives exclusively from captured constants."""

        if node.is_input or node.is_param_source or node.is_buffer or node.param_refs:
            return False
        if not node.parents:
            return True
        visited = set() if seen is None else seen
        if node.canonical_id in visited:
            return False
        visited.add(node.canonical_id)
        parents = [self.graph.node_for_label(label) for label in node.parents]
        resolved = [parent for parent in parents if parent is not None]
        return bool(resolved) and all(
            self._is_constant_lineage(parent, visited.copy()) for parent in resolved
        )

    def _rewrite_shape_uop_tree(self, node: SplitTraceNode, uop: Any) -> Any:
        """Apply exact shape recipes recursively inside a tinygrad UOp branch."""

        captured_shape = _tinygrad_shape_tuple(uop)
        if captured_shape is not None:
            return self._rewrite_shape_descriptor(node=node, shape_uop=uop)
        rewritten_uop = uop
        arg = getattr(uop, "arg", None)
        if isinstance(arg, (tuple, list)) and self.graph.shape_program is not None:
            try:
                arg_shape = tuple(int(dim) for dim in arg)
            except (TypeError, ValueError):
                arg_shape = ()
            if arg_shape and self._shape_binding is not None:
                replacement = self.graph.shape_program.rewrite(
                    node.canonical_id,
                    arg_shape,
                    self._shape_binding,
                )
                if tuple(replacement) != arg_shape and hasattr(uop, "replace"):
                    rewritten_uop = uop.replace(arg=tuple(replacement))
        src = tuple(getattr(rewritten_uop, "src", ()) or ())
        if not src or not hasattr(rewritten_uop, "replace"):
            return rewritten_uop
        rewritten_src = tuple(self._rewrite_shape_uop_tree(node, item) for item in src)
        if rewritten_src == src:
            return rewritten_uop
        return rewritten_uop.replace(src=rewritten_src)

    def _execute_uop(
        self,
        node: SplitTraceNode,
        overlay: dict[str, Any],
        *,
        preserve_autograd: bool = False,
        target_device: str | None = None,
    ) -> Any:
        """Replay one captured tinygrad UOp using runtime parent values."""

        if preserve_autograd and node.op_type == "buffer":
            live_param = self._live_child_param_source_value(node)
            if live_param is None:
                candidate = getattr(getattr(node, "op", None), "out", None)
                if self._backend.is_tensor(candidate) and getattr(
                    candidate, "requires_grad", False
                ):
                    live_param = candidate
            if live_param is not None:
                return live_param
        capture = node.target
        if not isinstance(capture, TinygradUOpCapture):
            raise SplitUnsupportedError(
                f"{node.label!r} has no tinygrad UOp capture.",
                context=self._context(node, "missing tinygrad UOp capture"),
            )
        if node.op_type == "buffer":
            # BUFFER UOps identify an allocation, not a portable parameter
            # value.  Rewriting only their DEVICE leaf can reinterpret a CPU
            # allocation as CUDA memory.  Bind the live parameter when one is
            # available; otherwise use the captured realized payload and copy
            # it through the backend device path.
            live_param = self._live_child_param_source_value(node)
            captured_live = getattr(capture, "live_tensor", None)
            if live_param is None and self._backend.is_tensor(captured_live):
                live_param = captured_live
            if live_param is not None:
                bound = self._move_source_to_device(live_param, target_device)
                return bound if preserve_autograd else self._backend._realized_copy(bound)
            captured_output = getattr(node.op, "out", None)
            if self._backend.is_tensor(captured_output):
                bound = self._move_source_to_device(captured_output, target_device)
                return bound if preserve_autograd else self._backend._realized_copy(bound)
            payload = getattr(capture, "payload_snapshot", None)
            if payload is not None:
                bound = self._move_source_to_device(payload, target_device)
                return bound if preserve_autograd else self._backend._realized_copy(bound)
        src = [
            _rewrite_tinygrad_uop_device(item, target_device)
            for item in (getattr(capture.uop, "src", ()) or ())
        ]
        if not src and not capture.parent_arg_positions:
            return self._move_source_to_device(capture.payload_snapshot, target_device)
        for position, parent_label in capture.parent_arg_positions:
            if position < 0 or position >= len(src):
                raise SplitUnsupportedError(
                    f"{node.label!r} has invalid tinygrad parent arg position {position!r}.",
                    context=self._context(node, "invalid parent position"),
                )
            if _is_tinygrad_literal_uop(src[position]):
                continue
            parent_node = self.graph.node_for_label(parent_label)
            if parent_node is not None and self._is_constant_lineage(parent_node):
                # Keep the native lazy broadcast tree: a realized replay value
                # loses the EXPAND node needed to encode a new batch extent.
                continue
            parent_value = self._resolve_parent_value(parent_label, node, overlay)
            if (
                preserve_autograd
                or self.spec.dynamic_batch is not None
                or _source_matches_payload(src[position], parent_value)
            ):
                src[position] = parent_value.uop
        src = self._rewrite_dynamic_uop_src(node, src, overlay)
        try:
            replay_uop = _rewrite_tinygrad_uop_device(
                capture.uop.replace(src=tuple(src)),
                target_device,
            )
            value = self._backend._tensor_from_uop(replay_uop)
            return value if preserve_autograd else self._backend._realized_copy(value)
        except Exception as exc:
            recipes = (
                ()
                if self.graph.shape_program is None
                else tuple(
                    recipe.captured
                    for recipe in self.graph.shape_program.recipes.get(node.canonical_id, ())
                )
            )
            raise SplitUnsupportedError(
                f"tinygrad split replay failed for {node.label!r}: {exc}; "
                f"shape_recipes={recipes!r}",
                context=self._context(node, "tinygrad UOp replay failed"),
            ) from exc

    def _execute_nodes(
        self,
        overlay: dict[str, Any],
        *,
        preserve_autograd: bool = False,
        target_device: str | None = None,
    ) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        target_device = target_device or self._overlay_device(overlay)
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
                        target_device=target_device,
                    )
                continue
            if node.is_output and node.target is None:
                continue
            overlay[node.canonical_id] = self._execute_uop(
                node,
                overlay,
                preserve_autograd=preserve_autograd,
                target_device=target_device,
            )
        return overlay


class TinygradGeneratedPrefix(_TinygradGeneratedSegmentBase):
    """Generated-eager tinygrad prefix segment."""

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
                "Runtime inputs do not match traced tinygrad tensor input count.",
                context=SplitErrorContext(
                    backend="tinygrad",
                    split_point=self.spec.boundary,
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
                backend="tinygrad",
                split_point=self.spec.boundary,
            )
        runtime_batch_size = None if self._shape_binding is None else self._shape_binding.batch_size
        self._execute_nodes(overlay, preserve_autograd=not detach_boundary)
        boundary_tensors: dict[str, Any] = {}
        for node_id in self.plan.boundary_node_ids:
            node = self._node_by_id[node_id]
            key = boundary_key_for_node(node.canonical_id, node.output_container_path)
            value = overlay[node_id]
            boundary_tensors[key] = (
                self._backend._realized_copy(value) if detach_boundary else value
            )
        return ReplayBoundary(
            backend="tinygrad",
            tensors=boundary_tensors,
            spec=self.plan.boundary_spec,
            metadata={
                "split_id": self.plan.split_id,
                "graph_shape_hash": self.graph.graph_shape_hash,
                "batch_symbol": self.spec.batch_symbol,
                "dynamic_batch": self.spec.dynamic_batch,
                "runtime_batch_size": runtime_batch_size,
                "shape_program_hash": (
                    None
                    if self.graph.shape_program is None
                    else self.graph.shape_program.fingerprint
                ),
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

        overlay = boundary_overlay(boundary, self.plan)
        runtime_batch_size = boundary.metadata.get("runtime_batch_size")
        if self.graph.shape_program is not None and runtime_batch_size is not None:
            self._shape_binding = self.graph.shape_program.binding_from_batch(
                int(runtime_batch_size)
            )
        preserve_autograd = bool(boundary.metadata.get("suffix_training_roots"))
        self._execute_nodes(
            overlay,
            preserve_autograd=preserve_autograd,
            target_device=self._overlay_device(overlay),
        )
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


class TinygradSplitAdapter(SplitPolicyMixin):
    """tinygrad split backend adapter."""

    name = "tinygrad"
    supports_replay = True
    supports_training = True
    supports_boundary_cache = True
    supports_dynamic_batch = True
    native_target_types = frozenset({"TinygradUOpCapture"})

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
        if not hasattr(value, "to"):
            return value
        detach = getattr(value, "detach", None)
        source = detach() if callable(detach) else value
        moved = source.to(str(device))
        realize = getattr(moved, "realize", None)
        return realize() if callable(realize) else moved

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
        spec: SplitRequest,
    ) -> SegmentBundle:
        """Build tinygrad UOp replay prefix/suffix segments."""

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
