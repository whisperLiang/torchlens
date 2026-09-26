"""Torch generated-eager split replay adapter."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType
from typing import Any

from ... import _state
from ...intervention.types import LiteralTensor, LiteralValue, ParentRef, Unsupported
from ...ir.container import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    TupleIndex,
    rebuild_container_from_spec,
)
from ...utils._torch_compat import (
    autocast_is_enabled,
    get_current_dispatch_mode_stack,
    get_torch_function_mode_stack_length,
)
from ...utils.rng import AutocastRestore, execute_with_restored_rng_autocast
from .._torch_liveness import release_schedule
from ..boundary import ReplayBoundary
from ..errors import SplitErrorContext, SplitUnsupportedError
from ..frontier import boundary_key_for_node
from ..graph import ReplayValueRef, SplitTraceGraph, SplitTraceNode
from ..ir import SplitRequest
from ..placement import DevicePlacement, SegmentName
from ..planner import SplitPlan
from ..shape_program import ShapeBinding, shape_semantic_for_node
from ..state import SegmentState, _StateReplicaPool
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


def _plain_torch_arguments(value: Any, torch: Any) -> bool:
    """Accept only built-in argument trees without user tensor dispatch hooks."""

    if isinstance(value, torch.Tensor):
        return type(value) in (torch.Tensor, torch.nn.Parameter)
    if type(value) in (tuple, list, torch.Size):
        return all(_plain_torch_arguments(item, torch) for item in value)
    if type(value) is dict:
        return all(
            _plain_torch_arguments(key, torch) and _plain_torch_arguments(item, torch)
            for key, item in value.items()
        )
    if type(value) is slice:
        return all(
            _plain_torch_arguments(item, torch) for item in (value.start, value.stop, value.step)
        )
    return value is Ellipsis or type(value) in (
        bool,
        int,
        float,
        str,
        type(None),
        torch.dtype,
        torch.device,
        torch.layout,
        torch.memory_format,
    )


def _plain_replay_template(component: Any, torch: Any) -> bool:
    """Certify a captured argument tree without resolving any runtime values."""

    if isinstance(component, (ParentRef, ReplayValueRef)):
        return True
    if isinstance(component, LiteralTensor):
        value = component.value
        return type(value) in (torch.Tensor, torch.nn.Parameter)
    if isinstance(component, LiteralValue):
        return _plain_torch_arguments(component.value, torch)
    if isinstance(component, Unsupported):
        return False
    if isinstance(component, tuple):
        if _is_template_dict(component):
            return all(
                _plain_torch_arguments(key, torch) and _plain_replay_template(value, torch)
                for key, value in component
            )
        return all(_plain_replay_template(value, torch) for value in component)
    return _plain_torch_arguments(component, torch)


def _rng_free_torch_targets(torch: Any) -> tuple[Any, ...]:
    """Resolve a small, exact-identity set of deterministic Torch builtins."""

    from ...utils.display import identity

    variable_functions = getattr(getattr(torch, "_C", None), "_VariableFunctionsClass", None)
    functions = (
        getattr(torch.nn.functional, "linear", None),
        getattr(torch.nn.functional, "gelu", None),
        getattr(variable_functions, "meshgrid", None),
        *(
            getattr(torch, name, None)
            for name in (
                "relu",
                "conv2d",
                "layer_norm",
                "cat",
                "stack",
                "concat",
                "sum",
                "split_with_sizes",
                "arange",
                "floor_divide",
                "pow",
                "linspace",
                "meshgrid",
                "topk",
                "gather",
                "grid_sampler",
                "zeros",
                "ones_like",
                "_shape_as_tensor",
            )
        ),
    )
    tensor_methods = (
        "view",
        "permute",
        "__getitem__",
        "__mul__",
        "__rmul__",
        "__add__",
        "__sub__",
        "__truediv__",
        "__invert__",
        "__eq__",
        "contiguous",
        "transpose",
        "reshape",
        "flatten",
        "unbind",
        "chunk",
        "unsqueeze",
        "sin",
        "cos",
        "masked_fill",
        "sum",
        "expand",
        "repeat",
        "cumsum",
        "exp",
        "max",
        "softmax",
        "float",
        "to",
        "new_zeros",
        "prod",
        "__gt__",
        "__lt__",
        "__and__",
        "all",
        "detach",
    )
    candidates = (*functions, *(getattr(torch.Tensor, name, None) for name in tensor_methods))
    originals = (_state._decorated_to_orig.get(id(target), target) for target in candidates)
    builtins = tuple(
        target
        for target in originals
        if type(target) in (BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType)
    )
    # The TorchLens identity placeholder only returns its argument. PyTorch's
    # SiLU and interpolate wrappers are deterministic for plain tensors; the
    # replay fast path separately excludes custom tensor dispatch and modes.
    python_targets = (identity, torch.nn.functional.silu, torch.nn.functional.interpolate)
    return (
        *builtins,
        *(_state._decorated_to_orig.get(id(target), target) for target in python_targets),
    )


def _module_for_param_ref(param_ref: Any) -> Any | None:
    """Return a parameter's owning module metadata when its address is live.

    ``Param.module`` resolves through ``trace.modules[module_address]``.  A
    captured parameter can belong to a module whose ``forward`` was never
    called (for example, an embedding accessed directly through ``.weight``),
    so that owner need not have an executed module-log entry. Module metadata
    is needed here only to add
    registered-buffer handles; the parameter handle itself is resolved in the
    strict pass above and must not be softened.
    """

    try:
        return getattr(param_ref, "module", None)
    except (AttributeError, KeyError):
        return None


def _shareable_param_sources(graph: SplitTraceGraph) -> frozenset[int]:
    """Certify frozen sources whose every captured use is read-only and nonaliasing.

    Parameters
    ----------
    graph
        Complete replay graph, including both sides of the cut.

    Returns
    -------
    frozenset
        Source identities eligible for runtime-local replica sharing. Unknown
        calls, views (including detach), mutable buffers, and trainable aliases
        disqualify their entire storage. Freezing alone is not an immutability
        proof: embedding renormalization, for example, writes a detached weight.
    """

    torch = _torch()
    # These calls produce fresh outputs and never mutate parameter arguments.
    # Keep this deliberately small; expanding it requires the same guarantee.
    read_only = {
        (namespace, name)
        for namespace in ("torch", "torch.nn.functional")
        for name in (
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
        )
    }
    sources: dict[int, tuple[str, int]] = {}
    unsafe: set[tuple[str, int]] = set()

    def storage_key(value: Any) -> tuple[str, int]:
        """Identify live storage, including distinct parameter views of it."""

        return str(value.device), value.untyped_storage().data_ptr()

    for node in graph.nodes:
        func_id = getattr(node.args_template, "func_id", None)
        call_key = (getattr(func_id, "namespace", ""), getattr(func_id, "qualname", ""))
        safe_call = call_key in read_only
        for ref in node.param_refs:
            handle = getattr(ref, "handle", None)
            if (
                handle is None
                or not isinstance(handle, torch.Tensor)
                or handle.layout != torch.strided
            ):
                continue
            key = storage_key(handle)
            sources[id(handle)] = key
            if (
                not safe_call
                or handle.requires_grad
                or type(handle) not in (torch.Tensor, torch.nn.Parameter)
            ):
                unsafe.add(key)
            module = _module_for_param_ref(ref)
            for buffer in (getattr(module, "buffers", None) or {}).values():
                value = getattr(buffer, "handle", None)
                if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                    unsafe.add(storage_key(value))
        for ref in node.buffer_refs:
            value = getattr(ref, "handle", None)
            if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                unsafe.add(storage_key(value))
    return frozenset(source_id for source_id, key in sources.items() if key not in unsafe)


def _rewrite_placement_device_args(
    node: SplitTraceNode,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    placement: DevicePlacement,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Relocate device arguments using Torch argument semantics, not literal values.

    Parameters
    ----------
    node:
        Captured Torch call whose argument template supplies the callable identity.
    args, kwargs:
        Reconstructed runtime arguments; unrelated strings and integers remain unchanged.
    placement:
        Segment device override, or the unplaced policy that preserves the capture.

    Returns
    -------
    tuple
        Positional and keyword arguments with explicit device destinations relocated.
    """

    if not placement.is_explicit:
        return args, kwargs
    func_id = getattr(node.args_template, "func_id", None)
    namespace = getattr(func_id, "namespace", "") or ""
    if namespace != "torch" and not namespace.startswith("torch."):
        return args, kwargs
    torch = _torch()
    device = torch.device(placement.device)
    if "device" in kwargs:
        kwargs = {**kwargs, "device": device}
    qualname = getattr(func_id, "qualname", "").rsplit(".", 1)[-1]
    if namespace == "torch.Tensor" and qualname == "to" and len(args) > 1:
        # Tensor.to has device, dtype and other-tensor overloads. Only the
        # device overload's positional slot is a device destination.
        destination = args[1]
        if isinstance(destination, (str, torch.device)) or type(destination) is int:
            args = (args[0], device, *args[2:])
    return args, kwargs


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

    segment: SegmentName = "prefix"

    def __init__(
        self,
        *,
        graph: SplitTraceGraph,
        plan: SplitPlan,
        spec: SplitRequest,
        node_ids: frozenset[str],
        use_live_param_sources: bool,
        placement: DevicePlacement | None = None,
        state: SegmentState | None = None,
    ) -> None:
        """Create a generated replay segment."""

        self.graph = graph
        self.plan = plan
        self.spec = spec
        self.node_ids = node_ids
        self.use_live_param_sources = use_live_param_sources
        self._node_by_id = graph.node_by_id
        self._label_to_id = graph.node_id_by_alias
        self._shape_binding: ShapeBinding | None = None
        self._shareable_param_ids = _shareable_param_sources(graph)
        retained_ids = set(plan.boundary_node_ids)
        if self.segment == "suffix":
            retained_ids = set(graph.output_node_ids)
            for node_id in graph.output_node_ids:
                retained_ids.update(
                    graph.node_id_by_alias.get(parent, parent)
                    for parent in graph.node_by_id[node_id].parents
                )
        self._release_after = release_schedule(graph, node_ids, frozenset(retained_ids))
        self._execution_steps = self._prepare_execution_steps()
        shape_program = graph.shape_program
        self._shape_rewrite_node_ids = (
            frozenset(shape_program.recipes) if shape_program is not None else frozenset()
        )
        self._reshape_node_ids = (
            frozenset(
                node.canonical_id
                for node in graph.nodes
                if shape_semantic_for_node(node) == "reshape"
            )
            if shape_program is not None
            else frozenset()
        )
        self._empty_param_cursor = _LiveParamCursor([])
        self._rng_free_targets = _rng_free_torch_targets(_torch())
        self._rng_free_target_ids = frozenset(id(target) for target in self._rng_free_targets)
        torch = _torch()
        self._dropout_target = _state._decorated_to_orig.get(id(torch.dropout), torch.dropout)
        attention = torch.nn.functional.scaled_dot_product_attention
        self._attention_target = _state._decorated_to_orig.get(id(attention), attention)
        function_stack_len = getattr(torch.overrides, "_len_torch_function_stack", None)
        self._function_mode_stack_len = (
            function_stack_len
            if callable(function_stack_len)
            else get_torch_function_mode_stack_length
        )
        try:
            from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
        except ImportError:
            self._dispatch_mode_stack = get_current_dispatch_mode_stack
        else:
            self._dispatch_mode_stack = _get_current_dispatch_mode_stack
        resolved_placement = (
            placement if placement is not None else spec.placement.for_segment(self.segment)
        )
        self._state = (
            state
            if state is not None
            else SegmentState(adapter=TorchSplitAdapter(), placement=resolved_placement)
        )
        self._fast_guard_state_refs: tuple[Any, ...] = ()
        self._fast_guard_certified = self._prepare_fast_guard()

    def _prepare_fast_guard(self) -> bool:
        """Certify a segment whose plain builtins cannot change execution modes."""

        torch = _torch()
        state_refs: dict[int, Any] = {}
        for node, group, executor in self._execution_steps:
            if group is None:
                if node.is_buffer:
                    return False
                value = getattr(node.op, "out", None)
                if value is None or not _plain_torch_arguments(value, torch):
                    return False
                continue
            if executor is None or executor.target is None or executor.buffer_refs:
                return False
            template = executor.args_template
            if template is None or not self._static_rng_free(executor.target, template):
                return False
            autocast_state = getattr(executor.op, "func_autocast_state", None) or {}
            if any(
                state["enabled"]
                for device, state in autocast_state.items()
                if not device.startswith("__")
            ):
                return False
            for ref in executor.param_refs:
                handle = getattr(ref, "handle", None)
                if type(handle) not in (torch.Tensor, torch.nn.Parameter):
                    return False
                state_refs[id(ref)] = ref
                module = _module_for_param_ref(ref)
                for buffer in (getattr(module, "buffers", None) or {}).values():
                    buffer_handle = getattr(buffer, "handle", None)
                    if type(buffer_handle) not in (torch.Tensor, torch.nn.Parameter):
                        return False
                    state_refs[id(buffer)] = buffer
            if not all(
                _plain_replay_template(component, torch) for component in template.args
            ) or not all(
                _plain_torch_arguments(key, torch) and _plain_replay_template(component, torch)
                for key, component in template.kwargs
            ):
                return False
        self._fast_guard_state_refs = tuple(state_refs.values())
        return True

    def _static_rng_free(self, target: Any, template: Any) -> bool:
        """Recognize pure targets including statically disabled dropout."""

        if id(target) in self._rng_free_target_ids:
            return True
        if target is self._dropout_target:
            component = (
                template.args[2] if len(template.args) > 2 else dict(template.kwargs).get("train")
            )
            return isinstance(component, LiteralValue) and component.value is False
        if target is self._attention_target:
            component = (
                template.args[4]
                if len(template.args) > 4
                else dict(template.kwargs).get("dropout_p", LiteralValue(0.0))
            )
            return (
                isinstance(component, LiteralValue)
                and type(component.value) in (int, float)
                and component.value == 0
            )
        return False

    def _can_fast_execute(self, values: Any) -> bool:
        """Check runtime inputs and thread-local modes for the certified path."""

        if not self._fast_guard_certified or self.placement.is_explicit:
            return False
        torch = _torch()
        if not all(_plain_torch_arguments(value, torch) for value in values):
            return False
        if any(
            type(getattr(ref, "handle", None)) not in (torch.Tensor, torch.nn.Parameter)
            for ref in self._fast_guard_state_refs
        ):
            return False
        for entry in self._state._entries.values():
            if type(entry.value) not in (torch.Tensor, torch.nn.Parameter):
                return False
        for entries in self._state._inherited.values():
            if any(
                type(entry.value) not in (torch.Tensor, torch.nn.Parameter) for entry in entries
            ):
                return False
        try:
            if self._function_mode_stack_len() != 0 or self._dispatch_mode_stack() != []:
                return False
            return not autocast_is_enabled("cpu") and not autocast_is_enabled("cuda")
        except Exception:
            return False

    def _prepare_execution_steps(
        self,
    ) -> tuple[
        tuple[SplitTraceNode, tuple[SplitTraceNode, ...] | None, SplitTraceNode | None], ...
    ]:
        """Resolve segment calls and multi-output groups once at preparation."""

        steps: list[
            tuple[SplitTraceNode, tuple[SplitTraceNode, ...] | None, SplitTraceNode | None]
        ] = []
        executed_call_ids: set[str] = set()
        call_by_output = self.graph.replay_call_by_output_id
        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids or node.is_input or node.is_output:
                continue
            if node.is_buffer or (node.target is None and self._is_replay_source_node(node)):
                steps.append((node, None, None))
                continue
            call = call_by_output.get(node.canonical_id)
            call_id = node.canonical_id if call is None else call.call_id
            if call_id in executed_call_ids:
                continue
            executed_call_ids.add(call_id)
            group = (
                (node,)
                if call is None
                else tuple(
                    self._node_by_id[node_id]
                    for node_id in call.output_node_ids
                    if node_id in self.node_ids
                )
            )
            executor = next((member for member in group if member.target is not None), None)
            steps.append((node, group, executor))
        return tuple(steps)

    @property
    def placement(self) -> DevicePlacement:
        """Return this segment's declared placement."""

        return self._state.placement

    def trainable_parameters(self) -> list[Any]:
        """Bind and return owned parameters before an optimizer or forward runs."""

        for node in self.graph.nodes:
            if node.canonical_id in self.node_ids and not (node.is_input or node.is_output):
                self._param_handles_for_node(node)
        return self._state.trainable_values()

    def state_report(self) -> dict[str, Any]:
        """Return diagnostics for this segment's state binding."""

        return self._state.as_dict()

    def bound_state_values(self) -> dict[str, Any]:
        """Bind and expose effective replay state without executing graph operations.

        Returns
        -------
        dict
            Tensor state keyed by stable node ID and template occurrence. Live
            parameter substitution follows the same cursor as argument replay,
            so unused captured copies do not contribute to state fingerprints.
        """

        values: dict[str, Any] = {}
        for node in self.graph.nodes:
            if node.canonical_id not in self.node_ids or node.is_input or node.is_output:
                continue
            if node.is_buffer or (node.target is None and self._is_replay_source_node(node)):
                values[f"{node.canonical_id}:source"] = self._source_value(node)
                continue
            template = node.args_template
            if template is None:
                continue
            param_cursor = _LiveParamCursor(self._param_handles_for_node(node))
            occurrence = 0

            def bind_component(
                component: Any,
                *,
                cursor: _LiveParamCursor = param_cursor,
                node_id: str = node.canonical_id,
            ) -> None:
                """Walk captured argument leaves in replay's resolution order."""

                nonlocal occurrence
                if isinstance(component, LiteralTensor):
                    value = cursor.maybe_replace(component.value)
                    if value is component.value:
                        value = self._state.resolve(value)
                    values[f"{node_id}:literal:{occurrence}"] = value
                    occurrence += 1
                elif isinstance(component, tuple):
                    if _is_template_dict(component):
                        for _key, item in component:
                            bind_component(item)
                    else:
                        for item in component:
                            bind_component(item)

            for component in template.args:
                bind_component(component)
            for _key, component in template.kwargs:
                bind_component(component)
        return values

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
            handle = self._state.resolve(
                handle,
                shareable=id(handle) in self._shareable_param_ids,
            )
            if id(handle) not in seen_handles:
                handles.append(handle)
                seen_handles.add(id(handle))
        for param_ref in node.param_refs:
            module = _module_for_param_ref(param_ref)
            buffers = getattr(module, "buffers", None)
            if buffers is None:
                continue
            for buffer in buffers.values():
                buffer_handle = getattr(buffer, "handle", None)
                if buffer_handle is not None:
                    buffer_handle = self._state.resolve(
                        buffer_handle, source_id=getattr(buffer, "source_id", id(buffer))
                    )
                    if id(buffer_handle) not in seen_handles:
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
            value = param_cursor.maybe_replace(component.value)
            return self._state.resolve(value) if value is component.value else value
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
        if node.canonical_id not in self._shape_rewrite_node_ids:
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
        node_id = node.canonical_id
        if node_id not in self._shape_rewrite_node_ids and node_id not in self._reshape_node_ids:
            return args, kwargs
        rewritten_args = self.graph.shape_program.rewrite(node_id, args, self._shape_binding)
        rewritten_kwargs = self.graph.shape_program.rewrite(node_id, kwargs, self._shape_binding)
        if node_id not in self._reshape_node_ids:
            return rewritten_args, rewritten_kwargs
        runtime_shape = self.graph.shape_program.value_shape(
            node_id,
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
        param_cursor = (
            _LiveParamCursor(self._param_handles_for_node(node))
            if node.param_refs
            else self._empty_param_cursor
        )
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
        args, kwargs = self._rewrite_dynamic_call_args(node, args, kwargs)
        return _rewrite_placement_device_args(node, args, kwargs, self.placement)

    def _execute_func(
        self,
        node: SplitTraceNode,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        fast_guard: bool = False,
    ) -> Any:
        """Execute one captured operation function."""

        if node.target is None:
            raise SplitUnsupportedError(
                f"{node.label!r} has no callable target for split replay.",
                context=self._context(node, "missing callable target"),
            )
        try:
            # Factory calls without a device argument must also allocate on
            # the segment's device. Explicit captured destinations are handled
            # separately during argument reconstruction above.
            if fast_guard:
                output = node.target(*args, **kwargs)
            elif self.placement.is_explicit:
                with _torch().device(self.placement.device):
                    output = self._invoke_func(node, args, kwargs)
            else:
                output = self._invoke_func(node, args, kwargs)
        except Exception as exc:
            raise SplitUnsupportedError(
                f"Torch replay failed at {node.canonical_id!r} ({node.op_type}): {exc}",
                context=self._context(node, "backend replay execution failed"),
            ) from exc
        if output is None and args:
            return args[0]
        return output

    def _invoke_func(
        self, node: SplitTraceNode, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Call one target under only the state guards it actually needs."""

        target = node.target
        assert target is not None
        autocast_state = getattr(node.op, "func_autocast_state", None)
        if self._can_skip_rng_guard(target, args, kwargs):
            if self._can_skip_autocast_restore(autocast_state):
                return target(*args, **kwargs)
            with AutocastRestore(autocast_state or {}):
                return target(*args, **kwargs)
        return execute_with_restored_rng_autocast(
            target,
            args,
            kwargs,
            rng_states=getattr(node.op, "func_rng_states", None),
            autocast_state=autocast_state,
        )

    def _can_skip_rng_guard(
        self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> bool:
        """Use the pure-op path only when custom dispatch cannot alter it."""

        rng_free = id(target) in getattr(self, "_rng_free_target_ids", frozenset())
        if target is getattr(self, "_dropout_target", None):
            rng_free = (args[2] if len(args) > 2 else kwargs.get("train")) is False
        elif target is getattr(self, "_attention_target", None):
            dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
            rng_free = type(dropout_p) in (int, float) and dropout_p == 0
        if not rng_free:
            return False
        if not _plain_torch_arguments(args, _torch()) or not _plain_torch_arguments(
            kwargs, _torch()
        ):
            return False
        try:
            function_modes = self._function_mode_stack_len()
            dispatch_modes = self._dispatch_mode_stack()
        except Exception:
            return False
        return function_modes == 0 and dispatch_modes == []

    @staticmethod
    def _can_skip_autocast_restore(autocast_state: dict[str, Any] | None) -> bool:
        """Skip disabled autocast contexts when the caller is already disabled."""

        for device, state in (autocast_state or {}).items():
            if device.startswith("__"):
                continue
            if state["enabled"] or autocast_is_enabled(device):
                return False
        return True

    def _source_value(self, node: SplitTraceNode) -> Any:
        """Return a replay value for an input/buffer/source node."""

        if self.use_live_param_sources and node.buffer_refs:
            buffer = node.buffer_refs[0]
            handle = getattr(buffer, "handle", buffer)
            return self._state.resolve(handle, source_id=getattr(buffer, "source_id", id(buffer)))
        value = getattr(node.op, "out", None)
        if value is None:
            raise SplitUnsupportedError(
                f"{node.label!r} source value is unavailable.",
                context=self._context(node, "missing source value"),
            )
        return self._state.resolve(value)

    @staticmethod
    def _is_replay_source_node(node: SplitTraceNode) -> bool:
        """Return whether a target-less node is safe to seed from trace payload."""

        return node.is_buffer or (not node.parents and not node.is_output)

    def _execute_nodes(self, overlay: dict[str, Any]) -> dict[str, Any]:
        """Execute this segment's node set into ``overlay``."""

        execute = self._execute_func
        fast_guard = getattr(
            execute, "__func__", None
        ) is _GeneratedSegmentBase._execute_func and self._can_fast_execute(overlay.values())
        torch = _torch() if fast_guard else None
        for node, group, executor in self._execution_steps:
            if group is None:
                if node.canonical_id not in overlay:
                    value = self._source_value(node)
                    overlay[node.canonical_id] = value
                    if fast_guard and not _plain_torch_arguments(value, torch):
                        fast_guard = False
                for value_id in self._release_after.get(node.canonical_id, ()):
                    overlay.pop(value_id, None)
                continue
            if executor is None:
                raise SplitUnsupportedError(
                    f"{node.label!r} has no callable target for split replay.",
                    context=self._context(node, "missing callable target"),
                )
            args, kwargs = self._reconstruct_args(executor, overlay)
            if fast_guard:
                output = execute(executor, args, kwargs, fast_guard=True)
            else:
                output = execute(executor, args, kwargs)
            for member in group:
                value = _slice_output_by_path(output, member.output_container_path)
                overlay[member.canonical_id] = value
                if fast_guard and not _plain_torch_arguments(value, torch):
                    fast_guard = False
            # Do not let loop locals retain the previous call's input/output
            # storage during the next kernel. Autograd keeps what it needs
            # independently when this is a graph-connected training segment.
            del args, kwargs, output, value
            for value_id in self._release_after.get(node.canonical_id, ()):
                overlay.pop(value_id, None)
        return overlay


class GeneratedPrefix(_GeneratedSegmentBase):
    """Generated-eager Torch prefix segment."""

    segment = "prefix"

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
                    reason="input count mismatch",
                ),
            )
        overlay = dict(zip(self.graph.input_node_ids, input_leaves))
        if self.graph.shape_program is not None:
            self._shape_binding = self.graph.shape_program.bind_flat_values(
                input_leaves,
                shape_of=lambda value: tuple(int(dim) for dim in value.shape),
                backend="torch",
                split_point=self.spec.boundary,
            )
            self.graph.shape_program.require_batch_resolvable(
                self._shape_binding.batch_size,
                self.node_ids,
                backend="torch",
                split_point=self.spec.boundary,
            )
        # The detached path owns this policy: detaching only at the boundary
        # would still save every prefix activation during forward. Keep normal
        # tensors here because public run_suffix() may build an autograd graph
        # directly from this boundary (inference tensors cannot be saved).
        del input_leaves, inputs, input_kwargs
        with torch.no_grad() if detach_boundary else nullcontext():
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

    segment = "suffix"

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
                backend="torch",
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
        if node.parents:
            raise SplitUnsupportedError(
                f"Final output {node.canonical_id!r} has no available replay parent.",
                context=self._context(node, "missing output replay value"),
            )
        # A declared parent-less output is a constant source, not a cached
        # substitute for a missing executed value. Keep its explicit payload.
        return self._source_value(node)

    def _reconstruct_output(self, overlay: dict[str, Any]) -> Any:
        """Reconstruct the traced model output container."""

        output_nodes = [self._node_by_id[node_id] for node_id in self.graph.output_node_ids]
        if not output_nodes:
            raise SplitUnsupportedError(
                "Torch split replay requires explicitly recorded final outputs.",
                context=SplitErrorContext(
                    backend="torch",
                    split_point=self.spec.boundary,
                    reason="missing output records",
                ),
            )
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
    supports_state_placement = True
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

    def device_of(self, value: Any) -> Any:
        """Return the device of a torch tensor."""

        return getattr(value, "device", None)

    def normalize_device(self, device: Any) -> Any:
        """Resolve device aliases using the active Torch device context."""

        torch = _torch()
        normalized = torch.device(device)
        if normalized.type == "cuda" and normalized.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        if normalized.type == "cpu":
            return torch.device("cpu")
        return normalized

    def replicate_state(self, value: Any, device: Any, *, trainable: bool) -> Any:
        """Create a device-local replica that can own its own gradients."""

        # A device transfer already allocates independent storage. copy=True
        # also gives independent ownership on the source device, without first
        # allocating a redundant full-size source clone on cross-device moves.
        replica = value.detach().to(device=device, copy=True)
        if trainable and hasattr(replica, "requires_grad_"):
            replica.requires_grad_(True)
        return replica

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
        replica_pool = _StateReplicaPool()
        prefix_state = SegmentState(
            adapter=self, placement=spec.placement.prefix, replica_pool=replica_pool
        )
        prefix = GeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.prefix_node_ids,
            use_live_param_sources=use_live,
            state=prefix_state,
        )
        training_spec = replace(
            spec,
            features=replace(spec.features, training=True, live_param_sources=True),
        )
        # Detaching a boundary must not create another copy of live prefix weights.
        # Captured-state inference remains separate when live sources were disabled.
        training_prefix = GeneratedPrefix(
            graph=graph,
            plan=plan,
            spec=training_spec,
            node_ids=plan.prefix_node_ids,
            use_live_param_sources=True,
            state=(
                prefix._state
                if use_live
                else SegmentState(
                    adapter=self, placement=spec.placement.prefix, replica_pool=replica_pool
                )
            ),
        )
        suffix = GeneratedSuffix(
            graph=graph,
            plan=plan,
            spec=spec,
            node_ids=plan.suffix_node_ids,
            use_live_param_sources=use_live,
            state=SegmentState(
                adapter=self, placement=spec.placement.suffix, replica_pool=replica_pool
            ),
        )
        return SegmentBundle(prefix=prefix, training_prefix=training_prefix, suffix=suffix)


__all__ = [
    "GeneratedPrefix",
    "GeneratedSuffix",
    "SegmentBundle",
    "TorchSplitAdapter",
]
