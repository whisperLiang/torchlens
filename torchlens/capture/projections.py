"""Fastlog projections over unified capture events."""

from __future__ import annotations

import traceback
import weakref
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, cast

import torch

from ..utils._torch_compat import tensor_version_or_none
from ..utils._torch_symbols import torch_attr

from ..fastlog.exceptions import PredicateError
from ..fastlog.types import (
    ActivationRecord,
    CaptureSpec,
    GradRecordContext,
    ModuleStackFrame,
    PredicateFailure,
    RecordContext,
    Recording,
    StorageIntent,
)
from ..ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ..ir.refs import DeviceRef, DtypeRef, TensorRef
from ..ir.predicate import EventKind, coerce_deferred_value
from ..ir.semantics import BackendSemantics, CapturePolicy
from ..utils.tensor_utils import get_memory_amount_from_metadata

if TYPE_CHECKING:
    from ..fastlog.options import RecordingOptions
    from ..data_classes.trace import Trace
    from ..ir.op_record import OpRecord

_active_recording_state: "RecordingState | None" = None

_EMPTY_ARG_TEMPLATE_REF = ArgTemplateRef(
    saved_args=None,
    saved_kwargs=None,
    args_template=None,
    kwargs_template=None,
    has_saved_args=False,
)
_EMPTY_BACKEND_SEMANTICS = BackendSemantics(
    backend_grad_handle=None,
    grad_fn_class_name=None,
    autograd_memory=0,
    num_autograd_tensors=0,
    mutated_input_positions=(),
    aliased_output_inputs=(),
    unknown_aliasing=False,
    bytes_delta_at_call=None,
    bytes_peak_at_call=None,
)
_CAPTURE_POLICY_CACHE: dict[tuple[bool, bool, str], CapturePolicy] = {}


class _GradFnContextMap:
    """Map autograd nodes to record contexts with weak keys when supported."""

    def __init__(self) -> None:
        """Initialize weak-key and object-key fallback storage."""

        self._weak: weakref.WeakKeyDictionary[Any, RecordContext] = weakref.WeakKeyDictionary()
        self._strong: dict[Any, RecordContext] = {}

    def __setitem__(self, key: Any, value: RecordContext) -> None:
        """Store a context by grad_fn_handle object."""

        try:
            self._weak[key] = value
        except TypeError:
            self._strong[key] = value

    def get(self, key: Any, default: RecordContext | None = None) -> RecordContext | None:
        """Return the context for a grad_fn_handle object, if present."""

        try:
            value = self._weak.get(key)
        except TypeError:
            value = None
        if value is not None:
            return value
        return self._strong.get(key, default)


class _StorageBackend(Protocol):
    """Protocol implemented by fastlog storage backends."""

    def append(self, record: ActivationRecord) -> None:
        """Append one retained record."""

    def resolve_payloads(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        intent: StorageIntent,
        *,
        options: "RecordingOptions",
        ctx: RecordContext | GradRecordContext | None,
        kind: str = "activation",
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve payloads for one selected tensor."""

    def finalize(self) -> None:
        """Finalize storage."""

    def abort(self, reason: str) -> None:
        """Abort storage."""


def _resolve_storage_intent(options: "RecordingOptions") -> StorageIntent:
    """Resolve storage destinations from StreamingOptions."""

    if options.streaming is None or options.streaming.bundle_path is None:
        return StorageIntent(in_ram=True, on_disk=False)
    return StorageIntent(
        in_ram=options.streaming.retain_in_memory,
        on_disk=True,
    )


def _empty_recording(options: "RecordingOptions") -> Recording:
    """Create an empty lazy Recording for a predicate capture session."""

    return Recording(
        records=[],
        by_pass={},
        by_label={},
        by_address={},
        orphan_records=[],
        bundle_path=(
            None
            if options.streaming is None or options.streaming.bundle_path is None
            else Path(options.streaming.bundle_path)
        ),
        n_ops=0,
        start_times=[],
        end_times=[],
        predicate_failures=[],
        predicate_failure_overflow_count=0,
        halted=False,
        halt_reason=None,
        halts_by_pass={},
        keep_op_repr=repr(options.keep_op) if options.keep_op is not None else None,
        history_size=options.history_size,
        save_grads_repr=repr(options.save_grads) if options.save_grads is not None else None,
        _activation_transform_repr=(
            repr(options.activation_transform) if options.activation_transform is not None else None
        ),
        _grad_transform_repr=(
            repr(options.grad_transform) if options.grad_transform is not None else None
        ),
    )


@dataclass(slots=True)
class RecordingState:
    """Mutable state for one active predicate recording pass."""

    options: "RecordingOptions"
    recording: Recording
    history: deque[RecordContext] = field(default_factory=deque)
    op_counts: dict[str, int] = field(default_factory=dict)
    module_stack: list[ModuleStackFrame] = field(default_factory=list)
    predicate_failures: list[PredicateFailure] = field(default_factory=list)
    predicate_failure_overflow_count: int = 0
    error_slot: BaseException | None = None
    sample_id: str | int | None = None
    pass_index: int = 0
    event_index: int = 0
    step_index: int = 0
    no_tensor_capture: bool = False
    all_contexts: list[RecordContext] = field(default_factory=list)
    storage_intent: StorageIntent = field(init=False)
    storage_backend: _StorageBackend = field(init=False)
    grad_fn_to_context: _GradFnContextMap = field(default_factory=_GradFnContextMap)
    runtime_trace: "Trace | None" = None
    active_save_grads_record_policy: Any | None = None
    intervene_selector_fire_count: int = 0
    module_event_fields: dict[
        tuple[ModuleStackFrame, ...],
        tuple[tuple[ModuleFrame, ...], tuple[tuple[str, int], ...]],
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize derived storage policy."""

        self.storage_intent = _resolve_storage_intent(self.options)
        if self.storage_intent.on_disk:
            from ..fastlog.storage_disk import DiskStorageBackend

            self.storage_backend = DiskStorageBackend(self.options, self.recording)
        else:
            from ..fastlog.storage_ram import RamStorageBackend

            self.storage_backend = RamStorageBackend(self.recording)

    def append_context(self, ctx: RecordContext) -> None:
        """Append an event context to the bounded sliding window."""

        self.all_contexts.append(ctx)
        window_size = self.options.lookback or self.options.history_size
        if window_size == 0:
            return
        self.history.append(ctx)
        while len(self.history) > window_size:
            self.history.popleft()

    def add_record(self, record: ActivationRecord) -> None:
        """Append a retained out record and update indexes."""

        self.storage_backend.append(record)

    def module_fields_for(
        self,
        ctx: RecordContext,
    ) -> tuple[tuple[ModuleFrame, ...], tuple[tuple[str, int], ...]]:
        """Return cached event module projections for a record context.

        Parameters
        ----------
        ctx
            Predicate context carrying a frozen module stack.

        Returns
        -------
        tuple[tuple[ModuleFrame, ...], tuple[tuple[str, int], ...]]
            IR module frames and legacy module membership tuples.
        """

        stack = cast(tuple[ModuleStackFrame, ...], ctx.module_stack)
        cached = self.module_event_fields.get(stack)
        if cached is not None:
            return cached
        frames = _module_frames_from_record_context(ctx)
        modules = tuple((frame.address, frame.call_index) for frame in frames)
        projected = (frames, modules)
        self.module_event_fields[stack] = projected
        return projected

    def resolve_storage(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        *,
        ctx: RecordContext | GradRecordContext | None = None,
        kind: str = "activation",
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve payloads through the active storage backend."""

        if self.no_tensor_capture:
            return None, None, None, None
        return self.storage_backend.resolve_payloads(
            tensor,
            spec,
            self.storage_intent,
            options=self.options,
            ctx=ctx,
            kind=kind,
        )

    def finalize_storage(self) -> None:
        """Finalize the active storage backend."""

        self.storage_backend.finalize()

    def abort_storage(self, reason: str) -> None:
        """Abort the active storage backend after a failed pass."""

        self.storage_backend.abort(reason)

    def effective_predicate_error_mode(self) -> str:
        """Return the concrete predicate exception policy for this session."""

        if self.options.on_predicate_error != "auto":
            return self.options.on_predicate_error
        return "fail-fast" if self.storage_intent.on_disk else "accumulate"

    def add_predicate_failure(self, ctx: RecordContext, exc: BaseException) -> None:
        """Record a predicate failure subject to the configured cap."""

        failure = PredicateFailure(
            event_index=ctx.event_index,
            kind=cast(EventKind, ctx.kind),
            label=ctx.label,
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        if len(self.predicate_failures) < self.options.max_predicate_failures:
            self.predicate_failures.append(failure)
        else:
            self.predicate_failure_overflow_count += 1
        object.__setattr__(self.recording, "predicate_failures", list(self.predicate_failures))
        object.__setattr__(
            self.recording,
            "predicate_failure_overflow_count",
            self.predicate_failure_overflow_count,
        )

    def handle_predicate_exception(self, ctx: RecordContext, exc: BaseException) -> None:
        """Apply the configured predicate exception policy."""

        self.add_predicate_failure(ctx, exc)
        if self.effective_predicate_error_mode() == "fail-fast":
            raise exc

    def raise_accumulated_predicate_error(self) -> None:
        """Raise a final PredicateError when accumulated failures exist."""

        if self.effective_predicate_error_mode() != "accumulate":
            return
        if not self.predicate_failures and self.predicate_failure_overflow_count == 0:
            return
        details = ""
        if self.predicate_failures:
            details = f": {self.predicate_failures[0].traceback.strip().splitlines()[-1]}"
        raise PredicateError(
            f"fastlog predicate failed during recording{details}",
            failures=list(self.predicate_failures),
            total_count=len(self.predicate_failures) + self.predicate_failure_overflow_count,
            overflow=self.predicate_failure_overflow_count,
        )


def get_active_recording_state() -> RecordingState:
    """Return the active recording state or fail clearly."""

    if _active_recording_state is None:
        raise RuntimeError("fastlog predicate state is not active")
    return _active_recording_state


@contextmanager
def active_recording_state(state: RecordingState) -> Iterator[RecordingState]:
    """Install fastlog recording state for one active logging scope."""

    global _active_recording_state
    previous = _active_recording_state
    _active_recording_state = state
    try:
        yield state
    finally:
        _active_recording_state = previous


def _read_field(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping or object with a default."""

    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _normalize_module_stack(
    module_stack: Iterable[ModuleStackFrame | Mapping[str, Any]] | None,
) -> tuple[ModuleStackFrame, ...]:
    """Normalize module stack input to ModuleStackFrame instances."""

    if module_stack is None:
        return ()
    frames: list[ModuleStackFrame] = []
    for frame in module_stack:
        if isinstance(frame, ModuleStackFrame):
            frames.append(frame)
            continue
        frames.append(
            ModuleStackFrame(
                address=str(frame.get("address", "")),
                module_type=str(frame.get("module_type", "")),
                module_id=int(frame.get("module_id", 0)),
                pass_index=int(frame.get("pass_index", 0)),
            )
        )
    return tuple(frames)


def _recent_ops_for_event(
    recent_events: Sequence[RecordContext],
    include_source_events: bool,
) -> tuple[RecordContext, ...]:
    """Return the operation-visible subset of recent events."""

    visible_kinds = {"op", "input", "buffer"} if include_source_events else {"op"}
    return tuple(event for event in recent_events if event.kind in visible_kinds)


def _build_record_context(
    *,
    kind: str,
    op_log_or_op_data: Any = None,
    module_stack: Iterable[ModuleStackFrame | Mapping[str, Any]] | None = None,
    history: Sequence[RecordContext] = (),
    op_counts: Mapping[str, int] | None = None,
    pass_index: int = 0,
    event_index: int = 0,
    step_index: int | None = None,
    time_since_pass_start: float = 0.0,
    include_source_events: bool = False,
    sample_id: str | int | None = None,
) -> RecordContext:
    """Build the single source-of-truth RecordContext schema."""

    data = op_log_or_op_data
    stack = _normalize_module_stack(module_stack)
    recent_events = tuple(history)
    layer_type = _read_field(data, "layer_type")
    if layer_type is None:
        layer_type = _read_field(data, "type")
    if layer_type is None:
        layer_type = _read_field(data, "func_name")
    if isinstance(layer_type, str):
        layer_type = layer_type.lower().replace("_", "")
    type_index = _read_field(data, "type_index")
    if type_index is None and layer_type is not None and op_counts is not None:
        type_index = op_counts.get(cast(str, layer_type))
    tensor = _read_field(data, "tensor")
    shape = _read_field(data, "shape")
    dtype = _read_field(data, "dtype")
    tensor_device = _read_field(data, "tensor_device")
    tensor_requires_grad = _read_field(data, "tensor_requires_grad")
    if isinstance(tensor, torch.Tensor):
        shape = tuple(tensor.shape)
        dtype = DtypeRef.from_value(tensor.dtype)
        tensor_device = DeviceRef.from_value(tensor.device)
        tensor_requires_grad = tensor.requires_grad
    else:
        dtype = DtypeRef.from_value(dtype)
        tensor_device = DeviceRef.from_value(tensor_device)
    raw_label = _read_field(data, "_label_raw", _read_field(data, "raw_label"))
    label = _read_field(data, "label", raw_label)
    if label is None:
        label = f"{kind}_{event_index}"
    parent_labels = tuple(_read_field(data, "parent_labels", ()))
    return RecordContext(
        kind=kind,
        label=str(label),
        raw_label=raw_label,
        pass_index=pass_index,
        event_index=event_index,
        step_index=step_index,
        layer_type=layer_type,
        type_index=type_index,
        raw_index=_read_field(data, "raw_index"),
        func_name=_read_field(data, "func_name"),
        address=_read_field(data, "address"),
        module_type=_read_field(data, "module_type"),
        module_pass_index=_read_field(data, "module_pass_index"),
        module_stack=stack,
        recent_events=recent_events,
        recent_ops=_recent_ops_for_event(recent_events, include_source_events),
        parent_labels=parent_labels,
        input_output_address=_read_field(data, "input_output_address"),
        shape=shape,
        dtype=dtype,
        tensor_device=tensor_device,
        tensor_requires_grad=tensor_requires_grad,
        output_index=_read_field(data, "output_index"),
        is_bottom_level_func=_read_field(data, "is_bottom_level_func"),
        time_since_pass_start=time_since_pass_start,
        sample_id=sample_id,
        label_raw=str(raw_label) if raw_label is not None else "",
        label_prefix=str(label).rsplit("_", 2)[0] if isinstance(label, str) else "",
        parent_labels_raw=parent_labels,
        is_transform=bool(_read_field(data, "is_transform", False)),
        transform_kind=_read_field(data, "transform_kind"),
        output_of_module_calls=tuple(_read_field(data, "output_of_module_calls", ()) or ()),
    )


def _module_frames_from_record_context(ctx: RecordContext) -> tuple[ModuleFrame, ...]:
    """Convert fastlog module-stack frames to IR module frames."""

    return tuple(
        ModuleFrame(
            address=_module_frame_address(frame),
            address_normalized=None,
            module_type=frame.module_type,
            call_index=frame.pass_index,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for frame in ctx.module_stack
    )


def _module_frame_address(frame: ModuleStackFrame) -> str:
    """Return the Trace-facing address for a predicate module-stack frame.

    Parameters
    ----------
    frame
        Predicate module frame from a capture-time ``RecordContext``.

    Returns
    -------
    str
        Public module address used by postprocess.
    """

    return frame.address or "self"


def _record_context_from_event(event: OpEvent) -> RecordContext:
    """Project one ``OpEvent`` back into the fastlog predicate schema."""

    ctx = getattr(event, "record_context", None)
    if isinstance(ctx, RecordContext):
        return ctx
    tensor = event.output.tensor
    return _build_record_context(
        kind=event.kind,
        op_log_or_op_data={
            "label": event.label_raw,
            "raw_label": event.label_raw,
            "_label_raw": event.label_raw,
            "raw_index": event.raw_index,
            "type": event.layer_type,
            "type_index": event.type_index,
            "func_name": event.function.func_name,
            "parent_labels": tuple(parent.parent_label_raw for parent in event.parents),
            "shape": tensor.shape,
            "dtype": DtypeRef.from_value(tensor.dtype),
            "tensor_device": DeviceRef.from_value(tensor.device),
            "tensor_requires_grad": tensor.requires_grad,
            "output_index": event.output.multi_output_index,
            "is_bottom_level_func": event.is_bottom_level,
        },
        event_index=event.raw_index,
        step_index=event.step_index,
    )


def _capture_policy_from_spec(spec: CaptureSpec) -> CapturePolicy:
    """Return the shared immutable event policy for a capture specification.

    Parameters
    ----------
    spec
        Predicate capture decision.

    Returns
    -------
    CapturePolicy
        Immutable policy shared by equivalent event decisions.
    """

    key = (spec.save_out, spec.keep_grad, spec.save_mode)
    policy = _CAPTURE_POLICY_CACHE.get(key)
    if policy is None:
        policy = CapturePolicy(
            must_keep_topology=False,
            save_payload=spec.save_out,
            requires_isolation=False,
            save_args=False,
            save_code=False,
            save_rng=False,
            save_grad=spec.keep_grad,
            stream=False,
            save_mode=spec.save_mode,
        )
        _CAPTURE_POLICY_CACHE[key] = policy
    return policy


@dataclass(slots=True)
class _SparseFreezeValues:
    """Shared value bundle both sparse freeze shapes construct from."""

    label_raw: str
    tensor_ref: TensorRef
    transformed_ref: TensorRef | None
    module_stack: tuple[ModuleFrame, ...]
    modules: tuple[tuple[str, int], ...]
    is_scalar_bool: bool | None
    bool_value: bool | None


def _sparse_freeze_values(
    ctx: RecordContext,
    *,
    tensor: torch.Tensor | None,
    ram_payload: torch.Tensor | None,
    transformed_ram_payload: torch.Tensor | None,
    module_fields: tuple[
        tuple[ModuleFrame, ...],
        tuple[tuple[str, int], ...],
    ]
    | None,
) -> _SparseFreezeValues:
    """Compute the sparse commit's shared values ONCE for either freeze shape."""

    label_raw = ctx.raw_label or ctx.label
    memory = (
        get_memory_amount_from_metadata(tensor, ctx.shape or tuple(tensor.shape), tensor.dtype)
        if tensor is not None
        else 0
    )
    tensor_requires_grad = cast(bool | None, coerce_deferred_value(ctx.tensor_requires_grad))
    is_scalar_bool = cast(bool | None, coerce_deferred_value(ctx.is_scalar_bool))
    bool_value = cast(bool | None, coerce_deferred_value(ctx.bool_value))
    if module_fields is None:
        module_stack = _module_frames_from_record_context(ctx)
        modules = tuple((frame.address, frame.call_index) for frame in module_stack)
    else:
        module_stack, modules = module_fields
    tensor_ref = TensorRef(
        label_raw=label_raw,
        shape=ctx.shape,
        dtype=str(ctx.dtype) if ctx.dtype is not None else None,
        device=str(ctx.tensor_device) if ctx.tensor_device is not None else None,
        requires_grad=tensor_requires_grad,
        memory=memory,
        payload=ram_payload,
        blob_ref=None,
        backend_handle_id=str(id(tensor)) if tensor is not None else None,
    )
    transformed_ref = None
    if transformed_ram_payload is not None:
        transformed_shape = tuple(transformed_ram_payload.shape)
        transformed_dtype = transformed_ram_payload.dtype
        transformed_memory = get_memory_amount_from_metadata(
            transformed_ram_payload,
            transformed_shape,
            transformed_dtype,
        )
        transformed_ref = TensorRef(
            label_raw=label_raw,
            shape=transformed_shape,
            dtype=str(transformed_dtype),
            device=str(transformed_ram_payload.device),
            requires_grad=transformed_ram_payload.requires_grad,
            memory=transformed_memory,
            payload=transformed_ram_payload,
            blob_ref=None,
            backend_handle_id=str(id(transformed_ram_payload)),
        )
    return _SparseFreezeValues(
        label_raw=label_raw,
        tensor_ref=tensor_ref,
        transformed_ref=transformed_ref,
        module_stack=module_stack,
        modules=modules,
        is_scalar_bool=is_scalar_bool,
        bool_value=bool_value,
    )


def _sparse_function_ref(ctx: RecordContext, function: FunctionCallRef | None) -> FunctionCallRef:
    """Return the sparse commit's function facet (name-only SHELL fallback)."""

    return function or FunctionCallRef(
        func=None,
        func_name=ctx.func_name,
        func_qualname=None,
        func_call_id=None,
        code_context=(),
        func_duration=None,
        flops_forward=None,
        flops_backward=None,
        func_rng_states=None,
        func_autocast_state=None,
        arg_names=(),
        num_args_total=0,
        num_pos_args=0,
        num_kwargs=0,
        non_tensor_pos_args=(),
        non_tensor_kwargs=(),
        func_non_tensor_args=(),
        is_inplace=False,
        func_config=(),
    )


def _sparse_output_ref(
    ctx: RecordContext,
    spec: CaptureSpec,
    values: _SparseFreezeValues,
    *,
    ram_payload: torch.Tensor | None,
    container_path: tuple[Any, ...],
) -> OutputRef:
    """Return the sparse commit's output ref (shared by both freeze shapes)."""

    return OutputRef(
        tensor=values.tensor_ref,
        transformed_tensor=values.transformed_ref,
        has_saved_activation=bool(spec.save_out and (ram_payload is not None)),
        output_device=str(ctx.tensor_device) if ctx.tensor_device is not None else None,
        activation_transform=None,
        detach_saved_activations=not spec.keep_grad,
        visualizer_path=None,
        multi_output_index=ctx.output_index,
        in_multi_output=bool(container_path),
        container_path=container_path,
        container_spec=None,
        child_versions=(),
    )


def _record_from_record_context(
    ctx: RecordContext,
    spec: CaptureSpec,
    *,
    tensor: torch.Tensor | None = None,
    ram_payload: torch.Tensor | None = None,
    transformed_ram_payload: torch.Tensor | None = None,
    predicate_matched: bool,
    backend_semantics: BackendSemantics | None = None,
    function: FunctionCallRef | None = None,
    container_path: tuple[Any, ...] = (),
    module_fields: tuple[
        tuple[ModuleFrame, ...],
        tuple[tuple[str, int], ...],
    ]
    | None = None,
) -> "OpRecord":
    """Sparse-pipeline decomposed freeze: ``OpCore`` + facets, no ``OpEvent``.

    Value computation routes through ``_sparse_freeze_values``; facet PRESENCE mirrors ``op_record_from_event``
    applied to the equivalent compat event (S5: an absent facet is never
    fabricated empty, and a facet is present exactly when the legacy event
    carries non-default values — plus ``graph``/``policy``, which the adapter
    constructs unconditionally).
    """

    from ..ir.op_record import (
        AnnotationsFacet,
        ControlFacet,
        GraphFacet,
        ModulesFacet,
        OpCore,
        OpRecord,
        PolicyFacet,
        RecordingFacet,
    )

    values = _sparse_freeze_values(
        ctx,
        tensor=tensor,
        ram_payload=ram_payload,
        transformed_ram_payload=transformed_ram_payload,
        module_fields=module_fields,
    )
    label_raw = values.label_raw
    core = OpCore(
        seq=0,
        kind=ctx.kind,
        label_raw=label_raw,
        layer_label_raw=label_raw,
        layer_type=ctx.layer_type or ctx.kind,
        raw_index=ctx.raw_index or ctx.event_index,
        type_index=ctx.type_index or 0,
        step_index=ctx.step_index or 0,
        pass_index=ctx.pass_index,
        parents=tuple(
            ParentEdge(parent_label_raw=parent, arg_position=None, edge_use="unknown")
            for parent in ctx.parent_labels
        ),
        output=_sparse_output_ref(
            ctx, spec, values, ram_payload=ram_payload, container_path=container_path
        ),
        is_bottom_level=bool(ctx.is_bottom_level_func),
        func_call_id=ctx.func_call_id,
    )
    annotations_payload = _reference_annotations(spec.save_mode, ram_payload)
    control = (
        ControlFacet(is_scalar_bool=values.is_scalar_bool, bool_value=values.bool_value)
        if values.is_scalar_bool is not None or values.bool_value is not None
        else None
    )
    return OpRecord(
        core=core,
        function=_sparse_function_ref(ctx, function),
        templates=_EMPTY_ARG_TEMPLATE_REF,
        graph=GraphFacet(
            parent_arg_positions={"args": {}, "kwargs": {}},
            is_output_parent=ctx.is_output_parent,
        ),
        modules_facet=(
            ModulesFacet(module_stack=values.module_stack, modules=values.modules)
            if values.module_stack or values.modules
            else None
        ),
        control=control,
        annotations_facet=(
            AnnotationsFacet(annotations=dict(annotations_payload)) if annotations_payload else None
        ),
        policy_facet=PolicyFacet(
            backend_semantics=backend_semantics
            if backend_semantics is not None
            else _EMPTY_BACKEND_SEMANTICS,
            policy=_capture_policy_from_spec(spec),
            predicate_matched=predicate_matched,
            tracing_finished=False,
            construction_done=True,
        ),
        recording=RecordingFacet(record_context=ctx, capture_spec=spec),
    )


# ---------------------------------------------------------------------------
# The ONE commit tail (producer unification 3.1): freeze -> atomic append,
# plus the declared exhaustive-only post-tail stages. The checked-in
# stage-applicability matrix is conformance-asserted by the parity suite.
# ---------------------------------------------------------------------------

# Stage applicability per pre-commit pipeline (DoR 3.1). "tail" rows are
# universal; everything else is declared to exactly one pipeline. The sparse
# pipeline NEVER writes `_capture_parent_edge_truth` (the edge-truth seal is
# not computable on sparse and would arm a deliberately dormant invariant).
COMMIT_STAGE_MATRIX: dict[str, tuple[str, ...]] = {
    "exhaustive": (
        "fire_results_park",
        "module_filter",
        "selector",
        "escrow_candidate",
        "payload_disposition",
        "predicate_saved_args",
        "edge_truth_seal",
        "freeze",
        "append",
        "grad_handle_index",
        "live_op_view",
        "lookback_retention_candidate",
        "nonfinite_check",
    ),
    "sparse": (
        "selector",
        "demanded_enrichment",
        "backend_semantics",
        "payload_disposition",
        "freeze",
        "append",
        "halt_evaluation",
    ),
}


class OpDraft(Protocol):
    """A finished pre-commit pipeline's draft, ready for freeze -> append."""

    pipeline: str

    def freeze(self) -> Any:
        """Materialize this draft into the record ``commit_op`` appends."""


@dataclass(slots=True)
class SparseOpDraft:
    """Sparse-pipeline draft (12 ``append_projected_event`` sites)."""

    ctx: RecordContext
    spec: CaptureSpec
    tensor: torch.Tensor | None
    ram_payload: torch.Tensor | None
    transformed_ram_payload: torch.Tensor | None
    predicate_matched: bool
    backend_semantics: BackendSemantics | None
    function: FunctionCallRef | None
    container_path: tuple[Any, ...]
    module_fields: (
        tuple[
            tuple[ModuleFrame, ...],
            tuple[tuple[str, int], ...],
        ]
        | None
    )

    pipeline: str = field(default="sparse", init=False)

    def freeze(self) -> Any:
        """Construct the journal record ONCE from the final draft state."""

        return _record_from_record_context(
            self.ctx,
            self.spec,
            tensor=self.tensor,
            ram_payload=self.ram_payload,
            transformed_ram_payload=self.transformed_ram_payload,
            predicate_matched=self.predicate_matched,
            backend_semantics=self.backend_semantics,
            function=self.function,
            container_path=self.container_path,
            module_fields=self.module_fields,
        )


def commit_op(trace: Any, draft: OpDraft) -> "LiveOpView | None":
    """The ONE commit tail: freeze -> atomic append (+ exhaustive stages).

    Freeze constructs the journal record ONCE from the final draft (the
    decomposed producer is the only producer since P7); append is the single
    sequencing authority. The
    two post-tail stages (grad-handle side index, ``LiveOpView``) exist only
    on the exhaustive pipeline per ``COMMIT_STAGE_MATRIX`` — the sparse
    pipeline returns ``None`` and pays neither.
    """

    record = draft.freeze()
    trace.capture_events.append(record)
    if draft.pipeline != "exhaustive":
        return None
    grad_fn_handle = draft.grad_fn_handle  # type: ignore[attr-defined]
    if grad_fn_handle is not None:
        trace.capture_events.grad_fn_handles_by_label_raw[record.label_raw] = grad_fn_handle
    return LiveOpView(trace, record)


def append_projected_event(
    trace: Any,
    ctx: RecordContext,
    spec: CaptureSpec,
    *,
    tensor: torch.Tensor | None = None,
    ram_payload: torch.Tensor | None = None,
    transformed_ram_payload: torch.Tensor | None = None,
    predicate_matched: bool,
    backend_semantics: BackendSemantics | None = None,
    function: FunctionCallRef | None = None,
    container_path: tuple[Any, ...] = (),
) -> None:
    """Append one lightweight predicate record to ``trace.capture_events``."""

    if not hasattr(trace, "capture_events"):
        from ..ir import CaptureEvents

        trace.capture_events = CaptureEvents()
    label_raw = ctx.raw_label or ctx.label
    if tensor is not None and ctx.kind == "op":
        from ..backends.torch.tensor_tracking import _add_tensor_backward_hook

        public_label = _public_record_context_label(ctx)
        trace.__dict__.setdefault("_raw_to_final_layer_labels", {})[label_raw] = public_label
        trace.__dict__.setdefault("_fastlog_grad_contexts", {})[public_label] = ctx
        _add_tensor_backward_hook(trace, tensor, label_raw)
    recording_state = _active_recording_state
    module_fields = None if recording_state is None else recording_state.module_fields_for(ctx)
    commit_op(
        trace,
        SparseOpDraft(
            ctx=ctx,
            spec=spec,
            tensor=tensor,
            ram_payload=ram_payload,
            transformed_ram_payload=transformed_ram_payload,
            predicate_matched=predicate_matched,
            backend_semantics=backend_semantics,
            function=function,
            container_path=container_path,
            module_fields=module_fields,
        ),
    )


class LiveOpViewFieldNotYetWritten(AttributeError):
    """Raised when a live op view field is populated only by postprocess."""


_OPLOG_FIELDS_KNOWN_LATE = frozenset(
    {
        "final_out",
        "layer_label",
        "layer_label_short",
        "label",
        "label_short",
        "layer_label",
        "layer_label_short",
    }
)


def _grad_fn_handle_from_index(trace: "Trace", event: OpEvent) -> Any:
    """Read the live autograd handle from its single owner, the journal index.

    grad_fn single ownership (producer unification P2): the journal's
    ``grad_fn_handles_by_label_raw`` side index is the one handle authority.
    The event-field fallback keeps byte-identity for detached streams until
    the compat ``OpEvent`` field dies with the legacy producer.
    """

    events = getattr(trace, "capture_events", None)
    if events is not None:
        handle = events.grad_fn_handles_by_label_raw.get(event.label_raw)
        if handle is not None:
            return handle
    # Compat OpEvents still carry the handle field; decomposed OpRecords never
    # do (single ownership) and read as None here by strict-protocol default.
    return getattr(event, "grad_fn_handle", None)


def _event_live_field(trace: "Trace", event: OpEvent, name: str) -> Any:
    """Return a forward-time field projected from an operation event.

    Parameters
    ----------
    trace
        Active trace owning the live index.
    event
        Operation event to project.
    name
        Op-style field name requested by a capture-time consumer.

    Returns
    -------
    Any
        Event-backed field value.
    """

    output = event.output
    function = event.function
    semantics = event.backend_semantics
    templates = event.templates

    # This is the capture-time equivalent of an Op property lookup.  Constructing
    # the complete Op-shaped dictionary here made every single attribute read walk
    # every parameter, edge, child, and module field and allocate all mutable
    # projections.  Dispatch only the requested column; mutable values remain fresh
    # on every read, preserving the previous adapter semantics.
    if name == "_label_raw":
        return event.label_raw
    if name == "_layer_label_raw":
        return event.layer_label_raw
    if name == "raw_index":
        return event.raw_index
    if name == "step_index":
        return event.step_index
    if name == "source_trace":
        return event.source_trace or trace
    if name == "_tracing_finished":
        return event.tracing_finished
    if name == "_construction_done":
        return event.construction_done
    if name == "type":
        return event.layer_type
    if name == "type_index":
        return event.type_index
    if name == "pass_index":
        return event.pass_index
    if name == "num_passes":
        return 1
    if name == "lookup_keys":
        return []
    if name == "out":
        return output.tensor.payload
    if name == "transformed_out":
        return None if output.transformed_tensor is None else output.transformed_tensor.payload
    if name == "has_saved_activation":
        return output.has_saved_activation
    if name == "activation_transform":
        return output.activation_transform
    if name == "annotations":
        return _event_annotations(event, output.tensor.payload)
    if name == "output_device":
        return output.output_device
    if name == "detach_saved_activations":
        return output.detach_saved_activations
    if name == "has_saved_args":
        return False if templates is None else templates.has_saved_args
    if name == "saved_args":
        return None if templates is None else templates.saved_args
    if name == "saved_kwargs":
        return None if templates is None else templates.saved_kwargs
    if name == "args_template":
        return None if templates is None else templates.args_template
    if name == "kwargs_template":
        return None if templates is None else templates.kwargs_template
    if name == "shape":
        return output.tensor.shape
    if name == "transformed_out_shape":
        return None if output.transformed_tensor is None else output.transformed_tensor.shape
    if name == "dtype":
        return output.tensor.dtype
    if name == "transformed_out_dtype":
        return None if output.transformed_tensor is None else output.transformed_tensor.dtype
    if name == "activation_memory":
        return output.tensor.memory
    if name == "transformed_activation_memory":
        return None if output.transformed_tensor is None else output.transformed_tensor.memory
    if name == "visualizer_path":
        return output.visualizer_path
    if name == "bytes_delta_at_call":
        return semantics.bytes_delta_at_call
    if name == "bytes_peak_at_call":
        return semantics.bytes_peak_at_call
    if name == "autograd_memory":
        return semantics.autograd_memory
    if name == "num_autograd_tensors":
        return semantics.num_autograd_tensors
    if name == "has_out_variations":
        return bool(output.child_versions)
    if name == "out_versions_by_child":
        return dict(output.child_versions)
    if name == "func":
        return function.func
    if name == "func_call_id":
        return function.func_call_id
    if name == "func_name":
        return function.func_name
    if name == "func_qualname":
        return function.func_qualname
    if name == "code_context":
        return list(function.code_context)
    if name == "func_duration":
        return function.func_duration or 0
    if name == "flops_forward":
        return function.flops_forward
    if name == "flops_backward":
        return function.flops_backward
    if name == "func_rng_states":
        return function.func_rng_states
    if name == "func_autocast_state":
        return function.func_autocast_state
    if name == "arg_names":
        return tuple(function.arg_names)
    if name == "num_args_total":
        return function.num_args_total
    if name == "num_pos_args":
        return function.num_pos_args
    if name == "num_kwargs":
        return function.num_kwargs
    if name == "non_tensor_pos_args":
        return list(function.non_tensor_pos_args)
    if name == "non_tensor_kwargs":
        return dict(function.non_tensor_kwargs)
    if name == "func_non_tensor_args":
        return list(function.func_non_tensor_args)
    if name == "is_inplace":
        return function.is_inplace
    if name == "grad_fn_class_name":
        return semantics.grad_fn_class_name
    if name == "grad_fn_class_qualname":
        return event.grad_fn_class_qualname
    if name == "grad_fn_object_id":
        handle = _grad_fn_handle_from_index(trace, event)
        return None if handle is None else id(handle)
    if name == "grad_fn_handle":
        return _grad_fn_handle_from_index(trace, event)
    if name == "grad_fn":
        return None
    if name == "in_multi_output":
        return output.in_multi_output
    if name == "multi_output_index":
        return output.multi_output_index
    if name == "multi_output_name":
        return None
    if name == "container_path":
        return output.container_path
    if name == "container_spec":
        return output.container_spec
    if name == "parent_params":
        return list(event.parent_params)
    if name == "_param_barcodes":
        return [param.barcode for param in event.params]
    if name == "parent_param_ops":
        return {param.barcode: event.pass_index for param in event.params}
    if name == "param_shapes":
        return [param.shape for param in event.params]
    if name == "num_params":
        return sum(0 if param.shape is None else prod(param.shape) for param in event.params)
    if name == "equivalence_class":
        return event.equivalence_class
    if name == "equivalent_ops":
        return {event.label_raw}
    if name == "recurrent_ops":
        return []
    if name == "parents":
        return [edge.parent_label_raw for edge in event.parents]
    if name == "parent_arg_positions":
        return event.parent_arg_positions
    if name == "_edge_uses":
        return list(event._edge_uses)
    if name == "root_ancestors":
        return set(event.root_ancestors)
    if name == "children":
        return list(trace.capture_events.live_index.children(event.label_raw))
    if name == "has_children":
        return bool(trace.capture_events.live_index.children(event.label_raw))
    if name == "is_input":
        return event.kind == "source" and event.layer_type == "input"
    if name == "input_was_parameter":
        return event.input_was_parameter
    if name == "has_input_ancestor":
        return bool(event.input_ancestors)
    if name == "input_ancestors":
        return set(event.input_ancestors)
    if name in {"is_output", "is_final_output", "has_output_descendant", "is_orphan"}:
        return False
    if name == "is_output_parent":
        return event.is_output_parent
    if name == "output_descendants":
        return set()
    if name == "io_role":
        return None
    if name == "is_buffer":
        return event.kind == "source" and event.layer_type == "buffer"
    if name == "is_internal_source":
        return event.layer_type != "input" and not event.parents
    if name == "has_internal_source_ancestor":
        return event.has_internal_source_ancestor
    if name == "internal_source_parents":
        return [
            edge.parent_label_raw
            for edge in event.parents
            if trace.capture_events.live_index.require_event(
                edge.parent_label_raw
            ).has_internal_source_ancestor
        ]
    if name == "internal_source_ancestors":
        return set(event.internal_source_ancestors)
    if name == "is_internal_sink":
        return False
    if name == "is_scalar_bool":
        return event.is_scalar_bool
    if name == "bool_value":
        return event.bool_value
    if name == "module":
        return event.modules[-1] if event.modules else None
    if name == "modules":
        return list(event.modules)
    if name == "module_call_stack":
        return list(trace.capture_events.live_index.module_stack_membership(event.label_raw))
    if name in {"input_to_module_calls", "output_of_modules", "output_of_module_calls"}:
        return []
    if name == "module_entry_arg_keys":
        return defaultdict(list)
    if name in {"is_module_output", "is_atomic_module"}:
        return False
    if name == "atomic_module_call":
        return None
    if name == "interventions":
        return [
            result.fire_record for result in event.fire_results if result.fire_record is not None
        ]
    if name == "intervention_replaced":
        return event.intervention_replaced
    if name == "func_config":
        return dict(function.func_config)
    if name in _OPLOG_FIELDS_KNOWN_LATE:
        raise LiveOpViewFieldNotYetWritten(
            f"LiveOpView.{name!r} is populated by postprocess Step 0; "
            "it is not available inside a forward-time callback."
        )
    raise AttributeError(f"LiveOpView has no attribute {name!r}.")


def _event_annotations(event: OpEvent, payload: Any) -> dict[str, Any]:
    """Return Op annotations projected from an operation event."""

    raw_annotations = event.transform_config.get("_tl_annotations")
    annotations = dict(raw_annotations) if isinstance(raw_annotations, Mapping) else {}
    annotations.update(_reference_annotations(event.policy.save_mode, payload))
    return annotations


def _reference_annotations(save_mode: str, payload: Any) -> dict[str, Any]:
    """Return saved-payload annotations needed by reference-mode tripwires."""

    if save_mode != "reference" or not isinstance(payload, torch.Tensor):
        return {}
    return {
        "save_mode": "reference",
        "saved_out_version": tensor_version_or_none(payload),
    }


class LiveOpView:
    """Read-only Op-shaped adapter over a capture-time operation event."""

    __slots__ = ("_trace_ref", "_record")

    def __init__(self, trace: "Trace", record: OpEvent) -> None:
        """Initialize the live view.

        Parameters
        ----------
        trace
            Active trace that owns the live record.
        record
            Operation event backing this view.
        """

        object.__setattr__(self, "_trace_ref", weakref.ref(trace))
        object.__setattr__(self, "_record", record)

    @property
    def _trace(self) -> "Trace":
        """Return the owning trace while it is still alive.

        Returns
        -------
        Trace
            Active trace.
        """

        trace = object.__getattribute__(self, "_trace_ref")()
        if trace is None:
            raise RuntimeError(
                "LiveOpView outlived its Trace (capture context ended); this view is invalid."
            )
        return trace

    def __getattr__(self, name: str) -> Any:
        """Return a live field value.

        Parameters
        ----------
        name
            Field name to read.

        Returns
        -------
        Any
            Current live field value.
        """

        record = object.__getattribute__(self, "_record")
        return _event_live_field(self._trace, record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject direct mutation of live views.

        Parameters
        ----------
        name
            Field name.
        value
            Ignored attempted value.

        Returns
        -------
        None
            This method always raises.
        """

        raise AttributeError(f"LiveOpView is read-only mid-forward; cannot set {name}")


def activation_record_from_event(event: OpEvent) -> ActivationRecord | None:
    """Project a retained ``OpEvent`` into an ``ActivationRecord``."""

    if not event.predicate_matched:
        return None
    # Events minted outside the predicate storage path (e.g. a replacement
    # boundary op logged for a raw forward hook's injected tensor) carry the
    # constructor default ``capture_spec=None``; they are structural facts
    # with no retained payload, so they project as metadata-only records.
    spec = getattr(event, "capture_spec", None) or CaptureSpec(save_out=False, save_metadata=True)
    ctx = _record_context_from_event(event)
    ram_payload = event.output.tensor.payload if spec.save_out else None
    transformed_ram_payload = (
        event.output.transformed_tensor.payload
        if event.output.transformed_tensor is not None and spec.save_out
        else None
    )
    return ActivationRecord(
        ctx=ctx,
        spec=spec,
        ram_payload=ram_payload if isinstance(ram_payload, torch.Tensor) else None,
        transformed_ram_payload=(
            transformed_ram_payload if isinstance(transformed_ram_payload, torch.Tensor) else None
        ),
    )


def sync_recording_grad_records_from_sidecar(state: RecordingState) -> None:
    """Rebuild fastlog gradient records from the unified backward sidecar.

    Parameters
    ----------
    state:
        Active recording state whose runtime trace owns backward events.

    Returns
    -------
    None
        ``state.recording.grad_records`` and its lookup indexes are replaced.
    """

    from ..fastlog.types import GradientRecord
    from ..ir.events import GradFnFired, OpGradObserved

    trace = state.runtime_trace
    if trace is None:
        return
    state.recording.grad_records.clear()
    state.recording.grad_by_pass.clear()
    state.recording.grad_by_label.clear()
    state.recording.grad_by_grad_fn_label.clear()
    backward_passes = getattr(trace, "backward_pass_logs", {})
    backward_events = tuple(getattr(trace, "backward_events", ()))
    # Public grad-record contexts expose the position of an event within the
    # BACKWARD sidecar (its lane ordinal), not the raw global journal ``seq``:
    # the journal counter now also spans forward events, so raw ``seq`` values
    # would renumber public ``event_index`` metadata with capture-size-dependent
    # gaps. Lane ordinals preserve the historical dense 1..N numbering exactly.
    lane_ordinals = {id(event): ordinal for ordinal, event in enumerate(backward_events, start=1)}
    for event in backward_events:
        ordinal = lane_ordinals[id(event)]
        if not isinstance(event, OpGradObserved):
            if isinstance(event, GradFnFired):
                _maybe_add_grad_fn_metadata_record(state, trace, event, ordinal)
            continue
        if event.payload_ref is None and event.transformed_payload_ref is None:
            continue
        ctx = _grad_record_context_from_op_grad_event(trace, event, backward_passes, ordinal)
        spec = CaptureSpec(
            save_out=event.payload_ref is not None or event.transformed_payload_ref is not None,
            save_metadata=True,
            keep_grad=False,
        )
        state.recording.add_grad_record(
            GradientRecord(
                ctx=ctx,
                spec=spec,
                ram_payload=event.payload_ref
                if isinstance(event.payload_ref, torch.Tensor)
                else None,
                transformed_ram_payload=(
                    event.transformed_payload_ref
                    if isinstance(event.transformed_payload_ref, torch.Tensor)
                    else None
                ),
                metadata={"timestamp": event.timestamp, "seq": ordinal},
                recorded_at=event.timestamp,
            )
        )


def _maybe_add_grad_fn_metadata_record(
    state: RecordingState, trace: "Trace", event: Any, ordinal: int
) -> None:
    """Append a metadata-only grad-fn record when the active policy selects it."""

    from ..fastlog.types import GradientRecord

    grad_fn = getattr(trace, "grad_fn_logs", {}).get(event.object_id)
    if grad_fn is None or getattr(grad_fn, "has_op", False):
        return
    pass_record = getattr(trace, "backward_pass_logs", {}).get(event.pass_index)
    ctx = GradRecordContext(
        label=grad_fn.label,
        grad_fn_class_name=grad_fn.class_name,
        type=grad_fn.type,
        backward_call_index=event.pass_index,
        grad_kind="grad_output",
        has_forward_op=False,
        has_op=False,
        pass_index=event.pass_index,
        order=getattr(pass_record, "order", None),
        event_index=ordinal,
    )
    policy = state.active_save_grads_record_policy
    decision = policy(ctx) if callable(policy) else policy
    if not isinstance(decision, CaptureSpec) or not decision.save_metadata:
        return
    state.recording.add_grad_record(
        GradientRecord(
            ctx=ctx,
            spec=CaptureSpec(save_out=False, save_metadata=True, keep_grad=False),
            metadata={"timestamp": event.timestamp, "seq": ordinal},
            recorded_at=event.timestamp,
        )
    )


def _grad_record_context_from_op_grad_event(
    trace: "Trace",
    event: Any,
    backward_passes: Mapping[int, Any],
    ordinal: int,
) -> GradRecordContext:
    """Build a ``GradRecordContext`` from one op-gradient sidecar event."""

    pass_record = backward_passes.get(event.pass_index)
    if event.op_label not in getattr(trace, "layer_dict_all_keys", {}):
        fastlog_ctx = getattr(trace, "_fastlog_grad_contexts", {}).get(event.op_label)
        if fastlog_ctx is not None:
            return GradRecordContext(
                label=event.op_label,
                grad_fn_class_name="",
                type=fastlog_ctx.layer_type or "op",
                backward_call_index=event.pass_index,
                grad_kind="grad_output",
                grad_output_index=0,
                layer_label=event.op_label,
                op_label=event.op_label,
                module_stack=tuple(getattr(fastlog_ctx, "module_stack", ()) or ()),
                has_forward_op=True,
                has_op=True,
                pass_index=event.pass_index,
                order=getattr(pass_record, "order", None),
                event_index=ordinal,
                shape=event.shape,
                dtype=_torch_dtype_from_string(event.dtype),
                tensor_device=_torch_device_from_string(
                    getattr(fastlog_ctx, "tensor_device", None)
                ),
            )
        return GradRecordContext(
            label=event.op_label,
            grad_fn_class_name="",
            type="op",
            backward_call_index=event.pass_index,
            grad_kind="grad_output",
            has_forward_op=False,
            has_op=False,
            pass_index=event.pass_index,
            order=getattr(pass_record, "order", None),
            event_index=ordinal,
            shape=event.shape,
            dtype=_torch_dtype_from_string(event.dtype),
            tensor_device=None,
        )
    op = trace.layer_dict_all_keys[event.op_label]
    return GradRecordContext(
        label=event.op_label,
        grad_fn_class_name=getattr(op, "grad_fn_class_name", None) or "",
        type=getattr(op, "layer_type", None) or "op",
        backward_call_index=event.pass_index,
        grad_kind="grad_output",
        grad_output_index=0,
        layer_label=event.op_label,
        op_label=event.op_label,
        module_stack=tuple(getattr(op, "module_stack", ()) or ()),
        has_forward_op=True,
        has_op=True,
        pass_index=event.pass_index,
        order=getattr(pass_record, "order", None),
        event_index=ordinal,
        shape=event.shape,
        dtype=_torch_dtype_from_string(event.dtype),
        tensor_device=_torch_device_from_string(getattr(op, "output_device", None)),
    )


def _public_record_context_label(ctx: RecordContext) -> str:
    """Return the compact public label for a predicate-mode op context."""

    if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
        return f"{ctx.layer_type}_{ctx.type_index}"
    return ctx.label


def _torch_dtype_from_string(dtype_name: str | None) -> torch.dtype | None:
    """Return a ``torch.dtype`` for canonical dtype strings when possible."""

    if dtype_name is None:
        return None
    if dtype_name.startswith("torch."):
        dtype_attr = dtype_name.removeprefix("torch.")
        dtype = torch_attr(dtype_attr)  # r47 secD_1: no lazy ``torch.__getattr__``
        if isinstance(dtype, torch.dtype):
            return dtype
    return None


def _torch_device_from_string(device_name: Any) -> torch.device | None:
    """Return a ``torch.device`` from a string-like field when possible."""

    if device_name is None:
        return None
    try:
        return torch.device(str(device_name))
    except (TypeError, RuntimeError):
        return None


def recording_trace_from_events(events: Any) -> tuple[RecordContext, ...]:
    """Project capture events into fastlog ``RecordContext`` objects.

    Reads the amended reducer view: retention amendments rebind outputs and
    policy facts the projected contexts must reflect.
    """

    return tuple(_record_context_from_event(event) for event in events.amended_op_records())
