"""Internal backend-neutral IR for TorchLens capture unification."""

from __future__ import annotations

from .capture_events import (
    CaptureEvents,
    LiveOpRecord,
    live_record_for_label,
    register_live_event,
    replace_op_event,
)
from .container import ContainerSpec
from .container_registry import (
    ContainerLeafOccurrence,
    ContainerRecord,
    ContainerRegistry,
    ContainerSnapshot,
    FuncSite,
    IdentityEntry,
    ModelSite,
    ModuleSite,
    Phase,
    Role,
    WalkResult,
    walk_container,
)
from .events import (
    ArgTemplateRef,
    BackwardPassEnd,
    BackwardPassStart,
    BlobRef,
    BufferWriteEvent,
    ConditionalEvent,
    EdgeUseKind,
    FunctionCallRef,
    GradFnDiscovered,
    GradFnFired,
    InterventionState,
    ModuleEnterEvent,
    ModuleExitEvent,
    ModuleFrame,
    ModulePrepEvent,
    OpEvent,
    OpEventKind,
    OpGradObserved,
    OutputRef,
    OutputVersionEvent,
    ParamGradObserved,
    ParentEdge,
    PreHookProvenanceEvent,
    edge_use_kind,
    is_control_edge_use,
    is_value_edge_use,
)
from .intervention import FireResult, FunctionEventInput, InterventionTemplateRef
from .live_index import LiveIndex, LiveIndexWindowError
from .predicate import (
    MLXValueUnavailableError,
    RecordContext,
    _DEFERRED_VALUE,
    coerce_deferred_value,
    is_deferred_value,
)
from .refs import DeferredRef, DeviceRef, DtypeRef, ParamRef, ReservedLabel, TensorRef
from .semantics import BackendSemantics, CapturePolicy
from .trace_build_state import TraceBuildState

__all__ = [
    "ArgTemplateRef",
    "BackendSemantics",
    "BackwardPassEnd",
    "BackwardPassStart",
    "BlobRef",
    "BufferWriteEvent",
    "CaptureEvents",
    "CapturePolicy",
    "ConditionalEvent",
    "ContainerSpec",
    "ContainerLeafOccurrence",
    "ContainerRecord",
    "ContainerRegistry",
    "ContainerSnapshot",
    "DeferredRef",
    "DeviceRef",
    "DtypeRef",
    "EdgeUseKind",
    "edge_use_kind",
    "FireResult",
    "FunctionCallRef",
    "FunctionEventInput",
    "FuncSite",
    "IdentityEntry",
    "InterventionState",
    "InterventionTemplateRef",
    "is_control_edge_use",
    "is_value_edge_use",
    "GradFnDiscovered",
    "GradFnFired",
    "LiveOpRecord",
    "LiveIndex",
    "LiveIndexWindowError",
    "MLXValueUnavailableError",
    "ModuleEnterEvent",
    "ModuleExitEvent",
    "ModuleFrame",
    "ModulePrepEvent",
    "ModelSite",
    "ModuleSite",
    "OpEvent",
    "OpEventKind",
    "OpGradObserved",
    "OutputRef",
    "OutputVersionEvent",
    "ParamGradObserved",
    "ParamRef",
    "ParentEdge",
    "PreHookProvenanceEvent",
    "Phase",
    "RecordContext",
    "ReservedLabel",
    "Role",
    "WalkResult",
    "walk_container",
    "TensorRef",
    "TraceBuildState",
    "_DEFERRED_VALUE",
    "coerce_deferred_value",
    "is_deferred_value",
    "live_record_for_label",
    "register_live_event",
    "replace_op_event",
    # Deprecated inert shims (warn on access; see __getattr__ below).
    "BufferEvent",
    "ModuleEvent",
]


def __getattr__(name: str) -> object:
    """Return deprecated event-type shims removed from the live IR.

    Parameters
    ----------
    name
        Attribute name requested from :mod:`torchlens.ir`.

    Returns
    -------
    object
        Inert compatibility class from :mod:`torchlens.ir._deprecated`.

    Raises
    ------
    AttributeError
        If ``name`` is not a deprecated compatibility export.
    """

    if name in ("ModuleEvent", "BufferEvent"):
        import warnings

        from . import _deprecated

        warnings.warn(
            f"torchlens.ir.{name} is deprecated: TorchLens no longer emits this "
            "event kind (module containment uses ModuleEnterEvent/ModuleExitEvent; "
            "buffer capture uses BufferWriteEvent). The class remains importable "
            "as an inert compatibility shim.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_deprecated, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
