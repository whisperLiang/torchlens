"""Public TorchLens exception classes."""

from __future__ import annotations

import importlib
from typing import Any

from ._base import (
    CaptureError,
    CompatibilityError,
    ConfigurationError,
    DiagnosticSeverityError,
    InterventionError,
    ScalarEscapeWarning,
    Severity,
    TorchLensError,
    TorchLensWarning,
    TraceNotReproducibleWarning,
    ValidationError,
)
from .runnable import (
    CollectiveBoundaryReplayError,
    NumericAttestationError,
    PathDivergenceError,
    PoisonedRunError,
    ReattachError,
    RunCapabilityUnavailableError,
    RunnablePreflightError,
    RunnableTLSPECError,
    RunPreconditionError,
    RuntimeSignatureDriftError,
    StateBindingError,
)

_LEGACY_EXCEPTION_PATHS = {
    "AmbiguousOpLookupError": ("torchlens._errors", "AmbiguousOpLookupError"),
    "MutatedReferenceError": ("torchlens._errors", "MutatedReferenceError"),
    "OutputAttributionError": ("torchlens._errors", "OutputAttributionError"),
    "TorchLensCaptureGapError": ("torchlens._errors", "TorchLensCaptureGapError"),
    "TorchLensCaptureGapWarning": ("torchlens._errors", "TorchLensCaptureGapWarning"),
    "PostTraceParamUnavailable": ("torchlens._errors", "PostTraceParamUnavailable"),
    "ShapeInferenceError": ("torchlens._errors", "ShapeInferenceError"),
    "TorchLensPostfuncError": ("torchlens._errors", "TorchLensPostfuncError"),
    "TorchLensIOError": ("torchlens._io", "TorchLensIOError"),
    "UnsupportedTensorVariantError": (
        "torchlens._robustness",
        "UnsupportedTensorVariantError",
    ),
    "TrainingModeConfigError": ("torchlens._training_validation", "TrainingModeConfigError"),
    "RecordingConfigError": ("torchlens.fastlog.exceptions", "RecordingConfigError"),
    "InvalidStorageError": ("torchlens.fastlog.exceptions", "InvalidStorageError"),
    "RecorderStateError": ("torchlens.fastlog.exceptions", "RecorderStateError"),
    "RecoveryError": ("torchlens.fastlog.exceptions", "RecoveryError"),
    "BundleNotFinalizedError": ("torchlens.fastlog.exceptions", "BundleNotFinalizedError"),
    "RecordContextFieldError": ("torchlens.fastlog.exceptions", "RecordContextFieldError"),
    "PredicateError": ("torchlens.fastlog.exceptions", "PredicateError"),
    "TorchLensInterventionError": (
        "torchlens.intervention.errors",
        "TorchLensInterventionError",
    ),
    "TorchLensInterventionWarning": (
        "torchlens.intervention.errors",
        "TorchLensInterventionWarning",
    ),
    "InterventionReadyConflictError": (
        "torchlens.intervention.errors",
        "InterventionReadyConflictError",
    ),
    "DirectActivationWriteWarning": (
        "torchlens.intervention.errors",
        "DirectActivationWriteWarning",
    ),
    "MutateInPlaceWarning": ("torchlens.intervention.errors", "MutateInPlaceWarning"),
    "DirectWriteIgnoredWarning": (
        "torchlens.intervention.errors",
        "DirectWriteIgnoredWarning",
    ),
    "InterventionAuditWarning": (
        "torchlens.intervention.errors",
        "InterventionAuditWarning",
    ),
    "MultiMatchWarning": ("torchlens.intervention.errors", "MultiMatchWarning"),
    "ReplayPreconditionError": (
        "torchlens.intervention.errors",
        "ReplayPreconditionError",
    ),
    "UntrustedCallableError": (
        "torchlens.intervention.errors",
        "UntrustedCallableError",
    ),
    "OpaqueCallableInExecutableSaveError": (
        "torchlens.intervention.errors",
        "OpaqueCallableInExecutableSaveError",
    ),
    "SpecPortabilityError": ("torchlens.intervention.errors", "SpecPortabilityError"),
    "DirectWriteInExecutableSaveError": (
        "torchlens.intervention.errors",
        "DirectWriteInExecutableSaveError",
    ),
    "GraphShapeMismatchError": ("torchlens.intervention.errors", "GraphShapeMismatchError"),
    "ControlFlowDivergenceWarning": (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceWarning",
    ),
    "ControlFlowDivergenceError": (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceError",
    ),
    "EngineDispatchError": ("torchlens.intervention.errors", "EngineDispatchError"),
    "ModelMismatchError": ("torchlens.intervention.errors", "ModelMismatchError"),
    "AppendMismatchError": ("torchlens.intervention.errors", "AppendMismatchError"),
    "AppendStreamingNotSupportedError": (
        "torchlens.intervention.errors",
        "AppendStreamingNotSupportedError",
    ),
    "AppendBatchDependenceError": (
        "torchlens.intervention.errors",
        "AppendBatchDependenceError",
    ),
    "AppendStateValidationWarning": (
        "torchlens.intervention.errors",
        "AppendStateValidationWarning",
    ),
    "MultiOutputModuleError": ("torchlens.intervention.errors", "MultiOutputModuleError"),
    "BatchNormTrainModeWarning": (
        "torchlens.intervention.errors",
        "BatchNormTrainModeWarning",
    ),
    "SpecMutationError": ("torchlens.intervention.errors", "SpecMutationError"),
    "SiteResolutionError": ("torchlens.intervention.errors", "SiteResolutionError"),
    "SiteAmbiguityError": ("torchlens.intervention.errors", "SiteAmbiguityError"),
    "RecursiveTracingError": ("torchlens.intervention.errors", "RecursiveTracingError"),
    "AxisAmbiguityError": ("torchlens.intervention.errors", "AxisAmbiguityError"),
    "SpliceModuleDtypeError": ("torchlens.intervention.errors", "SpliceModuleDtypeError"),
    "SpliceModuleDeviceError": ("torchlens.intervention.errors", "SpliceModuleDeviceError"),
    "HookSignatureError": ("torchlens.intervention.errors", "HookSignatureError"),
    "HookValueError": ("torchlens.intervention.errors", "HookValueError"),
    "HookSiteCoverageError": ("torchlens.intervention.errors", "HookSiteCoverageError"),
    "LiveModeLabelError": ("torchlens.intervention.errors", "LiveModeLabelError"),
    "BundleMemberError": ("torchlens.intervention.errors", "BundleMemberError"),
    "BundleRelationshipError": ("torchlens.intervention.errors", "BundleRelationshipError"),
    "BaselineUndeterminedError": (
        "torchlens.intervention.errors",
        "BaselineUndeterminedError",
    ),
    "NoParentError": ("torchlens.intervention.errors", "NoParentError"),
    "DeadParentError": ("torchlens.intervention.errors", "DeadParentError"),
    "MetadataInvariantError": ("torchlens.validation.invariants", "MetadataInvariantError"),
}

_LAZY_EXCEPTION_PATHS = {
    **_LEGACY_EXCEPTION_PATHS,
    # Resolved lazily like the legacy names, but for the opposite reason: the
    # defining modules import ``errors._base``, so eagerly importing them here
    # would create a cycle.
    "ArtifactSchemaAgeWarning": ("torchlens._io", "ArtifactSchemaAgeWarning"),
    "ArtifactVersionBelowFloorError": ("torchlens._io", "ArtifactVersionBelowFloorError"),
    "ArgumentConflictError": ("torchlens._errors", "ArgumentConflictError"),
    "ArgumentTypeError": ("torchlens._errors", "ArgumentTypeError"),
    "BackendAmbiguityError": ("torchlens.backends", "BackendAmbiguityError"),
    "BackendCapabilityConformanceError": (
        "torchlens.backends",
        "BackendCapabilityConformanceError",
    ),
    "BackendMismatchError": ("torchlens.backends", "BackendMismatchError"),
    "BackendPayloadUnsupportedError": (
        "torchlens.backends",
        "BackendPayloadUnsupportedError",
    ),
    "BackendRegistryError": ("torchlens.backends", "BackendRegistryError"),
    "BackendRuntimeCompatibilityError": (
        "torchlens.backends",
        "BackendRuntimeCompatibilityError",
    ),
    "BackendUnsupportedError": ("torchlens.backends", "BackendUnsupportedError"),
    "CaptureContextError": ("torchlens._errors", "CaptureContextError"),
    "InvalidArgumentError": ("torchlens._errors", "InvalidArgumentError"),
    "UnknownBackendError": ("torchlens.backends", "UnknownBackendError"),
    "DistributedCaptureUnsupportedError": (
        "torchlens._distributed",
        "DistributedCaptureUnsupportedError",
    ),
    "SaveBudgetExceededError": ("torchlens._save_budget", "SaveBudgetExceededError"),
    "CaptureOutcomeError": ("torchlens.capture.outcome", "CaptureOutcomeError"),
    "StopSignalSwallowedError": ("torchlens.capture.outcome", "StopSignalSwallowedError"),
    "PartialCaptureLookupError": ("torchlens.partial", "PartialCaptureLookupError"),
}


def __getattr__(name: str) -> Any:
    """Resolve lazily-bound exception names from their defining modules.

    Parameters
    ----------
    name:
        Public exception name requested from ``torchlens.errors``.

    Returns
    -------
    Any
        Exception or warning class matching ``name``.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the public error surface.
    """

    if name in _LAZY_EXCEPTION_PATHS:
        class_module, attr_name = _LAZY_EXCEPTION_PATHS[name]
        module_obj = importlib.import_module(class_module)
        return getattr(module_obj, attr_name)
    raise AttributeError(f"module 'torchlens.errors' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return visible ``torchlens.errors`` attributes.

    Returns
    -------
    list[str]
        Sorted eager globals plus lazily-resolved exception names.
    """

    return sorted([*globals(), *_LAZY_EXCEPTION_PATHS])


__all__ = [
    "CaptureError",
    "CompatibilityError",
    "ConfigurationError",
    "DiagnosticSeverityError",
    "InterventionError",
    "NumericAttestationError",
    "PathDivergenceError",
    "CollectiveBoundaryReplayError",
    "PoisonedRunError",
    "ReattachError",
    "RunCapabilityUnavailableError",
    "RunPreconditionError",
    "RunnablePreflightError",
    "RunnableTLSPECError",
    "RuntimeSignatureDriftError",
    "ScalarEscapeWarning",
    "Severity",
    "StateBindingError",
    "TorchLensError",
    "TorchLensWarning",
    "TraceNotReproducibleWarning",
    "ValidationError",
    *_LAZY_EXCEPTION_PATHS,
]
