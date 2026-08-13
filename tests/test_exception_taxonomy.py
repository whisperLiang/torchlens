"""Tests for the TorchLens 2.0 exception taxonomy."""

from __future__ import annotations

import importlib
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from torchlens import errors, user_funcs
from torchlens._io.tlspec import coerce_tlspec_save_level
from torchlens.backends import (
    BackendAmbiguityError,
    BackendMismatchError,
    BackendPayloadUnsupportedError,
    BackendRuntimeCompatibilityError,
    BackendUnsupportedError,
    UnknownBackendError,
)
from torchlens.intervention.errors import SiteResolutionError
from torchlens.intervention.types import InterventionDecision
from torchlens.options import (
    CaptureOptions,
    StreamingOptions,
    VisualizationOptions,
    merge_capture_options,
    merge_visualization_options,
)

BASE_CLASSES = (
    errors.TorchLensError,
    errors.InterventionError,
    errors.CaptureError,
    errors.ConfigurationError,
    errors.CompatibilityError,
    errors.ValidationError,
)

OLD_EXCEPTION_MAPPING: tuple[tuple[str, str, type[BaseException], str], ...] = (
    ("torchlens._errors", "AmbiguousOpLookupError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "InvalidArgumentError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "ArgumentTypeError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "ArgumentConflictError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "KeywordConflictError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "CaptureContextError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "TorchLensPostfuncError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "PostTraceParamUnavailable", errors.CaptureError, "subclass"),
    ("torchlens._io", "TorchLensIOError", errors.CompatibilityError, "subclass"),
    (
        "torchlens._robustness",
        "UnsupportedTensorVariantError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens._training_validation",
        "TrainingModeConfigError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.fastlog.exceptions",
        "RecordingConfigError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.fastlog.exceptions", "RecorderStateError", errors.CaptureError, "subclass"),
    ("torchlens.fastlog.exceptions", "RecoveryError", errors.CaptureError, "subclass"),
    ("torchlens.fastlog.exceptions", "BundleNotFinalizedError", errors.CaptureError, "subclass"),
    (
        "torchlens.fastlog.exceptions",
        "RecordContextFieldError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.fastlog.exceptions", "PredicateError", errors.CaptureError, "subclass"),
    (
        "torchlens.intervention.errors",
        "TorchLensInterventionError",
        errors.InterventionError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "TorchLensInterventionWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "InterventionReadyConflictError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "DirectActivationWriteWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "MutateInPlaceWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "DirectWriteIgnoredWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "InterventionAuditWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "MultiMatchWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ReplayPreconditionError",
        errors.InterventionError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "OpaqueCallableInExecutableSaveError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpecPortabilityError",
        errors.ConfigurationError,
        "alias",
    ),
    (
        "torchlens.intervention.errors",
        "DirectWriteInExecutableSaveError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "GraphShapeMismatchError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "EngineDispatchError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ModelMismatchError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AppendMismatchError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AppendStreamingNotSupportedError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AppendBatchDependenceError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AppendStateValidationWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BatchNormTrainModeWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpecMutationError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SiteResolutionError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SiteAmbiguityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "RecursiveTracingError",
        errors.CaptureError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AxisAmbiguityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpliceModuleDtypeError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpliceModuleDeviceError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "HookSignatureError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.intervention.errors", "HookValueError", errors.InterventionError, "subclass"),
    (
        "torchlens.intervention.errors",
        "HookSiteCoverageError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "LiveModeLabelError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BundleMemberError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BundleRelationshipError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BaselineUndeterminedError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.intervention.errors", "NoParentError", errors.ConfigurationError, "subclass"),
    ("torchlens.intervention.errors", "DeadParentError", errors.ConfigurationError, "subclass"),
    (
        "torchlens.validation.invariants",
        "MetadataInvariantError",
        errors.ValidationError,
        "subclass",
    ),
    ("torchlens._errors", "BackwardStreamUnavailableError", errors.CaptureError, "subclass"),
    (
        "torchlens.intervention.errors",
        "NonExecutableSpecError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "UnserializableDictKeyError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BatchChunkInputAmbiguityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ChunkedForwardConfigError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SelectorCompositionError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SelectorCapabilityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "UnclassifiedSelectorError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "HelperMountError",
        errors.ConfigurationError,
        "subclass",
    ),
)


# Builtin-lineage golden: the EXACT builtin-exception memberships of every
# public ``torchlens.errors`` class. The error-refusal contract promises that
# typed refusals "retain their historical built-in exception compatibility";
# the 2026-08 ArgumentConflictError incident (17 historically-TypeError doors
# silently reparented onto ValueError) happened because nothing pinned that
# promise. Any reparenting of a public exception class must consciously edit
# this table in the same change as the code, glossary, and contract doc.
_LINEAGE_PROBE_BUILTINS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    RuntimeError,
    Warning,
)

BUILTIN_LINEAGE_GOLDEN: dict[str, tuple[str, ...]] = {
    "AmbiguousOpLookupError": ("ValueError",),
    "AppendBatchDependenceError": ("ValueError",),
    "AppendMismatchError": ("ValueError",),
    "AppendStateValidationWarning": ("Warning",),
    "AppendStreamingNotSupportedError": ("ValueError",),
    "ArgumentConflictError": ("ValueError",),
    "ArgumentTypeError": ("TypeError",),
    "ArtifactSchemaAgeWarning": ("Warning",),
    "ArtifactVersionBelowFloorError": ("RuntimeError",),
    "AxisAmbiguityError": ("ValueError",),
    "BackendAmbiguityError": ("ValueError",),
    "BackendCapabilityConformanceError": ("ValueError", "RuntimeError"),
    "BackendMismatchError": ("ValueError",),
    "BackendPayloadUnsupportedError": ("ValueError", "RuntimeError"),
    "BackendRegistryError": ("ValueError",),
    "BackendRuntimeCompatibilityError": ("ValueError",),
    "BackendUnsupportedError": ("ValueError", "RuntimeError"),
    "BackwardStreamUnavailableError": ("RuntimeError",),
    "BaselineUndeterminedError": ("ValueError",),
    "BatchChunkInputAmbiguityError": ("ValueError",),
    "BatchNormTrainModeWarning": ("Warning",),
    "BundleMemberError": ("ValueError",),
    "BundleNotFinalizedError": ("RuntimeError",),
    "BundleRelationshipError": ("ValueError",),
    "CaptureContextError": ("RuntimeError",),
    "CaptureError": (),
    "ChunkedForwardConfigError": ("ValueError",),
    "CaptureOutcomeError": (),
    "CollectiveBoundaryReplayError": ("RuntimeError",),
    "CompatibilityError": (),
    "ConfigurationError": (),
    "ControlFlowDivergenceError": ("RuntimeError",),
    "ControlFlowDivergenceWarning": ("Warning",),
    "DeadParentError": ("ValueError",),
    "DiagnosticSeverityError": ("ValueError",),
    "DirectActivationWriteWarning": ("Warning",),
    "DirectWriteIgnoredWarning": ("Warning",),
    "DirectWriteInExecutableSaveError": ("ValueError",),
    "DistributedCaptureUnsupportedError": ("RuntimeError",),
    "EngineDispatchError": ("ValueError",),
    "GraphShapeMismatchError": ("ValueError",),
    "GraphvizRenderError": ("RuntimeError",),
    "HelperMountError": ("ValueError",),
    "HookSignatureError": ("TypeError",),
    "HookSiteCoverageError": ("ValueError",),
    "HookValueError": ("ValueError",),
    "InterventionAuditWarning": ("Warning",),
    "InterventionError": (),
    "InterventionReadyConflictError": ("ValueError",),
    "InvalidArgumentError": ("ValueError",),
    "InvalidStorageError": ("ValueError",),
    "KeywordConflictError": ("TypeError",),
    "LiveModeLabelError": ("ValueError",),
    "MetadataInvariantError": ("ValueError",),
    "ModelMismatchError": ("RuntimeError",),
    "MultiMatchWarning": ("Warning",),
    "MultiOutputModuleError": ("ValueError",),
    "MutateInPlaceWarning": ("Warning",),
    "MutatedReferenceError": ("RuntimeError",),
    "NoParentError": ("ValueError",),
    "NonExecutableSpecError": ("RuntimeError",),
    "NumericAttestationError": ("RuntimeError",),
    "OpaqueCallableInExecutableSaveError": ("ValueError",),
    "OutputAttributionError": ("RuntimeError",),
    "PartialCaptureLookupError": ("ValueError",),
    "PathDivergenceError": ("RuntimeError",),
    "PayloadUnavailableError": ("ValueError",),
    "PoisonedRunError": ("RuntimeError",),
    "PostTraceParamUnavailable": ("RuntimeError",),
    "PredicateError": ("RuntimeError",),
    "ReattachError": ("RuntimeError",),
    "RecordBindingError": ("RuntimeError",),
    "RecordContextFieldError": (),
    "RecorderStateError": ("RuntimeError",),
    "RecordingConfigError": ("ValueError",),
    "RecoveryError": ("RuntimeError",),
    "RecursiveTracingError": ("RuntimeError",),
    "ReplayPreconditionError": ("RuntimeError",),
    "RunCapabilityUnavailableError": ("RuntimeError",),
    "RunPreconditionError": ("ValueError",),
    "RunnablePreflightError": ("ValueError",),
    "RunnableTLSPECError": (),
    "RuntimeSignatureDriftError": ("RuntimeError",),
    "SaveBudgetExceededError": ("RuntimeError",),
    "ScalarEscapeWarning": ("Warning",),
    "SelectorCapabilityError": ("ValueError",),
    "SelectorCompositionError": ("ValueError",),
    "ShapeInferenceError": ("RuntimeError",),
    "SiteAmbiguityError": ("ValueError",),
    "SiteResolutionError": ("ValueError",),
    "SpecMutationError": ("ValueError",),
    "SpecPortabilityError": ("ValueError",),
    "SpliceModuleDeviceError": ("RuntimeError",),
    "SpliceModuleDtypeError": ("RuntimeError",),
    "StateBindingError": ("ValueError",),
    "StopSignalSwallowedError": (),
    "TorchLensCaptureGapError": ("RuntimeError",),
    "TorchLensCaptureGapWarning": ("Warning",),
    "TorchLensError": (),
    "TorchLensIOError": ("RuntimeError",),
    "TorchLensInterventionError": ("RuntimeError",),
    "TorchLensInterventionWarning": ("Warning",),
    "TorchLensPostfuncError": ("RuntimeError",),
    "TorchLensWarning": ("Warning",),
    "TraceNotReproducibleWarning": ("Warning",),
    "TrainingModeConfigError": ("ValueError",),
    "UnclassifiedSelectorError": ("ValueError",),
    "UnknownBackendError": ("ValueError",),
    "UnserializableDictKeyError": ("TypeError",),
    "UnsupportedRendererCapabilityError": ("RuntimeError",),
    "UnsupportedTensorVariantError": ("RuntimeError",),
    "UntrustedCallableError": ("RuntimeError",),
    "ValidationError": (),
}


def _public_error_classes() -> dict[str, type[BaseException]]:
    """Return every public ``torchlens.errors`` exception or warning class.

    Returns
    -------
    dict[str, type[BaseException]]
        Mapping from public name to the resolved class object.
    """

    discovered: dict[str, type[BaseException]] = {}
    for name in dir(errors):
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            discovered[name] = obj
    return discovered


def test_builtin_lineage_golden_is_closed() -> None:
    """The lineage golden covers exactly the public error surface.

    A new public exception class cannot ship without a conscious lineage row,
    and a removed class cannot leave a stale row behind.
    """

    discovered = set(_public_error_classes())
    golden = set(BUILTIN_LINEAGE_GOLDEN)

    assert discovered - golden == set(), (
        "public error classes missing a builtin-lineage golden row: "
        f"{sorted(discovered - golden)}"
    )
    assert golden - discovered == set(), (
        f"stale builtin-lineage golden rows: {sorted(golden - discovered)}"
    )


@pytest.mark.parametrize(
    ("class_name", "expected_builtins"),
    sorted(BUILTIN_LINEAGE_GOLDEN.items()),
)
def test_builtin_lineage_matches_golden(
    class_name: str,
    expected_builtins: tuple[str, ...],
) -> None:
    """Every public error class keeps its exact builtin-exception bases.

    This is the regression guard for the r2/r3 reparenting incident class: a
    lineage flip (e.g. a historically-``TypeError`` door becoming
    ``ValueError``-based) shows up here as an exact-tuple mismatch.
    """

    cls = getattr(errors, class_name)
    actual = tuple(
        builtin.__name__
        for builtin in _LINEAGE_PROBE_BUILTINS
        if issubclass(cls, builtin)
    )

    assert actual == expected_builtins, (
        f"{class_name} builtin lineage changed: expected {expected_builtins}, "
        f"got {actual}. If this reparenting is intentional, update the golden, "
        "the error-refusal contract doc, and the glossary in the same change."
    )


def _import_exception(class_module: str, class_name: str) -> Any:
    """Import an exception or warning class from a module path.

    Parameters
    ----------
    class_module:
        Module path containing the class.
    class_name:
        Exception or warning class name.

    Returns
    -------
    Any
        Imported class object.
    """

    module_obj = importlib.import_module(class_module)
    return getattr(module_obj, class_name)


@pytest.mark.parametrize("base_cls", BASE_CLASSES)
def test_base_payload_contract(base_cls: type[errors.TorchLensError]) -> None:
    """Base error classes store the shared structured payload fields."""

    instance = base_cls(
        "problem",
        file_path="model.py",
        line_no=12,
        affected_sites=["relu_1_2"],
        severity="fatal",
        detail="shape mismatch",
    )

    assert isinstance(instance, errors.TorchLensError)
    assert instance.file_path == "model.py"
    assert instance.line_no == 12
    assert instance.affected_sites == ["relu_1_2"]
    assert instance.severity == "fatal"
    assert instance.fields == {"detail": "shape mismatch"}
    assert str(instance) == "problem"


def test_warning_payload_contract() -> None:
    """TorchLensWarning stores the same structured payload fields."""

    instance = errors.TorchLensWarning(
        file_path="model.py",
        line_no=12,
        affected_sites=["relu_1_2"],
        note="non-canonical state",
    )

    assert isinstance(instance, Warning)
    assert instance.file_path == "model.py"
    assert instance.line_no == 12
    assert instance.affected_sites == ["relu_1_2"]
    assert instance.severity == "informational"
    assert "non-canonical state" in str(instance)


def test_invalid_severity_is_rejected() -> None:
    """Severity is runtime-validated against the documented literal values."""

    with pytest.raises(errors.DiagnosticSeverityError) as exc_info:
        errors.TorchLensError(severity="warning")  # type: ignore[arg-type]

    assert exc_info.value.fields["code"] == "diagnostic_severity_invalid"
    assert "Remedy:" in str(exc_info.value)


@pytest.mark.parametrize(
    ("class_module", "class_name", "expected_base", "status"), OLD_EXCEPTION_MAPPING
)
def test_old_exception_class_maps_to_new_base(
    class_module: str,
    class_name: str,
    expected_base: type[BaseException],
    status: str,
) -> None:
    """Every inventoried old exception class is preserved under the new taxonomy."""

    cls = _import_exception(class_module, class_name)

    assert issubclass(cls, expected_base)
    assert getattr(errors, class_name) is cls
    if status == "alias":
        assert class_name == "SpecPortabilityError"
    else:
        assert cls.__name__ == class_name


def test_spec_portability_alias_is_unchanged() -> None:
    """SpecPortabilityError remains an alias for the executable-save error."""

    from torchlens.intervention import errors as intervention_errors

    assert (
        intervention_errors.SpecPortabilityError
        is intervention_errors.OpaqueCallableInExecutableSaveError
    )


@pytest.mark.parametrize(
    ("error_cls", "expected_code"),
    (
        (UnknownBackendError, "unknown_backend"),
        (BackendMismatchError, "backend_mismatch"),
        (BackendAmbiguityError, "backend_ambiguity"),
        (BackendUnsupportedError, "backend_unsupported"),
        (BackendPayloadUnsupportedError, "backend_payload_unsupported"),
        (BackendRuntimeCompatibilityError, "backend_runtime_compatibility"),
    ),
)
def test_backend_refusals_use_shared_fields_and_remedies(
    error_cls: type[errors.ConfigurationError],
    expected_code: str,
) -> None:
    """Backend refusal vocabulary participates in the shared error contract."""

    error = error_cls("backend request failed")

    assert isinstance(error, errors.ConfigurationError)
    assert isinstance(error, ValueError)
    assert getattr(errors, error_cls.__name__) is error_cls
    assert error.fields["code"] == expected_code
    assert error.fields["remedy"]
    assert "Remedy:" in str(error)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is error_cls
    assert str(restored) == str(error)
    assert restored.fields == error.fields


TOP_REFUSAL_CASES: tuple[tuple[str, Callable[[], object]], ...] = (
    (
        "visualization_node_style_invalid",
        lambda: VisualizationOptions(node_style="unknown"),  # type: ignore[arg-type]
    ),
    (
        "visualization_layout_invalid",
        lambda: VisualizationOptions(layout="unknown"),  # type: ignore[arg-type]
    ),
    (
        "visualization_intervention_mode_invalid",
        lambda: VisualizationOptions(intervention_mode="unknown"),  # type: ignore[arg-type]
    ),
    (
        "buffer_visibility_invalid",
        lambda: VisualizationOptions(show_buffers="unknown"),  # type: ignore[arg-type]
    ),
    ("collapse_level_invalid", lambda: VisualizationOptions(collapse=1.5)),
    (
        "collapse_mode_invalid",
        lambda: VisualizationOptions(collapse="unknown"),  # type: ignore[arg-type]
    ),
    (
        "fold_repeats_invalid",
        lambda: VisualizationOptions(fold_repeats="yes"),  # type: ignore[arg-type]
    ),
    (
        "jax_control_flow_invalid",
        lambda: CaptureOptions(jax_control_flow="unknown"),  # type: ignore[arg-type]
    ),
    (
        "jax_unroll_type_invalid",
        lambda: CaptureOptions(jax_max_control_flow_unroll="64"),  # type: ignore[arg-type]
    ),
    ("jax_unroll_range_invalid", lambda: CaptureOptions(jax_max_control_flow_unroll=0)),
    ("distributed_witness_invalid", lambda: CaptureOptions(distributed_witness="unknown")),
    (
        "distributed_payload_witness_unsupported",
        lambda: CaptureOptions(distributed_witness="payload"),
    ),
    (
        "deprecated_argument_conflict",
        lambda: CaptureOptions(mark_layer_depths=True, compute_input_output_distances=True),
    ),
    (
        "deprecated_argument_conflict",
        lambda: CaptureOptions(num_context_lines=1, source_context_lines=2),
    ),
    (
        "deprecated_argument_conflict",
        lambda: CaptureOptions(capture_output_structure=True, capture_container_structure=True),
    ),
    (
        "deprecated_argument_conflict",
        lambda: VisualizationOptions(mode="rolled", view="unrolled"),
    ),
    (
        "deprecated_argument_conflict",
        lambda: VisualizationOptions(max_module_depth=1, depth=2),
    ),
    (
        "deprecated_argument_conflict",
        lambda: VisualizationOptions(layout_engine="dot", layout="rank"),
    ),
    (
        "deprecated_argument_conflict",
        lambda: VisualizationOptions(node_mode="default", node_style="profiling"),
    ),
    (
        "deprecated_argument_conflict",
        lambda: StreamingOptions(save_outs_to="old", bundle_path="new"),
    ),
    (
        "option_group_conflict",
        lambda: merge_capture_options(capture=CaptureOptions(name="one"), name="two"),
    ),
    (
        "option_group_keyword_conflict",
        lambda: merge_visualization_options(
            function_default_mode="none",
            visualization=VisualizationOptions(layout="dot"),
            layout="rank",
        ),
    ),
    ("artifact_save_level_invalid", lambda: coerce_tlspec_save_level("unknown")),
)


@pytest.mark.parametrize(("expected_code", "trigger"), TOP_REFUSAL_CASES)
def test_top_refusal_messages_name_a_remedy(
    expected_code: str,
    trigger: Callable[[], object],
) -> None:
    """Likely public option refusals expose codes and concrete remedies."""

    with pytest.raises(errors.ConfigurationError) as exc_info:
        trigger()

    assert exc_info.value.fields["code"] == expected_code
    remedy = exc_info.value.fields.get("remedy")
    assert isinstance(remedy, str) and remedy
    assert "Remedy:" in str(exc_info.value)
    assert remedy in str(exc_info.value)


@pytest.mark.parametrize(
    ("operation", "trigger"),
    (
        ("record_kpi_in_graph", lambda: user_funcs.record_kpi_in_graph("score", 1.0)),
        (
            "register_tensor_connection",
            lambda: user_funcs.register_tensor_connection(object(), object()),  # type: ignore[arg-type]
        ),
    ),
)
def test_capture_context_refusals_name_the_active_trace_remedy(
    operation: str,
    trigger: Callable[[], object],
) -> None:
    """Capture-only public helpers name both their operation and entry remedy."""

    with pytest.raises(errors.CaptureContextError) as exc_info:
        trigger()

    assert exc_info.value.fields["code"] == "capture_context_required"
    assert exc_info.value.fields["operation"] == operation
    assert "tl.trace()" in str(exc_info.value)
    assert "Remedy:" in str(exc_info.value)


@pytest.mark.parametrize(
    "error_cls",
    (
        errors.InvalidArgumentError,
        errors.ArgumentTypeError,
        errors.ArgumentConflictError,
        errors.CaptureContextError,
    ),
)
def test_actionable_refusal_pickle_round_trip(
    error_cls: type[errors.TorchLensError],
) -> None:
    """Strict actionable constructors preserve fields across exception pickle."""

    original = error_cls(
        "request failed",
        code="pickle_probe",
        remedy="change the request",
        argument="probe",
    )

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is error_cls
    assert str(restored) == str(original)
    assert restored.fields == original.fields
    assert restored.severity == original.severity


def test_save_argument_door_is_typed_and_redirects_save_all() -> None:
    """The ``save=`` type door refuses typed and names the `'all'` remedy.

    ``save='all'`` was documented as a valid spelling while the door raised a
    raw ``TypeError`` with no code or remedy; only ``layers_to_save`` accepts
    ``'all'``. Historical ``TypeError`` lineage is preserved.
    """

    import torch
    from torch import nn

    import torchlens as tl

    with pytest.raises(errors.ArgumentTypeError) as exc_info:
        tl.trace(nn.Identity(), torch.randn(2), save="all")

    assert exc_info.value.fields["code"] == "save_predicate_type_invalid"
    assert isinstance(exc_info.value, TypeError)
    assert "layers_to_save='all'" in exc_info.value.fields["remedy"]
    assert "Remedy:" in str(exc_info.value)


def test_collapse_order_mode_door_has_its_own_code() -> None:
    """The collapse_order diagnostic surface refuses under its own code.

    ``collapse_order(mode=)`` accepts only the two landmark policies, so it no
    longer shares ``collapse_mode_invalid`` with the render surfaces whose
    documented remedy (``'none'``, floats) would refuse again here.
    """

    from torchlens.visualization.auto_collapse import collapse_order

    with pytest.raises(errors.InvalidArgumentError) as exc_info:
        collapse_order(object(), mode="none")  # type: ignore[arg-type]

    assert exc_info.value.fields["code"] == "collapse_order_mode_invalid"
    assert "auto" in exc_info.value.fields["remedy"]


def test_code_panel_doors_split_render_from_config_refusals() -> None:
    """The two code-panel refusals carry distinct codes and builtins.

    A callable that returns a non-string at render time
    (``code_panel_callable_return_invalid``, ``TypeError``) is a different
    caller problem from an unknown mode literal at configuration time
    (``code_panel_option_invalid``, ``ValueError``); one code no longer
    covers both.
    """

    import weakref

    from torch import nn

    from torchlens.visualization.code_panel import resolve_code_panel_source

    model = nn.Identity()
    with pytest.raises(errors.ArgumentTypeError) as return_info:
        resolve_code_panel_source(lambda live_model: 123, {}, weakref.ref(model))
    assert return_info.value.fields["code"] == "code_panel_callable_return_invalid"
    assert isinstance(return_info.value, TypeError)
    assert not isinstance(return_info.value, ValueError)

    with pytest.raises(errors.InvalidArgumentError) as mode_info:
        resolve_code_panel_source("sideways", {}, None)  # type: ignore[arg-type]
    assert mode_info.value.fields["code"] == "code_panel_option_invalid"
    assert isinstance(mode_info.value, ValueError)
    assert not isinstance(mode_info.value, TypeError)


def test_predicate_type_doors_are_multiclass_by_surface() -> None:
    """The predicate-type codes carry a per-surface builtin, per site history.

    ``intervention_predicate_type_invalid`` / ``halt_predicate_type_invalid``
    are ``ArgumentTypeError`` (historically raw ``TypeError``) on the
    ``tl.trace`` surface but ``InvalidArgumentError`` (historically raw
    ``ValueError``) on the ``tl.record`` surface. The contract doc documents
    the multiclass explicitly; this pin makes any silent unification loud.
    """

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens.fastlog.options import RecordingOptions

    for kwarg in ("intervene", "halt"):
        with pytest.raises(errors.ArgumentTypeError) as trace_info:
            tl.trace(nn.Identity(), torch.randn(2), **{kwarg: 123})
        assert trace_info.value.fields["code"] == f"{'intervention' if kwarg == 'intervene' else 'halt'}_predicate_type_invalid"
        assert isinstance(trace_info.value, TypeError)
        assert not isinstance(trace_info.value, ValueError)

        with pytest.raises(errors.InvalidArgumentError) as record_info:
            RecordingOptions(**{kwarg: 123})
        assert record_info.value.fields["code"] == f"{'intervention' if kwarg == 'intervene' else 'halt'}_predicate_type_invalid"
        assert isinstance(record_info.value, ValueError)
        assert not isinstance(record_info.value, TypeError)


def test_option_group_conflict_doors_split_by_site_history() -> None:
    """Each grouped/flat conflict door keeps its historical builtin, per code.

    The five merge entrypoints historically raised ``raise
    ValueError(conflict_message)`` and now raise ``ArgumentConflictError``
    under ``option_group_conflict``; the visualization merge historically
    raised a raw ``TypeError`` and now raises ``KeywordConflictError`` under
    ``option_group_keyword_conflict``. One code maps to one catchable builtin.
    """

    with pytest.raises(errors.ArgumentConflictError) as value_info:
        merge_capture_options(capture=CaptureOptions(name="one"), name="two")
    assert value_info.value.fields["code"] == "option_group_conflict"
    assert isinstance(value_info.value, ValueError)
    assert not isinstance(value_info.value, TypeError)

    with pytest.raises(errors.KeywordConflictError) as keyword_info:
        merge_visualization_options(
            function_default_mode="none",
            visualization=VisualizationOptions(layout="dot"),
            layout="rank",
        )
    assert keyword_info.value.fields["code"] == "option_group_keyword_conflict"
    assert isinstance(keyword_info.value, TypeError)
    assert not isinstance(keyword_info.value, ValueError)


def test_intervention_direction_doors_split_by_site_history() -> None:
    """Predicate-side direction refusal keeps its historical TypeError lineage.

    The door raised a raw ``TypeError`` since the 2.16 intervention era, and
    the live capture-path callers catch ``TypeError`` to convert a bad
    predicate result into ``PredicateError``. Its code is distinct from the
    ValueError-lineage trace-side ``intervention_direction_invalid`` doors so
    ``fields["code"]`` determines the catchable builtin.
    """

    from torchlens.intervention.predicates import as_intervention_decision

    with pytest.raises(errors.ArgumentTypeError) as exc_info:
        as_intervention_decision(lambda out: out, direction="sideways")  # type: ignore[arg-type]

    assert exc_info.value.fields["code"] == "intervention_action_direction_invalid"
    assert isinstance(exc_info.value, TypeError)
    assert not isinstance(exc_info.value, ValueError)
    assert "Remedy:" in str(exc_info.value)


def test_selector_direction_refusal_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward intervention normalization propagates typed selector refusals."""

    def _raise_typed_refusal(selector: object) -> str:
        """Raise the typed selector refusal used to probe the former broad catch."""

        del selector
        raise SiteResolutionError("selector taxonomy refusal", code="selector_taxonomy_refusal")

    def predicate(context: object) -> None:
        """Return no intervention decision for the synthetic predicate."""

        del context

    predicate.selector = object()  # type: ignore[attr-defined]
    predicate.decision = InterventionDecision(action="transform")  # type: ignore[attr-defined]
    monkeypatch.setattr(user_funcs, "_selector_resolution_direction", _raise_typed_refusal)

    with pytest.raises(SiteResolutionError) as exc_info:
        user_funcs._backward_intervention_spec_from_predicate(predicate)

    assert exc_info.value.fields["code"] == "selector_taxonomy_refusal"
