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
        "option_group_conflict",
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
