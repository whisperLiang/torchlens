"""Non-vacuous replay validation for the TensorFlow eager backend."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from ..._validation_status import ValidationReplaySource, ValidationStatus
from .._constants import CONSTANT_PARENT_PAYLOAD_REASON, CONSTANT_PRODUCER_LEAF_REASON
from .._invariants import run_metadata_invariants
from .._validation_shared import float_replay_tolerances, ops_by_label as _ops_by_label
from ._native_op_specs import _NATIVE_OP_SPECS
from ._raw_op_replay import (
    _import_tensorflow,
    _native_output_slots,
    _pause_tf_replay,
    _prepare_native_call,
    _PreparedNativeCall,
)
from .op_callback_capture import TFInputCapture, TFOpCapture

TFOpClass = Literal[
    "value",
    "source",
    "pure-resource-read",
    "structural-value",
    "annotation",
    "effect-denied",
    "control-region",
]

#: Literal producers (SPEC section 2, rule 2): ops whose every input is a host literal or
#: shape, so the op has no graph parents and its output is a constant of the forward.
#: ``Fill`` is also a variable-initializer op type; the classification keeps the
#: initializer-contamination failure for a ``Fill`` nothing in the forward consumed and
#: admits only a ``Fill`` whose output feeds a materialized op (``_is_literal_producer``).
_LITERAL_PRODUCER_OPS = frozenset({"Fill"})

#: Validation decision reasons this backend emits. All are members of the
#: closed ``VALIDATION_REASONS`` list in ``backends/_backend_protocol.py``
#: (``replay_match`` is the validated-decision spelling, not a failure reason).
_REASON_REPLAYED = "replay_match"
_REASON_REPLAY_MISMATCH = "replay_mismatch"
_REASON_PERTURBATION = "perturbation_insensitive"
_REASON_NO_FUNCTION = "no_replayable_function"
_REASON_EFFECT = "effectful_region"
_REASON_UNCLASSIFIED = "op_unclassified"
_REASON_INCONSISTENT = "capture_inconsistent"
_REASON_CONSTANT_LEAF = CONSTANT_PRODUCER_LEAF_REASON
_REASON_CONSTANT_PERTURB = CONSTANT_PARENT_PAYLOAD_REASON
_UNREGISTERED_CLASSES: Mapping[str, TFOpClass] = {
    "Const": "source",
    "Placeholder": "source",
    "constant": "source",
    "Size": "structural-value",
    "Rank": "structural-value",
    "Gelu": "value",
    "If": "control-region",
    "StatelessIf": "control-region",
    "While": "control-region",
    "StatelessWhile": "control-region",
    "Case": "control-region",
    "Switch": "control-region",
    "Merge": "control-region",
    "StatefulPartitionedCall": "control-region",
    "Assert": "effect-denied",
    "AssignAddVariableOp": "effect-denied",
    "AssignSubVariableOp": "effect-denied",
    "DestroyResourceOp": "effect-denied",
    "NoOp": "effect-denied",
    "PrintV2": "effect-denied",
    "RandomStandardNormal": "effect-denied",
}


@dataclass(frozen=True)
class TFValidationCounts:
    """TensorFlow validation outcome counts.

    Parameters
    ----------
    replayed_node_count
        Number of pure nodes replayed and perturbation-checked.
    pure_unverified_node_count
        Number of cleanly classified pure value nodes outside the replay
        allowlist or missing replayable payloads.
    effect_region_node_count
        Number of stateful, resource-read, control, structural, or annotation
        nodes excluded from pure per-op replay.
    failed_node_count
        Number of validation failures.
    """

    replayed_node_count: int = 0
    pure_unverified_node_count: int = 0
    effect_region_node_count: int = 0
    failed_node_count: int = 0


@dataclass(frozen=True)
class TFValidationResult:
    """Detailed TensorFlow validation result.

    Parameters
    ----------
    counts
        Replay and failure counts.
    failures
        Stable failure reason strings.
    classes
        Per-op class assignment keyed by raw label.
    replayed_histogram
        Replayed op-type histogram.
    pure_unverified_histogram
        Pure but non-replayed op-type histogram.
    effect_region_histogram
        Effect or non-value op-type histogram.
    """

    counts: TFValidationCounts
    failures: tuple[str, ...]
    classes: dict[str, TFOpClass]
    replayed_histogram: Counter[str]
    pure_unverified_histogram: Counter[str]
    effect_region_histogram: Counter[str]
    decisions: tuple[Mapping[str, Any], ...] = ()
    unverified_reason_counts: Mapping[str, int] = field(default_factory=dict)
    exempted_reason_counts: Mapping[str, int] = field(default_factory=dict)
    first_failure: tuple[str, str, str] | None = None


def validate_tf_trace(trace: Any, *, validate_metadata: bool = True) -> ValidationStatus:
    """Validate a TensorFlow trace with the P4 non-vacuous tripwire.

    Parameters
    ----------
    trace
        TensorFlow trace produced by the eager op-callback backend.
    validate_metadata
        Whether backend-neutral metadata invariants should also run.

    Returns
    -------
    ValidationStatus
        Failed, passed, or honestly unverified replay-validation status.
    """

    del validate_metadata
    result = validate_tf_trace_detailed(trace)
    source: ValidationReplaySource = (
        "loaded" if getattr(trace, "_loaded_from_bundle", False) else "live"
    )
    status = ValidationStatus.from_replay_counts(
        backend="tf",
        source=source,
        num_replayed_ops=result.counts.replayed_node_count,
        num_unverified_ops=0,
        num_pure_unverified_ops=result.counts.pure_unverified_node_count,
        num_effect_region_ops=result.counts.effect_region_node_count,
        num_failed_ops=result.counts.failed_node_count,
        payload_load_status=getattr(trace, "payload_load_status", None),
        unverified_reason_counts=result.unverified_reason_counts,
        exempted_reason_counts=result.exempted_reason_counts,
        decisions=result.decisions,
    )
    if status.state == "failed" and result.first_failure is not None:
        label, op_type, reason = result.first_failure
        status = replace(
            status,
            message=(
                f"Replay validation failed at {label} ({op_type}): {reason}; "
                f"{result.counts.failed_node_count} failing op(s), "
                f"{result.counts.replayed_node_count} replayed."
            ),
        )
    trace._tf_validation_result = result
    trace._validation_replay_status = status
    return status


def validate_tf_trace_detailed(trace: Any) -> TFValidationResult:
    """Run TensorFlow validation and return detailed counts and diagnostics.

    Parameters
    ----------
    trace
        TensorFlow trace produced by the eager op-callback backend.

    Returns
    -------
    TFValidationResult
        Detailed validation outcome.
    """

    replayed_histogram: Counter[str] = Counter()
    pure_unverified_histogram: Counter[str] = Counter()
    effect_region_histogram: Counter[str] = Counter()

    ops_by_label = _ops_by_label(trace)
    captures_by_label = _captures_by_label(trace)
    failures, classes = _structural_failures(trace, ops_by_label, captures_by_label)
    if failures:
        return _structural_failure_result(failures, classes, ops_by_label)

    decisions_list: list[Mapping[str, Any]] = []
    unverified_reasons: Counter[str] = Counter()
    exempted_reasons: Counter[str] = Counter()
    multi_output_calls = _multi_output_captures(captures_by_label)
    replayed_slots: dict[str, str] = {}
    for label, op in _raw_label_items(ops_by_label):
        op_class = classes[label]
        op_type = str(getattr(op, "func_name", ""))
        op_label = str(getattr(op, "label", label))
        if op_class == "source":
            _record_source_decisions(op_label, op_type, decisions_list, exempted_reasons)
            continue
        if op_class != "value":
            if op_class in {"pure-resource-read", "effect-denied", "control-region"}:
                effect_region_histogram[op_type] += 1
                unverified_reasons[_REASON_EFFECT] += 1
                decisions_list.append(
                    _decision(op_label, op_type, "replay", "unverified", _REASON_EFFECT)
                )
            continue
        outcome = _value_outcome(
            label, op_type, ops_by_label, captures_by_label, multi_output_calls, replayed_slots
        )
        if outcome == _REASON_NO_FUNCTION:
            pure_unverified_histogram[op_type] += 1
            unverified_reasons[_REASON_NO_FUNCTION] += 1
            decisions_list.append(
                _decision(op_label, op_type, "replay", "unverified", _REASON_NO_FUNCTION)
            )
            continue
        if outcome == _REASON_REPLAYED:
            replayed_histogram[op_type] += 1
            decisions_list.append(_decision(op_label, op_type, "replay", "validated", outcome))
            continue
        failures.append(f"replay_or_perturbation_failed:{label}:{op_type}:{outcome}")
        phase = "perturbation" if outcome == _REASON_PERTURBATION else "replay"
        decisions_list.append(_decision(op_label, op_type, phase, "failed", outcome))

    return _detailed_result(
        counts=TFValidationCounts(
            replayed_node_count=sum(replayed_histogram.values()),
            pure_unverified_node_count=sum(pure_unverified_histogram.values()),
            effect_region_node_count=sum(effect_region_histogram.values()),
            failed_node_count=len(failures),
        ),
        failures=failures,
        classes=classes,
        histograms=(replayed_histogram, pure_unverified_histogram, effect_region_histogram),
        decision_tallies=(decisions_list, unverified_reasons, exempted_reasons),
    )


def _value_outcome(
    label: str,
    op_type: str,
    ops_by_label: Mapping[str, Any],
    captures_by_label: Mapping[str, TFOpCapture],
    multi_output_calls: Mapping[int, list[TFOpCapture]],
    replayed_slots: dict[str, str],
) -> str:
    """Use the native policy and share the result among sibling output records."""

    spec = _NATIVE_OP_SPECS.get(op_type)
    capture = captures_by_label.get(label)
    if spec is None or spec.validation != "perturb" or capture is None:
        return _REASON_NO_FUNCTION
    if capture._output_count <= 1:
        return _replay_and_perturb_call((capture,), ops_by_label)[label]
    if label not in replayed_slots:
        siblings = multi_output_calls[capture._call_id]
        replayed_slots.update(_replay_and_perturb_call(siblings, ops_by_label))
    return replayed_slots[label]


def _raw_label_items(ops_by_label: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Visit each operation once in the sidecar's raw label space."""

    return (
        (label, op)
        for label, op in ops_by_label.items()
        if label == getattr(op, "_label_raw", None)
    )


def _record_source_decisions(
    op_label: str,
    op_type: str,
    decisions_list: list[Mapping[str, Any]],
    exempted_reasons: Counter[str],
) -> None:
    """Record a closure constant as an exempted leaf (constants rule item 3)."""

    if op_type != "constant":
        return
    # Constants rule item 3: a closure constant is a leaf whose
    # saved payload is the record; nothing upstream to perturb.
    decisions_list.extend(_constant_leaf_decisions(op_label))
    exempted_reasons[_REASON_CONSTANT_LEAF] += 1
    exempted_reasons[_REASON_CONSTANT_PERTURB] += 1


def _detailed_result(
    *,
    counts: TFValidationCounts,
    failures: list[str],
    classes: dict[str, TFOpClass],
    histograms: tuple[Counter[str], Counter[str], Counter[str]],
    decision_tallies: tuple[list[Mapping[str, Any]], Counter[str], Counter[str]],
) -> TFValidationResult:
    """Assemble the detailed result from the loop's tallies (decisions, unverified, exempted)."""

    replayed_histogram, pure_unverified_histogram, effect_region_histogram = histograms
    decisions_list, unverified_reasons, exempted_reasons = decision_tallies
    decisions = tuple(decisions_list)
    return TFValidationResult(
        counts=counts,
        failures=tuple(failures),
        classes=classes,
        replayed_histogram=replayed_histogram,
        pure_unverified_histogram=pure_unverified_histogram,
        effect_region_histogram=effect_region_histogram,
        decisions=decisions,
        unverified_reason_counts=dict(unverified_reasons),
        exempted_reason_counts=dict(exempted_reasons),
        first_failure=_first_failure(decisions),
    )


def _structural_failures(
    trace: Any, ops_by_label: Mapping[str, Any], captures_by_label: Mapping[str, Any]
) -> tuple[list[str], dict[str, TFOpClass]]:
    """Classify every op and check the capture's self-consistency.

    Returns
    -------
    tuple[list[str], dict[str, TFOpClass]]
        The structural failure strings (empty when the capture is sound) and
        the op class by raw label.
    """

    failures: list[str] = []
    classes: dict[str, TFOpClass] = {}
    init_labels = set(getattr(trace, "_tf_init_op_labels", ()))
    _validate_classification(
        ops_by_label=ops_by_label,
        init_labels=init_labels,
        classes=classes,
        failures=failures,
    )
    _validate_self_consistency(
        trace=trace,
        ops_by_label=ops_by_label,
        captures_by_label=captures_by_label,
        failures=failures,
    )
    return failures, classes


def _structural_failure_result(
    failures: list[str], classes: dict[str, TFOpClass], ops_by_label: Mapping[str, Any]
) -> TFValidationResult:
    """The result of a capture whose structure failed before any replay ran."""

    decisions = tuple(_structural_failure_decision(item, ops_by_label) for item in failures)
    return TFValidationResult(
        counts=TFValidationCounts(failed_node_count=len(failures)),
        failures=tuple(failures),
        classes=classes,
        replayed_histogram=Counter(),
        pure_unverified_histogram=Counter(),
        effect_region_histogram=Counter(),
        decisions=decisions,
        first_failure=_first_failure(decisions),
    )


def _decision(
    op_label: str, func_name: str, phase: str, decision: str, reason: str
) -> Mapping[str, Any]:
    """One per-op validation decision in the torch record shape."""

    return {
        "op_label": op_label,
        "func_name": func_name,
        "phase": phase,
        "decision": decision,
        "reason": reason,
    }


def _constant_leaf_decisions(op_label: str) -> tuple[Mapping[str, Any], ...]:
    """The two named leaf decisions of a ``constant`` source op (constants rule item 3)."""

    replay = dict(_decision(op_label, "constant", "replay", "exempted", _REASON_CONSTANT_LEAF))
    replay["justification"] = "closure constant: its saved payload is the record"
    perturb = dict(
        _decision(op_label, "constant", "perturbation", "exempted", _REASON_CONSTANT_PERTURB)
    )
    perturb["justification"] = "closure constant: no upstream op to perturb"
    return (replay, perturb)


def _structural_failure_decision(item: str, ops_by_label: Mapping[str, Any]) -> Mapping[str, Any]:
    """Decision for one classification or self-consistency failure string.

    Failure strings are ``<kind>:<raw label>:<detail...>``; the op label is
    resolved through the op index so the decision names the public label.
    """

    kind, _, rest = item.partition(":")
    raw_label, _, detail = rest.partition(":")
    op = ops_by_label.get(raw_label)
    op_label = str(getattr(op, "label", raw_label)) if op is not None else raw_label
    func_name = str(getattr(op, "func_name", "")) if op is not None else detail
    structural = {"unclassified_op", "initializer_contamination"}
    reason = _REASON_UNCLASSIFIED if kind in structural else _REASON_INCONSISTENT
    record = dict(_decision(op_label, func_name, "metadata", "failed", reason))
    record["justification"] = item
    return record


def _first_failure(decisions: tuple[Mapping[str, Any], ...]) -> tuple[str, str, str] | None:
    """Label, func name and reason of the first failed decision, if any."""

    for record in decisions:
        if record.get("decision") == "failed":
            return (
                str(record.get("op_label")),
                str(record.get("func_name")),
                str(record.get("reason")),
            )
    return None


def validate_tf_trace_status(trace: Any, *, validate_metadata: bool = True) -> ValidationStatus:
    """The protocol's ``validate``: always a ``ValidationStatus``, never a bool.

    Parameters
    ----------
    trace
        TensorFlow trace produced by the eager op-callback backend.
    validate_metadata
        Whether backend-neutral metadata invariants run first.

    Returns
    -------
    ValidationStatus
        ``unavailable`` for a loaded trace whose runtime captures were
        stripped; otherwise the replay result with per-op ``decisions`` and,
        on failure, a message naming the first offending op.
    """

    status = trace.validation_replay_status
    if not status.is_available:
        trace._validation_replay_status = status
        return status
    # Direct read of the load-provenance marker: present only on a loaded
    # trace, so its absence IS the typed "live" reading (the live-reading seam).
    try:
        loaded_from_bundle = bool(trace._loaded_from_bundle)
    except AttributeError:
        loaded_from_bundle = False
    if loaded_from_bundle and not getattr(trace, "_tf_op_captures", ()):
        # Portable save strips the runtime callback captures; a loaded trace
        # has nothing to replay, which is "unavailable", never "failed".
        stripped = ValidationStatus.unavailable_loaded_runtime_stripped(
            backend="tf", payload_load_status=getattr(trace, "payload_load_status", None)
        )
        trace._validation_replay_status = stripped
        return stripped
    source: ValidationReplaySource = "loaded" if loaded_from_bundle else "live"
    try:
        if validate_metadata:
            run_metadata_invariants(trace)
        return validate_tf_trace(trace, validate_metadata=validate_metadata)
    except Exception as exc:  # noqa: BLE001 - the tripwire reports, never raises
        failed = ValidationStatus.result(
            passed=False,
            backend="tf",
            source=source,
            payload_load_status=getattr(trace, "payload_load_status", None),
            num_failed_ops=1,
            decisions=(
                _decision("<trace>", "", "metadata", "failed", "metadata_invariant_exception"),
            ),
        )
        failed = replace(
            failed,
            message=f"Replay validation failed before replay: {type(exc).__name__}: {exc}",
        )
        trace._validation_replay_status = failed
        return failed


def replay_allowlist() -> tuple[str, ...]:
    """Return the TensorFlow pure-op raw replay allowlist.

    Returns
    -------
    tuple[str, ...]
        Sorted replayable TensorFlow op types.
    """

    return tuple(
        sorted(name for name, spec in _NATIVE_OP_SPECS.items() if spec.validation == "perturb")
    )


def _validate_classification(
    *,
    ops_by_label: Mapping[str, Any],
    init_labels: set[str],
    classes: dict[str, TFOpClass],
    failures: list[str],
) -> None:
    """Classify every materialized op exactly once and fail closed.

    Parameters
    ----------
    ops_by_label
        Materialized ops keyed by raw and public labels.
    init_labels
        Raw labels marked as initializer contamination during capture.
    classes
        Mutable class mapping to populate.
    failures
        Mutable failure list.

    Returns
    -------
    None
        Mutates ``classes`` and ``failures``.
    """

    for label, op in _raw_label_items(ops_by_label):
        op_type = str(getattr(op, "func_name", ""))
        if label in init_labels and not _is_literal_producer(op):
            failures.append(f"initializer_contamination:{label}:{op_type}")
            continue
        op_class = _classify_op(op)
        if op_class is None:
            failures.append(f"unclassified_op:{label}:{op_type}")
            continue
        classes[label] = op_class
    _propagate_control_region_classes(ops_by_label=ops_by_label, classes=classes)


def _propagate_control_region_classes(
    *,
    ops_by_label: Mapping[str, Any],
    classes: dict[str, TFOpClass],
) -> None:
    """Mark ops downstream of control regions as unverified regions.

    Parameters
    ----------
    ops_by_label
        Materialized ops keyed by raw and public labels.
    classes
        Mutable class mapping to update.

    Returns
    -------
    None
        Mutates ``classes`` in place.
    """

    def _raw_parent_labels(op: Any) -> tuple[str, ...]:
        """Return the op's parents resolved to raw label space.

        Classes are keyed by RAW labels while grouped edges hold final
        pass-qualified labels, so parents resolve through the op index back
        to their raw capture identity before region membership tests.
        """

        resolved: list[str] = []
        for parent in getattr(op, "parents", ()):
            parent_text = str(parent)
            parent_op = ops_by_label.get(parent_text)
            parent_raw = getattr(parent_op, "_label_raw", None) if parent_op is not None else None
            resolved.append(parent_raw if isinstance(parent_raw, str) else parent_text)
        return tuple(resolved)

    changed = True
    while changed:
        changed = False
        region_labels = {
            label for label, op_class in classes.items() if op_class == "control-region"
        }
        for label, op in ops_by_label.items():
            if label != getattr(op, "_label_raw", None):
                continue
            if classes.get(label) in {"source", "control-region"}:
                continue
            if any(parent in region_labels for parent in _raw_parent_labels(op)):
                classes[label] = "control-region"
                changed = True


def _is_literal_producer(op: Any) -> bool:
    """Whether ``op`` is a forward literal producer rather than initializer contamination.

    A literal producer (SPEC section 2, rule 2) has no graph parents and its
    output is consumed by at least one materialized op of the forward. An
    initializer ``Fill`` feeds only ``AssignVariableOp`` (never materialized),
    so it has no children and stays a contamination failure.
    """

    op_type = str(getattr(op, "func_name", ""))
    if op_type not in _LITERAL_PRODUCER_OPS:
        return False
    if tuple(getattr(op, "parents", ()) or ()):
        return False
    return bool(tuple(getattr(op, "children", ()) or ()))


def _classify_op(op: Any) -> TFOpClass | None:
    """Classify one TensorFlow op for validation accounting.

    Parameters
    ----------
    op
        Materialized TorchLens op.

    Returns
    -------
    TFOpClass | None
        Exactly one validation class, or ``None`` for fail-closed unknown ops.
    """

    op_type = str(getattr(op, "func_name", ""))
    if bool(getattr(op, "is_input", False)) or op_type == "input":
        return "source"
    if op_type.startswith(("__inference_", "region:")):
        return "control-region"
    spec = _NATIVE_OP_SPECS.get(op_type)
    if spec is None:
        return _UNREGISTERED_CLASSES.get(op_type)
    return "value" if spec.validation == "perturb" else spec.validation


def _validate_self_consistency(
    *,
    trace: Any,
    ops_by_label: Mapping[str, Any],
    captures_by_label: Mapping[str, TFOpCapture],
    failures: list[str],
) -> None:
    """Validate callback output and consumed-input producer consistency.

    Parameters
    ----------
    trace
        TensorFlow trace with side-channel records.
    ops_by_label
        Materialized ops keyed by raw and public labels.
    captures_by_label
        Callback captures keyed by raw label.
    failures
        Mutable failure list.

    Returns
    -------
    None
        Mutates ``failures`` on coverage gaps.
    """

    unresolved = tuple(getattr(trace, "_tf_unresolved_producers", ()))
    if unresolved:
        failures.extend(
            f"unresolved_producer:{item.consumer_op_type}:{item.input_index}" for item in unresolved
        )
    produced_labels: set[str] = set()
    source_labels = _source_labels(ops_by_label)
    for capture in captures_by_label.values():
        if capture.label_raw not in ops_by_label:
            failures.append(f"captured_output_missing_materialized_op:{capture.label_raw}")
        produced_labels.add(capture.label_raw)
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        if bool(getattr(op, "is_input", False)) or label in source_labels:
            continue
        if label not in captures_by_label:
            failures.append(f"materialized_op_missing_capture:{label}")
    for capture in captures_by_label.values():
        expected_graph_parents: set[str] = set()
        for input_record in capture.inputs:
            graph_parent = input_record.producer_label_raw or input_record.source_label_raw
            if graph_parent is not None:
                expected_graph_parents.add(graph_parent)
        for input_record in capture.inputs:
            producer_label = input_record.producer_label_raw
            source_label = input_record.source_label_raw
            if producer_label is not None and producer_label not in produced_labels:
                failures.append(
                    "input_producer_missing_prior_callback_output:"
                    f"{capture.label_raw}:{producer_label}:{input_record.input_index}"
                )
            if producer_label is None and source_label is None and input_record.source_kind is None:
                failures.append(
                    f"input_missing_producer_or_source:{capture.label_raw}:"
                    f"{input_record.input_index}"
                )
            if source_label is not None and source_label not in source_labels:
                failures.append(
                    f"input_source_label_missing:{capture.label_raw}:{source_label}:"
                    f"{input_record.input_index}"
                )
        op = ops_by_label.get(capture.label_raw)
        if op is None:
            continue
        # Capture records speak RAW label space. Recurrence grouping rewrites graph edges to final
        # pass-qualified labels, so parents are resolved back to raw space before the conservation
        # comparison; an unresolvable parent keeps its literal label and fails closed.
        graph_parents: set[str] = set()
        for parent in getattr(op, "parents", ()):
            parent_text = str(parent)
            parent_op = ops_by_label.get(parent_text)
            parent_raw = getattr(parent_op, "_label_raw", None) if parent_op is not None else None
            graph_parents.add(parent_raw if isinstance(parent_raw, str) else parent_text)
        if expected_graph_parents != graph_parents:
            failures.append(
                f"graph_parent_edges_not_conserved:{capture.label_raw}:"
                f"{sorted(expected_graph_parents)!r}:{sorted(graph_parents)!r}"
            )


def _source_labels(ops_by_label: Mapping[str, Any]) -> set[str]:
    """Raw labels of the trace's source ops: inputs and constants."""

    source_labels: set[str] = set()
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        if bool(getattr(op, "is_input", False)) or getattr(op, "func_name", None) in {
            "input",
            "constant",
        }:
            source_labels.add(label)
    return source_labels


def _multi_output_captures(
    captures_by_label: Mapping[str, TFOpCapture],
) -> dict[int, list[TFOpCapture]]:
    """Index sibling output captures by their shared native invocation."""

    groups: dict[int, list[TFOpCapture]] = {}
    for capture in captures_by_label.values():
        if capture._output_count > 1:
            groups.setdefault(capture._call_id, []).append(capture)
    return groups


def _replay_and_perturb_call(
    captures: Sequence[TFOpCapture], ops_by_label: Mapping[str, Any]
) -> dict[str, str]:
    """Validate every output slot using one native call per input set."""

    try:
        if len(captures) > 1:
            captures = tuple(captures[index] for index in _native_output_slots(captures))
        capture = captures[0]
        tf = _import_tensorflow()
        native, records = _prepare_native_call(tf, capture)
        if native.output_count != len(captures):
            raise ValueError(f"{capture.op_type} has incomplete TensorFlow call outputs")
        captured_tensor = capture.output_tensor
        device = captured_tensor.device if isinstance(captured_tensor, tf.Tensor) else None
        with tf.device(device) if device else nullcontext(), _pause_tf_replay():
            inputs, parent_slots = _rebuild_inputs(tf, records, ops_by_label)
            outputs = native._outputs(native._execute(inputs))
            return _compare_slots(tf, native, inputs, parent_slots, captures, outputs, ops_by_label)
    except Exception:
        return dict.fromkeys((capture.label_raw for capture in captures), _REASON_REPLAY_MISMATCH)


def _compare_slots(
    tf: Any,
    native: _PreparedNativeCall,
    inputs: list[Any],
    parent_slots: Mapping[str, list[int]],
    captures: Sequence[TFOpCapture],
    outputs: Sequence[Any],
    ops_by_label: Mapping[str, Any],
) -> dict[str, str]:
    """Compare each saved slot and require each match to respond to perturbation."""

    outcomes: dict[str, str] = {}
    matched: dict[int, Any] = {}
    for capture, replayed in zip(captures, outputs, strict=True):
        op = ops_by_label[capture.label_raw]
        try:
            saved = _saved_payload(op)
            matches = _slot_matches(tf, capture, op, replayed, saved)
        except Exception:
            matches = False
        if matches:
            matched[capture.output_index] = saved
        outcomes[capture.label_raw] = _REASON_REPLAYED if matches else _REASON_REPLAY_MISMATCH

    insensitive = _insensitive_slots(tf, native, inputs, parent_slots, matched)
    for index in insensitive:
        outcomes[captures[index].label_raw] = _REASON_PERTURBATION
    return outcomes


def _slot_matches(tf: Any, capture: TFOpCapture, op: Any, replayed: Any, saved: Any) -> bool:
    """Compare the slot's capture provenance, native placement and saved payload."""

    if (capture.op_type, capture.output_index) != (op.func_name, op.multi_output_index or 0):
        return False
    original = capture.output_tensor
    if isinstance(original, tf.Tensor) and original.device != replayed.device:
        return False
    return _payloads_close(replayed, saved)


def _rebuild_inputs(
    tf: Any, records: tuple[TFInputCapture, ...], ops_by_label: Mapping[str, Any]
) -> tuple[list[Any], dict[str, list[int]]]:
    """Bind saved inputs and index all slots substituted together for each parent."""

    inputs: list[Any] = []
    parent_slots: dict[str, list[int]] = {}
    for record in records:
        label = record.producer_label_raw or record.source_label_raw
        if label is not None:
            parent_slots.setdefault(label, []).append(record.input_index)
            payload = _saved_payload(ops_by_label[label])
        elif record.source_kind is not None:
            payload = _tensor_to_numpy(record.tensor)
        else:
            raise LookupError(f"unresolved input {record.input_index}")
        inputs.append(tf.convert_to_tensor(payload))
    return inputs, parent_slots


def _insensitive_slots(
    tf: Any,
    native: _PreparedNativeCall,
    inputs: list[Any],
    parent_slots: Mapping[str, list[int]],
    saved_outputs: Mapping[int, Any],
) -> set[int]:
    """Find the slots unchanged by every applicable parent substitution."""

    pending = set(saved_outputs)
    attempted = False
    for slots in parent_slots.values():
        if not pending:
            break
        for replacement in _perturb_candidates(tf, inputs[slots[0]]):
            attempted = True
            modified = inputs.copy()
            for index in slots:
                modified[index] = replacement
            try:
                outputs = native._outputs(native._execute(modified))
            except Exception:
                continue
            pending = {
                index for index in pending if _payloads_close(outputs[index], saved_outputs[index])
            }
            if not pending:
                return pending
    return pending if attempted else set()


def _perturb_candidates(tf: Any, tensor: Any) -> tuple[Any, ...]:
    """Return deterministic TensorFlow perturbation candidates.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    tensor
        TensorFlow tensor to perturb.

    Returns
    -------
    tuple[Any, ...]
        Candidate tensors with matching dtype.
    """

    array = _tensor_to_numpy(tensor)
    dtype = getattr(tensor, "dtype", None)
    candidates: tuple[Any, ...]
    if np.issubdtype(array.dtype, np.bool_):
        candidates = (np.logical_not(array),)
    elif np.issubdtype(array.dtype, np.integer):
        candidates = (array + 1, np.zeros_like(array), np.invert(array))  # ~x flips the sign
    elif np.issubdtype(array.dtype, np.floating):
        magnitude = np.max(np.abs(array)).item() + 1.0 if array.size else 1.0
        ramp = np.arange(array.size, dtype=np.float64).reshape(array.shape)
        ramp = (ramp + 1.0) * (magnitude / max(array.size, 1))
        candidates = (array + magnitude, array - magnitude, np.zeros_like(array))
        candidates = (*candidates, array + ramp.astype(array.dtype, copy=False))
    else:
        return ()
    return tuple(tf.convert_to_tensor(candidate, dtype=dtype) for candidate in candidates)


def _payloads_close(left: Any, right: Any) -> bool:
    """Return whether two TensorFlow payloads match within backend tolerance.

    Parameters
    ----------
    left
        Left payload or TensorFlow tensor.
    right
        Right payload or TensorFlow tensor.

    Returns
    -------
    bool
        True when shape, dtype, and values match.
    """

    left_array = _tensor_to_numpy(left)
    right_array = _tensor_to_numpy(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if np.issubdtype(left_array.dtype, np.bool_) or np.issubdtype(left_array.dtype, np.integer):
        return bool(np.array_equal(left_array, right_array))
    if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
        left_array.dtype,
        np.complexfloating,
    ):
        # equal_nan matches the jax/mlx/paddle sibling oracles and torch's tensor_nanequal doctrine:
        # an identical NaN pattern is agreement, NaN-vs-number still fails elementwise. Omitting it
        # false-FAILED every legitimately NaN-bearing TF payload. Complex payloads take the same
        # component-eps band as mlx/jax (R17 consistency sweep: bit-exact complex here false-failed
        # legitimate replay jitter); ``np.finfo`` reports component precision for complex dtypes.
        rtol, atol = float_replay_tolerances(np.finfo(left_array.dtype))
        return bool(np.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert a TensorFlow tensor or payload to a NumPy array.

    Parameters
    ----------
    value
        TensorFlow tensor, NumPy array, or scalar payload.

    Returns
    -------
    np.ndarray
        NumPy representation.
    """

    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    return np.asarray(value)


def _saved_payload(op: Any) -> Any:
    """Return an op's saved output payload or raise on missing data.

    Parameters
    ----------
    op
        Materialized TorchLens op.

    Returns
    -------
    Any
        Saved output payload.
    """

    if not bool(getattr(op, "has_saved_activation", False)):
        raise ValueError(f"op has no saved activation: {getattr(op, 'label', op)!r}")
    return op.out


def _captures_by_label(trace: Any) -> dict[str, TFOpCapture]:
    """Return TensorFlow callback captures keyed by raw labels.

    Parameters
    ----------
    trace
        Materialized TensorFlow trace.

    Returns
    -------
    dict[str, TFOpCapture]
        Captures keyed by raw output label.
    """

    return {
        capture.label_raw: capture
        for capture in getattr(trace, "_tf_op_captures", ())
        if isinstance(capture, TFOpCapture)
    }


__all__ = [
    "TFValidationCounts",
    "TFValidationResult",
    "replay_allowlist",
    "validate_tf_trace",
    "validate_tf_trace_detailed",
]
