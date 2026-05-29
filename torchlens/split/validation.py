"""Validation helpers for TorchLens split runtimes."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal
from typing import Any

import torch

from .boundary import ReplayBoundary
from .errors import SplitErrorContext, SplitUnsupportedError


@dataclass
class SplitSweepCaseResult:
    """Result for one split target in a split sweep."""

    model_name: str
    split_target: str
    boundary: str
    status: Literal["supported", "unsupported", "failed"]
    batch_sizes_tested: tuple[int, ...]
    unsupported_reason: str | None = None
    module_path: str | None = None
    op_type: str | None = None
    torchlens_label: str | None = None
    error_message: str | None = None
    traced_shape: Any = None
    runtime_shape: Any = None
    dtype: str | None = None
    batch_statuses: dict[str, Literal["supported", "unsupported", "failed"]] = field(
        default_factory=dict
    )


@dataclass
class SplitSweepCoverageReport:
    """Coverage summary for a split sweep."""

    model_name: str
    total_targets: int
    supported_targets: int
    unsupported_targets: int
    failed_targets: int
    support_ratio: float
    unsupported_by_reason: dict[str, int]
    unsupported_by_module_path: dict[str, int]
    unsupported_by_op_type: dict[str, int]
    cases: list[SplitSweepCaseResult]


def canonical_boundary_signature(runtime: Any) -> tuple[tuple[Any, ...], ...]:
    """Return device-neutral canonical boundary signature."""

    rows = []
    for label, spec in runtime.boundary_spec.items():
        rows.append((label, spec.canonical_id, spec.shape, spec.dtype, spec.role))
    return tuple(rows)


def split_unsupported_context(exc: SplitUnsupportedError) -> SplitErrorContext | None:
    """Return structured unsupported context from an exception, when available."""

    context = getattr(exc, "context", None)
    return context if isinstance(context, SplitErrorContext) else None


def make_split_sweep_case_result(
    *,
    model_name: str,
    split_target: str,
    boundary: str,
    status: Literal["supported", "unsupported", "failed"],
    batch_sizes_tested: tuple[int, ...],
    batch_statuses: dict[str, Literal["supported", "unsupported", "failed"]] | None = None,
    exc: BaseException | None = None,
) -> SplitSweepCaseResult:
    """Build a split sweep case result from optional exception context."""

    context = split_unsupported_context(exc) if isinstance(exc, SplitUnsupportedError) else None
    return SplitSweepCaseResult(
        model_name=model_name,
        split_target=split_target,
        boundary=boundary,
        status=status,
        batch_sizes_tested=batch_sizes_tested,
        unsupported_reason=context.reason if context is not None else None,
        module_path=context.module_path if context is not None else None,
        op_type=context.op_type if context is not None else None,
        torchlens_label=context.layer_label if context is not None else None,
        error_message=str(exc) if exc is not None else None,
        traced_shape=context.traced_shape if context is not None else None,
        runtime_shape=context.runtime_shape if context is not None else None,
        dtype=str(context.dtype) if context is not None and context.dtype is not None else None,
        batch_statuses=batch_statuses or {},
    )


def summarize_split_sweep_coverage(
    model_name: str,
    cases: list[SplitSweepCaseResult],
) -> SplitSweepCoverageReport:
    """Summarize split sweep case results into support coverage metrics."""

    unsupported_cases = [case for case in cases if case.status == "unsupported"]
    supported_targets = sum(case.status == "supported" for case in cases)
    unsupported_targets = len(unsupported_cases)
    failed_targets = sum(case.status == "failed" for case in cases)
    total_targets = len(cases)
    support_ratio = supported_targets / total_targets if total_targets else 0.0
    return SplitSweepCoverageReport(
        model_name=model_name,
        total_targets=total_targets,
        supported_targets=supported_targets,
        unsupported_targets=unsupported_targets,
        failed_targets=failed_targets,
        support_ratio=support_ratio,
        unsupported_by_reason=_count_optional(case.unsupported_reason for case in unsupported_cases),
        unsupported_by_module_path=_count_optional(case.module_path for case in unsupported_cases),
        unsupported_by_op_type=_count_optional(case.op_type for case in unsupported_cases),
        cases=cases,
    )


def write_split_sweep_coverage_report(
    report: SplitSweepCoverageReport,
    path: str | Path,
) -> None:
    """Write a split sweep coverage report as stable JSON."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_json_ready(asdict(report)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_split_sweep_coverage(
    report: SplitSweepCoverageReport,
    min_support_ratio: float,
    *,
    allow_unsupported: bool = True,
) -> None:
    """Assert split sweep coverage did not regress past the requested threshold."""

    if report.supported_targets == 0:
        raise AssertionError(f"{report.model_name} split sweep supported zero targets.")
    if report.failed_targets > 0:
        raise AssertionError(
            f"{report.model_name} split sweep had {report.failed_targets} failed target(s)."
        )
    if not allow_unsupported and report.unsupported_targets > 0:
        raise AssertionError(
            f"{report.model_name} split sweep had {report.unsupported_targets} unsupported target(s)."
        )
    if report.support_ratio < min_support_ratio:
        raise AssertionError(
            f"{report.model_name} split sweep support ratio {report.support_ratio:.2%} "
            f"is below {min_support_ratio:.2%}."
        )


def format_split_sweep_coverage_summary(
    report: SplitSweepCoverageReport,
    *,
    top_n: int = 5,
) -> str:
    """Return a compact pytest-friendly split sweep coverage summary."""

    lines = [
        f"[split sweep coverage] {report.model_name}",
        (
            f"total={report.total_targets} supported={report.supported_targets} "
            f"unsupported={report.unsupported_targets} failed={report.failed_targets} "
            f"support_ratio={report.support_ratio:.2%}"
        ),
        "top unsupported reasons:",
    ]
    if report.unsupported_by_reason:
        for reason, count in sorted(
            report.unsupported_by_reason.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]:
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("  none")
    return "\n".join(lines)


def assert_split_unsupported_error_context(
    exc: SplitUnsupportedError,
    expected_boundary: str,
) -> SplitErrorContext:
    """Assert unsupported split errors carry actionable structured context."""

    context = split_unsupported_context(exc)
    if context is None:
        raise AssertionError("SplitUnsupportedError is missing structured context.")
    missing = []
    if context.split_point != expected_boundary:
        missing.append("split_point")
    for field_name in ("module_path", "op_type", "layer_label", "reason"):
        if not getattr(context, field_name):
            missing.append(field_name)
    if context.traced_shape is None:
        missing.append("traced_shape")
    if context.dtype is None:
        missing.append("dtype")
    if missing:
        raise AssertionError(
            f"SplitUnsupportedError context is missing/invalid fields {missing}: {exc}"
        )
    return context


def assert_canonical_abi_equivalent(left: Any, right: Any) -> None:
    """Assert two runtimes have the same device-neutral boundary ABI."""

    if canonical_boundary_signature(left) != canonical_boundary_signature(right):
        raise AssertionError("Split canonical boundary ABI differs.")


def validate_dynamic_batches(
    runtime: Any,
    input_factory: Any,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Validate split replay equivalence for multiple runtime batch sizes."""

    for batch_size in batch_sizes:
        inputs = input_factory(batch_size)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not runtime.validate_equivalence(runtime.model, inputs, atol=atol, rtol=rtol):
            raise AssertionError(f"Split replay mismatch for batch size {batch_size}.")


def boundary_liveness_zero_probe(
    runtime: Any,
    boundary: ReplayBoundary,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, bool]:
    """Probe whether each boundary tensor affects suffix output."""

    baseline = runtime.run_suffix(boundary)
    live: dict[str, bool] = {}
    for tensor_id in boundary.tensors:
        candidate = boundary.clone()
        candidate.tensors[tensor_id].zero_()
        output = runtime.run_suffix(candidate)
        live[tensor_id] = not _tree_allclose(baseline, output, atol=atol, rtol=rtol)
    return live


def _tree_allclose(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        return all(_tree_allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_tree_allclose(left[key], right[key], atol=atol, rtol=rtol) for key in left)
    return left == right


def _count_optional(values: Any) -> dict[str, int]:
    counts: Counter[str] = Counter(str(value) for value in values if value)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


__all__ = [
    "SplitSweepCaseResult",
    "SplitSweepCoverageReport",
    "assert_split_sweep_coverage",
    "assert_split_unsupported_error_context",
    "assert_canonical_abi_equivalent",
    "boundary_liveness_zero_probe",
    "canonical_boundary_signature",
    "format_split_sweep_coverage_summary",
    "make_split_sweep_case_result",
    "split_unsupported_context",
    "summarize_split_sweep_coverage",
    "validate_dynamic_batches",
    "write_split_sweep_coverage_report",
]
