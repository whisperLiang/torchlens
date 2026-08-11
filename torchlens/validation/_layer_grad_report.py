"""Layer-gradient parity report for PATH E module-output validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ._stock_layer_grads import ModuleOutputGradKey
else:
    ModuleOutputGradKey = tuple[str, int, int]

@dataclass
class LayerGradReport:
    """PATH E module-output gradient comparison report.

    Coverage is keyed by module-call label for single-output calls and by
    ``module:call[index]`` for multi-output calls. The classifier buckets are:

    - ``covered`` / ``mismatched``: eligible outputs that were compared;
    - classified LEGITIMATE exclusions (never block passing):
      ``skipped_root_module``, ``skipped_identity_output``,
      ``skipped_no_tensor_output`` (a module call that produced no captured
      tensor output has no meaningful first tensor leaf to compare), and the
      diagnostic-only ``skipped_module_less`` counter;
    - fail-closed gaps (any occurrence sinks the verdict):
      ``skipped_no_grad`` (an eligible output whose gradient was not captured)
      and ``unresolved_output_label`` (a module call names an output layer the
      trace cannot resolve — an internal inconsistency, not an exclusion).

    The acceptance rule is EXACT: 100% of the classified-eligible denominator
    must be ``covered``. There is no coverage-ratio tolerance; a tolerance
    here could hide missing hooks, which is a disarmed tripwire.
    """

    mode: Literal["module_output"]
    overall_passed: bool
    coverage: dict[str, str]
    covered_count: int
    skipped_no_tensor_output_count: int
    unresolved_output_label_count: int
    skipped_module_less_count: int
    skipped_no_grad_count: int
    skipped_identity_output_count: int
    skipped_root_module_count: int
    mismatched_count: int
    unexpected_count: int
    candidate_grad_count: int
    atol: float
    rtol: float
    mismatched_labels: tuple[str, ...] = ()
    max_abs_diffs: dict[str, float] = field(default_factory=dict)
    max_rel_diffs: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return the aggregate pass/fail result.

        Returns
        -------
        bool
            ``overall_passed``.
        """

        return self.overall_passed


def _compare_module_output_grads(
    trace: "Trace",
    stock_module_grads: Mapping[ModuleOutputGradKey, torch.Tensor],
    stock_identity_addresses: set[ModuleOutputGradKey],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> LayerGradReport:
    """Compare candidate module-call output grads to stock module-output grads.

    Parameters
    ----------
    trace:
        Candidate TorchLens trace with logged backward grads.
    stock_module_grads:
        Stock gradients keyed by ``(module_address, call_index, output_index)``.
    stock_identity_addresses:
        Module-output keys whose stock output is identical to input.
    atol:
        Absolute allclose tolerance.
    rtol:
        Relative allclose tolerance.

    Returns
    -------
    LayerGradReport
        Module-output gradient comparison report.
    """

    coverage: dict[str, str] = {}
    max_abs_diffs: dict[str, float] = {}
    max_rel_diffs: dict[str, float] = {}
    mismatched: list[str] = []
    candidate_grad_count = 0
    skipped_module_less_count = 0

    modules_map = getattr(trace, "modules", None)
    pass_dict = getattr(modules_map, "_pass_dict", {}) if modules_map is not None else {}
    for call_log in list(pass_dict.values()):
        addr = getattr(call_log, "address", None)
        call_index = getattr(call_log, "call_index", None)
        if addr is None or call_index is None:
            continue
        call_label = f"{addr}:{call_index}"
        if addr == "self":
            coverage[call_label] = "skipped_root_module"
            continue
        output_ops = (
            getattr(call_log, "output_ops", None) or getattr(call_log, "output_layers", None) or []
        )
        if not output_ops:
            # Classified legitimate exclusion: no captured tensor output means
            # there is no meaningful first tensor leaf to compare.
            coverage[call_label] = "skipped_no_tensor_output"
            continue
        multi_output = len(output_ops) > 1
        for output_index, output_label in enumerate(output_ops):
            coverage_label = f"{call_label}[{output_index}]" if multi_output else call_label
            try:
                cand_layer = trace[output_label]
            except (KeyError, IndexError):
                # Fail-closed gap: a module call naming an output layer the
                # trace cannot resolve is an internal inconsistency, never a
                # legitimate exclusion.
                coverage[coverage_label] = "unresolved_output_label"
                continue
            key = (addr, call_index, output_index)
            if key in stock_identity_addresses:
                coverage[coverage_label] = "skipped_identity_output"
                continue
            cand_grad = getattr(cand_layer, "grad", None)
            if cand_grad is None:
                coverage[coverage_label] = "skipped_no_grad"
                continue
            stock_grad = stock_module_grads.get(key)
            if stock_grad is None:
                coverage[coverage_label] = "skipped_no_grad"
                continue
            if cand_grad.shape != stock_grad.shape:
                coverage[coverage_label] = "mismatched"
                mismatched.append(coverage_label)
                continue
            abs_diff = (cand_grad - stock_grad).abs()
            max_abs_diffs[coverage_label] = abs_diff.max().item()
            max_rel_diffs[coverage_label] = (
                (abs_diff / stock_grad.abs().clamp(min=1e-30)).max().item()
            )
            if torch.allclose(cand_grad, stock_grad, atol=atol, rtol=rtol):
                coverage[coverage_label] = "covered"
            else:
                coverage[coverage_label] = "mismatched"
                mismatched.append(coverage_label)

    for layer in trace.layer_list:
        if not getattr(layer, "has_grad", False):
            continue
        candidate_grad_count += 1
        if not (getattr(layer, "modules", None) or []):
            skipped_module_less_count += 1

    covered_count = sum(value == "covered" for value in coverage.values())
    mismatched_count = sum(value == "mismatched" for value in coverage.values())
    skipped_no_tensor_output_count = sum(
        value == "skipped_no_tensor_output" for value in coverage.values()
    )
    unresolved_output_label_count = sum(
        value == "unresolved_output_label" for value in coverage.values()
    )
    skipped_no_grad_count = sum(value == "skipped_no_grad" for value in coverage.values())
    skipped_identity_output_count = sum(
        value == "skipped_identity_output" for value in coverage.values()
    )
    skipped_root_module_count = sum(value == "skipped_root_module" for value in coverage.values())
    unexpected_count = sum(value == "unexpected" for value in coverage.values())

    # Eligibility-classifier acceptance (replaces the former 0.80 coverage
    # ratio): the classified-eligible denominator is {covered, mismatched,
    # skipped_no_grad, unresolved_output_label} and 100% of it must be
    # covered. Legitimate exclusions were classified out above; any
    # unexplained gap fails closed rather than hiding inside a tolerance.
    overall_passed = (
        unexpected_count == 0
        and mismatched_count == 0
        and skipped_no_grad_count == 0
        and unresolved_output_label_count == 0
        and covered_count > 0
    )

    return LayerGradReport(
        mode="module_output",
        overall_passed=overall_passed,
        coverage=coverage,
        covered_count=covered_count,
        skipped_no_tensor_output_count=skipped_no_tensor_output_count,
        unresolved_output_label_count=unresolved_output_label_count,
        skipped_module_less_count=skipped_module_less_count,
        skipped_no_grad_count=skipped_no_grad_count,
        skipped_identity_output_count=skipped_identity_output_count,
        skipped_root_module_count=skipped_root_module_count,
        mismatched_count=mismatched_count,
        unexpected_count=unexpected_count,
        candidate_grad_count=candidate_grad_count,
        atol=atol,
        rtol=rtol,
        mismatched_labels=tuple(mismatched),
        max_abs_diffs=max_abs_diffs,
        max_rel_diffs=max_rel_diffs,
    )
