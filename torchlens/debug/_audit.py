"""One-call health reports assembled from trace-local debug diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from ._common import _compute_ops
from ._cost import hot_path
from ._gradients import gradient_flow_audit
from ._graph import dead_neurons
from ._nan import _nonfinite_kind, bisect_nan, find_nan_in_trace
from ._recompute import recompute_candidates


if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace
    from torchlens.partial import PartialTrace


AuditSeverity = Literal["critical", "warning", "info"]
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class AuditFinding:
    """One prioritized health finding from :func:`audit_trace`.

    Parameters
    ----------
    severity:
        Priority assigned to the finding.
    check:
        Name of the diagnostic that produced the finding.
    message:
        Human-readable result summary.
    ops:
        Offending operation labels.
    modules:
        Offending module addresses.
    follow_up:
        Call users can run to inspect the issue further.
    """

    severity: AuditSeverity
    check: str
    message: str
    ops: tuple[str, ...]
    modules: tuple[str, ...]
    follow_up: str


@dataclass(frozen=True)
class TraceAudit:
    """Notebook-friendly report from trace-local health diagnostics.

    Parameters
    ----------
    findings:
        Severity-ordered findings.
    checks_run:
        Names of diagnostics that ran.
    skipped:
        ``(check, reason)`` entries for diagnostics not supported by the capture.
    """

    findings: tuple[AuditFinding, ...]
    checks_run: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]

    def __repr__(self) -> str:
        """Render a compact audit suitable for notebooks.

        Returns
        -------
        str
            Health summary with findings and skipped-check reasons.
        """

        heading = (
            f"TraceAudit: {len(self.findings)} issue(s); {len(self.checks_run)} checks run, "
            f"{len(self.skipped)} skipped"
        )
        if not self.findings:
            heading = (
                f"TraceAudit: no issues found; {len(self.checks_run)} checks run, "
                f"{len(self.skipped)} skipped"
            )
        lines = [heading]
        lines.extend(
            f"- [{finding.severity}] {finding.check}: {finding.message} "
            f"Follow up: {finding.follow_up}"
            for finding in self.findings
        )
        lines.extend(f"- skipped {check}: {reason}" for check, reason in self.skipped)
        return "\n".join(lines)


def _has_full_saved_activations(trace: "Trace") -> bool:
    """Return whether every compute operation retains an output payload.

    Parameters
    ----------
    trace:
        Completed trace to inspect.

    Returns
    -------
    bool
        Whether payload-dependent checks can cover the complete computation.
    """

    return all(bool(getattr(op, "has_saved_activation", False)) for op in _compute_ops(trace))


def _has_saved_gradients(trace: "Trace") -> tuple[bool, str | None]:
    """Determine whether a trace supports a gradient-flow check.

    Parameters
    ----------
    trace:
        Completed trace to inspect.

    Returns
    -------
    tuple[bool, str | None]
        Support flag and a skip reason when unsupported.
    """

    try:
        if len(trace.backward_passes) == 0:
            return False, "forward-only trace; no backward pass was captured"
        if len(trace.saved_grad_ops) == 0:
            return False, "no saved gradients; re-trace with save_grads=True and log_backward()"
    except ValueError as exc:
        return False, str(exc)
    return True, None


def _audit_partial_trace(partial: "PartialTrace") -> TraceAudit:
    """Audit a failed partial capture without full-trace assumptions.

    Parameters
    ----------
    partial:
        Partial capture and its original exception.

    Returns
    -------
    TraceAudit
        Evidence-backed exception/non-finite findings and explicit skipped scope.
    """

    raw_layers = partial.raw_layers
    last = raw_layers[-1] if raw_layers else None
    last_label = (
        str(getattr(last, "_label_raw", getattr(last, "_layer_label_raw", "unknown")))
        if last is not None
        else "unknown"
    )
    modules = tuple(
        str(module)
        for module in (getattr(last, "module_call_stack", ()) if last is not None else ())
    )
    exception = partial.original_exception
    findings = [
        AuditFinding(
            severity="critical",
            check="partial_capture_exception",
            message=f"{type(exception).__name__}: {exception}",
            ops=(last_label,) if last is not None else (),
            modules=modules,
            follow_up="tl.report.explain(partial, format='json')",
        )
    ]
    nonfinite = partial.first_nonfinite()
    checks_run = ["partial_capture_exception", "find_nan"]
    if not nonfinite.startswith("No non-finite"):
        nonfinite_op = next(
            (
                op
                for op in raw_layers
                if isinstance((output := getattr(op, "out", None)), torch.Tensor)
                and _nonfinite_kind(output) != "none"
            ),
            None,
        )
        nonfinite_label = (
            str(
                getattr(
                    nonfinite_op,
                    "_label_raw",
                    getattr(nonfinite_op, "_layer_label_raw", "unknown"),
                )
            )
            if nonfinite_op is not None
            else "unknown"
        )
        nonfinite_modules = tuple(
            str(module) for module in getattr(nonfinite_op, "module_call_stack", ())
        )
        findings.append(
            AuditFinding(
                severity="critical",
                check="find_nan",
                message=nonfinite,
                ops=(nonfinite_label,) if nonfinite_op is not None else (),
                modules=nonfinite_modules,
                follow_up="partial.first_nonfinite()",
            )
        )
    reason = "partial capture did not complete full-trace postprocessing"
    skipped = tuple(
        (check, reason)
        for check in (
            "bisect_nan",
            "compare",
            "dead_neurons",
            "gradient_flow_audit",
            "hot_path",
            "infer_input_shape",
            "lineage",
            "recompute_candidates",
        )
    )
    findings.sort(
        key=lambda finding: (_SEVERITY_ORDER[finding.severity], finding.check, finding.ops)
    )
    return TraceAudit(tuple(findings), tuple(checks_run), skipped)


def audit_trace(trace: "Trace | PartialTrace") -> TraceAudit:
    """Run every trace-local health diagnostic supported by one capture.

    Diagnostics requiring a second trace, a selected start operation, or a
    fresh model execution are explicitly listed as skipped. Sparse traces run
    ``find_nan`` over their saved outputs and report its uncertainty zone;
    checks that require complete payload coverage remain skipped.

    Parameters
    ----------
    trace:
        Completed TorchLens trace or failed :class:`PartialTrace`.

    Returns
    -------
    TraceAudit
        Severity-ordered findings, executed checks, and honest skip reasons.
    """

    from torchlens.partial import PartialTrace

    if isinstance(trace, PartialTrace):
        return _audit_partial_trace(trace)

    findings: list[AuditFinding] = []
    checks_run: list[str] = []
    skipped: list[tuple[str, str]] = [
        ("compare", "requires a second trace"),
        ("lineage", "requires a selected starting operation"),
        ("infer_input_shape", "requires a model and a new probe execution"),
    ]
    result = find_nan_in_trace(trace)
    checks_run.append("find_nan")
    if result.found:
        findings.append(
            AuditFinding(
                severity="critical",
                check="find_nan",
                message=result.message,
                ops=(result.label,) if result.label is not None else (),
                modules=(result.module_address,) if result.module_address is not None else (),
                follow_up="trace.find_nan()",
            )
        )

    full_payloads = _has_full_saved_activations(trace)
    if full_payloads:
        bisect_nan(trace)
        checks_run.append("bisect_nan")
        dead_neurons(trace)
        checks_run.append("dead_neurons")
    else:
        reason = "selective-save trace does not retain every compute activation"
        skipped.extend([("bisect_nan", reason), ("dead_neurons", reason)])

    has_gradients, gradient_reason = _has_saved_gradients(trace)
    if has_gradients:
        frame = gradient_flow_audit(trace)
        checks_run.append("gradient_flow_audit")
        for _, row in frame[frame["severity"] > 0].iterrows():
            findings.append(
                AuditFinding(
                    severity="critical" if bool(row["exploding"]) else "warning",
                    check="gradient_flow_audit",
                    message=str(row["reason"] or "gradient-flow anomaly"),
                    ops=(str(row["op"]),),
                    modules=(),
                    follow_up="tl.debug.gradient_flow_audit(trace)",
                )
            )
    else:
        skipped.append(("gradient_flow_audit", gradient_reason or "saved gradients unavailable"))

    # These trace-local rankings are useful contextual diagnostics but do not
    # themselves establish a model-health issue.
    hot_path(trace, by="flops")
    checks_run.append("hot_path")
    recompute_candidates(trace)
    checks_run.append("recompute_candidates")
    findings.sort(
        key=lambda finding: (_SEVERITY_ORDER[finding.severity], finding.check, finding.ops)
    )
    return TraceAudit(tuple(findings), tuple(checks_run), tuple(skipped))
