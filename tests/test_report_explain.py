"""Tests for ``torchlens.report.explain``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes import FuncCallLocation


class TinyReportModel(nn.Module):
    """Small deterministic model for report tests."""

    def __init__(self) -> None:
        """Initialize deterministic weights."""

        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a tiny nonlinear forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU-transformed projection.
        """

        return torch.relu(self.proj(x))


class FailingShapeModel(nn.Module):
    """Model that fails at a recorded shape-mismatch boundary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fail after completing a ReLU and constant construction.

        Parameters
        ----------
        x:
            Input tensor with four features.

        Returns
        -------
        torch.Tensor
            Unreachable matrix product.
        """

        activated = torch.relu(x)
        return activated @ torch.ones(3, 2)


def _captured_log() -> tl.Trace:
    """Return a deterministic captured log.

    Returns
    -------
    tl.Trace
        Captured log for ``TinyReportModel``.
    """

    return tl.trace(TinyReportModel(), torch.tensor([[2.0, 3.0]]))


def test_report_namespace_is_not_top_level_all() -> None:
    """``tl.report.explain`` should be reachable without expanding ``tl.__all__``.

    The namespace size is checked against the namespace itself so this test
    guards the report names without hard-coding unrelated top-level API churn.
    """

    assert hasattr(tl.report, "explain")
    public_names = set(tl.__all__)
    assert len(tl.__all__) == len(public_names)
    assert "report" not in tl.__all__
    assert "explain" not in tl.__all__


def test_explain_returns_sensible_string_for_each_audience() -> None:
    """All supported audiences should produce a human-readable report."""

    log = _captured_log()
    for audience in ("researcher", "practitioner", "auto"):
        text = tl.report.explain(log, audience=audience)
        assert isinstance(text, str)
        assert "TorchLens report" in text
        assert "Model summary" in text
        assert "Capture summary" in text
        assert "Backward summary" in text
        assert "Anomalies" in text
        assert "Interventions" in text
        assert "Notable patterns" in text
        assert "TinyReportModel" in text
        assert "No backward passes are recorded" in text


def test_explain_json_uses_stable_full_trace_schema() -> None:
    """JSON reports expose the documented v1 keys for complete traces."""

    report = tl.report.explain(_captured_log(), format="json")

    assert isinstance(report, dict)
    assert report["schema"] == "torchlens.explain.v1"
    assert report["capture_status"] == "complete"
    assert {
        "audience",
        "model_class",
        "layer_count",
        "operation_count",
        "saved_tensor_count",
        "total_tensor_count",
        "has_backward_pass",
        "exception_type",
        "exception_message",
        "last_completed_op_label",
        "last_completed_op_shape",
        "last_completed_op_dtype",
        "last_completed_op_device",
        "failing_boundary",
        "first_nonfinite",
    }.issubset(report)


def test_explain_and_audit_diagnose_real_partial_failure() -> None:
    """Partial reports name recorded last-op and captured exception evidence."""

    with pytest.raises(RuntimeError) as exc_info:
        tl.trace(FailingShapeModel(), torch.randn(2, 4))
    partial = tl.partial.from_failed_capture(exc_info.value)
    last = partial.raw_layers[-1]
    last_label = str(getattr(last, "_label_raw"))

    text = tl.report.explain(partial)
    report = tl.report.explain(partial, format="json")
    audit = partial.audit()

    assert isinstance(text, str)
    assert "This is a partial capture" in text
    assert last_label in text
    assert "RuntimeError" in text
    assert "mat1 and mat2 shapes cannot be multiplied" in text
    assert isinstance(report, dict)
    assert report["capture_status"] == "partial"
    assert report["last_completed_op_label"] == last_label
    assert report["exception_type"] == "RuntimeError"
    assert report["exception_message"] == str(exc_info.value)
    assert "forward at" in report["failing_boundary"]
    assert audit == tl.debug.audit_trace(partial)
    failure = next(
        finding for finding in audit.findings if finding.check == "partial_capture_exception"
    )
    assert failure.message == f"RuntimeError: {exc_info.value}"
    assert failure.ops == (last_label,)
    assert all("may" not in finding.message.lower() for finding in audit.findings)


def test_operational_status_line_reports_real_streamed_ops_not_a_fake_constant(
    tmp_path: Path,
) -> None:
    """``streamed_ops`` must reflect real streaming state, not a hardcoded ``1``.

    Regression for a bug where ``_operational_status_line`` always printed
    ``streamed_ops=1`` regardless of whether the trace used streaming at all.
    """

    from torchlens.report._explain import _operational_status_line

    plain_log = _captured_log()
    plain_line = _operational_status_line(plain_log)
    assert "streamed_ops=0" in plain_line

    plain_text = tl.report.explain(plain_log, audience="practitioner")
    assert "streamed_ops=0" in plain_text

    bundle_path = tmp_path / "streamed.tlspec"
    streamed_log = tl.trace(
        TinyReportModel(),
        torch.tensor([[2.0, 3.0]]),
        storage=tl.to_disk(bundle_path, retain_in_memory=False),
    )
    streamed_line = _operational_status_line(streamed_log)
    assert "streamed_ops=0" not in streamed_line
    num_layers = len(streamed_log.layer_list)
    assert f"streamed_ops={num_layers}" in streamed_line

    streamed_text = tl.report.explain(streamed_log, audience="practitioner")
    assert f"streamed_ops={num_layers}" in streamed_text


def test_explain_reports_backward_capture() -> None:
    """Backward logs should include pass, GradFn, and saved-gradient counts."""

    x = torch.tensor([[2.0, 3.0]], requires_grad=True)
    log = tl.trace(TinyReportModel(), x, save_grads=True)
    log.log_backward(log[log.output_layers[0]].out.sum())

    text = tl.report.explain(log)

    assert "Backward summary" in text
    assert "Backward passes: 1." in text
    assert "GradFn records:" in text
    assert "Op gradient records saved:" in text


def test_explain_reports_nonfinite_out() -> None:
    """The anomaly section should flag saved NaN or Inf outs."""

    log = _captured_log()
    log["linear_1_1"].out[0, 0] = torch.nan
    text = tl.report.explain(log)
    assert "NaN or Inf" in text
    assert "linear_1_1" in text
    assert "vscode://file/" in log.first_nonfinite(link_format="html")


def test_source_locations_keep_repr_plain_and_expose_html_links() -> None:
    """Source locations should keep repr plain and expose VS Code HTML links."""

    location = FuncCallLocation(
        file="/tmp/demo.py",
        line_number=12,
        func_name="forward",
        source_loading_enabled=False,
    )
    assert "\033]8;;file://" not in repr(location)
    assert "vscode://file/" in location.to_html_link()
