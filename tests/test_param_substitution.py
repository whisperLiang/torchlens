"""Parameter substitution ("as if" the parameter were changed) — replay-only.

Pins the JMT-ruled param-operand contract (2026-08-17): ``fork.do(
tl.params(...), edit)`` substitutes the VALUE each consuming op sees at its
derived occurrence address on the replay engine, through the shipped
tier-(ii) edge-substitution store, and the live parameter object is never
written (bit-identical proof below). Engine scope (rerun/set_only refuse
``param_substitution_engine_unsupported``), fail-closed occurrence
derivation (``param_substitution_occurrence_underivable``), edit-then-
scatter masking, validation boundary corroboration (DIFFERENT check, never
NO check), and re-splice persistence under later pushes are all pinned.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.selection import SelectionError
from torchlens.validation.core import _check_edge_intervention_boundary

pytestmark = pytest.mark.smoke


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _capture():
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    return model, x, trace


@pytest.fixture(scope="module")
def capture():
    model, x, trace = _capture()
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _as_if_forward(model: _Net, x: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.fc2(torch.relu(x @ w1.T + model.fc1.bias))


def test_basic_as_if_substitution(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.0))
    expected = _as_if_forward(model, x, torch.zeros_like(model.fc1.weight))
    assert torch.allclose(fork.output_ops[0].out, expected, atol=1e-6)
    # the source trace is untouched (fork is the sanctioned mutation surface)
    with torch.no_grad():
        assert torch.allclose(trace.output_ops[0].out, model(x))


def test_live_parameter_bit_identical(capture):
    """THE ruling's core guarantee: the parameter object is never written."""

    model, x, trace = capture
    weight = model.fc1.weight
    before_bytes = weight.detach().clone()
    before_ptr = weight.data_ptr()
    before_version = weight._version
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(3.0))
    assert weight.data_ptr() == before_ptr
    assert weight._version == before_version
    assert torch.equal(weight.detach(), before_bytes)
    assert weight.detach().view(torch.uint8).equal(before_bytes.view(torch.uint8)), (
        "parameter storage must be BIT-identical after an intervened replay"
    )


def test_masked_param_edit_scatters_only_selected_rows(capture):
    model, x, trace = capture
    mask = torch.zeros_like(model.fc1.weight, dtype=torch.bool)
    mask[0] = True
    fork = trace.fork()
    fork.do(tl.params("fc1.weight", mask=mask), tl.scale(0.0))
    as_if = model.fc1.weight.detach().clone()
    as_if[0] = 0.0
    expected = _as_if_forward(model, x, as_if)
    assert torch.allclose(fork.output_ops[0].out, expected, atol=1e-6)


def test_helpers_and_raw_tensor_compose(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.add(0.25))
    expected = _as_if_forward(model, x, model.fc1.weight.detach() + 0.25)
    assert torch.allclose(fork.output_ops[0].out, expected, atol=1e-6)
    replacement = torch.randn_like(model.fc1.weight)
    fork2 = trace.fork()
    fork2.do(tl.params("fc1.weight"), replacement)
    expected2 = _as_if_forward(model, x, replacement)
    assert torch.allclose(fork2.output_ops[0].out, expected2, atol=1e-6)


def test_engine_scoping_refusals(capture):
    model, x, trace = capture
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(
            tl.params("fc1.weight"),
            tl.scale(0.0),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert excinfo.value.fields["code"] == "param_substitution_engine_unsupported"
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(
            tl.params("fc1.weight"),
            tl.scale(0.0),
            model=model,
            x=x,
            intervention=tl.options.InterventionOptions(engine="rerun"),
        )
    assert excinfo.value.fields["code"] == "param_substitution_engine_unsupported"
    # auto with model+x resolves to rerun -> same refusal
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(tl.params("fc1.weight"), tl.scale(0.0), model=model, x=x)
    assert excinfo.value.fields["code"] == "param_substitution_engine_unsupported"


def test_multipass_consumer_refuses_typed():
    """Recurrently reused params (multi-pass consumers) refuse fail-closed."""

    torch.manual_seed(0)

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 4) * 0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x @ self.w) @ self.w

    model = Tied()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        with pytest.raises(SelectionError) as excinfo:
            trace.fork().do(tl.params("w"), tl.scale(2.0))
        assert excinfo.value.fields["code"] == "param_substitution_occurrence_underivable"
    finally:
        trace.cleanup()


def test_zero_mask_is_disclosure_not_error(capture):
    model, x, trace = capture
    fork = trace.fork()
    before = fork.output_ops[0].out.clone()
    fork.do(
        tl.params("fc1.weight", mask=torch.zeros_like(model.fc1.weight, dtype=torch.bool)),
        tl.scale(0.0),
    )
    assert torch.equal(fork.output_ops[0].out, before)
    assert fork.intervention_audit[-1]["kind"] == "PARAM"
    assert fork.intervention_audit[-1]["params"] == []


def test_audit_record_discloses_substitution_not_change(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    audit = fork.intervention_audit[-1]
    assert audit["kind"] == "PARAM"
    assert "substituted at consumption" in audit["disclosure"]
    assert "live parameters unchanged" in audit["disclosure"]
    (param_row,) = audit["params"]
    assert param_row["param_address"] == "fc1.weight"
    (occurrence,) = param_row["occurrences"]
    assert occurrence["consumer"] == param_row["consumers"][0]
    assert occurrence["value_digest"]


def test_fire_record_and_store_mark_param_kind(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    child = fork["fc1"].ops[0]
    (store_key,) = child.edge_substitutions.keys()
    payload = child.edge_substitutions[store_key]
    assert payload["substitution_kind"] == "param"
    assert payload["param_address"] == "fc1.weight"
    stamp = child.edge_replacement_stamps[store_key]
    assert stamp["substitution_kind"] == "param"
    fire = next(r for r in child.interventions if r.edge_address is not None)
    assert fire.replaced is False  # substitution replaces no node's OUTPUT
    assert fire.engine == "replay"


def test_validation_boundary_corroborates(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    child = fork["fc1"].ops[0]
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None
    assert verdict.decision == "edge_intervention_boundary"  # DISTINCT term, never "exempted"
    assert not verdict.failed


def test_validation_tamper_strip_fire_record_fails(capture):
    """Tripwire intact: an uncorroborated param entry FAILS validation."""

    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    child = fork["fc1"].ops[0]
    child._internal_set("interventions", [r for r in child.interventions if not r.edge_address])
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_uncorroborated"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is False


def test_substitution_survives_later_push(capture):
    """Param-kind entries re-splice during cone recomputation (never revert)."""

    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.0))
    after_do = fork.output_ops[0].out.clone()

    def _noop(out: torch.Tensor, *, hook) -> torch.Tensor:
        return out

    fork.do(tl.func("relu"), _noop)
    assert torch.allclose(fork.output_ops[0].out, after_do)


def test_foreign_resolved_selection_refuses(capture):
    model, x, trace = capture
    other_model, other_x, other = _capture()
    try:
        resolved = tl.params("fc1.weight").resolve(other)
        with pytest.raises(SelectionError) as excinfo:
            trace.fork().do(resolved, tl.scale(0.0))
        assert excinfo.value.fields["code"] == "selection_trace_mismatch"
    finally:
        other.cleanup()
