"""PATH E per-module-output gradient oracle tests."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import pytest
import torch
from torch import nn

from torchlens.validation import backward as backward_validation
from torchlens.validation.invariants import MetadataInvariantError
from torchlens.validation._layer_grad_report import (
    LayerGradReport,
    _compare_module_output_grads,
)
from torchlens.validation._stock_layer_grads import (
    _StockModuleGradCollector,
    _candidate_module_call_for,
    _candidate_root_module,
    _first_leaf_tensor,
    _pass_index_from_layer_modules,
    _tensor_leaves,
)


class TinyMLP(nn.Module):
    """Small MLP used by the module-output oracle tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.l1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.l2(self.relu(self.l1(x)))


class TinyRNN(nn.Module):
    """Small recurrent model used as the RNN acceptance fixture."""

    def __init__(self) -> None:
        """Initialize recurrent cell and output head."""

        super().__init__()
        self.cell = nn.RNNCell(3, 5)
        self.out = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent steps.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 3, 3)``.

        Returns
        -------
        torch.Tensor
            Final output.
        """

        h = torch.zeros(x.shape[0], 5, device=x.device)
        for step in range(3):
            h = self.cell(x[:, step, :], h)
        return self.out(h)


class TinyResNet(nn.Module):
    """Small residual block used when torchvision is unavailable."""

    def __init__(self) -> None:
        """Initialize residual layers."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a residual forward pass."""

        return self.head(torch.relu(self.block(x) + x))


class IdentityWrapper(nn.Module):
    """Module with an identity-output submodule."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with identity output."""

        return self.identity(self.linear(x))


class NearIdentityWrapper(nn.Module):
    """Module with a numerically close but non-identity output."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.near = NearIdentity()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through a near-identity module."""

        return self.linear(self.near(x))


class NearIdentity(nn.Module):
    """Return values close to input without returning the same tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a numerically close but distinct tensor."""

        return x + 1e-6


class Pair(nn.Module):
    """Return two differentiable output tensors."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two scaled outputs."""

        return x * 2, x * 3


@dataclasses.dataclass
class TensorBox:
    """Dataclass container for first-leaf tests."""

    ignored: int
    tensor: torch.Tensor


class SyntheticTrace:
    """Minimal trace stub consumed by ``_compare_module_output_grads``."""

    def __init__(self, call_logs: list[Any], layers: dict[str, Any]) -> None:
        """Initialize the synthetic trace.

        Parameters
        ----------
        call_logs:
            Synthetic module-call logs.
        layers:
            Layer mapping by label.
        """

        self.modules = SimpleNamespace(_pass_dict={call.call_label: call for call in call_logs})
        self._layers = layers
        self.layer_list = list(layers.values())

    def __getitem__(self, label: str) -> Any:
        """Return one synthetic layer by label."""

        return self._layers[label]


def _loss(output: torch.Tensor) -> torch.Tensor:
    """Reduce a model output to a scalar loss."""

    return output.sum()


def _assert_acceptance(report: LayerGradReport) -> None:
    """Assert the eligibility-classifier module-output acceptance criteria.

    100% of the classified-eligible denominator must be covered: any
    mismatch, uncaptured-eligible gradient, or unresolved output label sinks
    the verdict (the former 0.80 ratio tolerance is gone).
    """

    assert report.overall_passed
    assert report.covered_count > 0
    assert report.mismatched_count == 0
    assert report.skipped_no_grad_count == 0
    assert report.unresolved_output_label_count == 0


def _run_public_layer_grad_validation(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any],
    loss_fn: Callable[[Any], torch.Tensor],
    *,
    atol: float,
    rtol: float,
    random_seed: int,
    validate_metadata: bool = True,
) -> LayerGradReport:
    """Run the shipped backward path and return its captured layer-grad report.

    Parameters
    ----------
    model:
        Model passed to the public backward validator.
    input_args:
        Positional model inputs.
    input_kwargs:
        Keyword model inputs.
    loss_fn:
        Scalar loss callable.
    atol:
        Absolute tolerance for parameter and layer gradients.
    rtol:
        Relative tolerance for parameter and layer gradients.
    random_seed:
        Seed shared by stock and captured passes.
    validate_metadata:
        Whether the shipped path should additionally run metadata invariants.
        The deleted private ``_validate_layer_grads`` copy these tests used to
        call never ran them, so passing ``False`` reproduces the ORIGINAL
        coverage exactly for the two real-world models that trip a PRE-EXISTING
        backward-metadata bug (see
        ``test_shipped_backward_path_resnet50_metadata_invariant_is_broken``).

    Returns
    -------
    LayerGradReport
        The report produced inside ``validate_backward_pass``.
    """

    reports: list[LayerGradReport] = []

    def capture_report(*args: Any, **kwargs: Any) -> LayerGradReport:
        """Record and return the production comparison report."""

        report = _compare_module_output_grads(*args, **kwargs)
        reports.append(report)
        return report

    with patch(
        "torchlens.validation._layer_grad_report._compare_module_output_grads",
        side_effect=capture_report,
    ):
        passed = backward_validation.validate_backward_pass(
            model,
            input_args,
            input_kwargs=input_kwargs,
            loss_fn=loss_fn,
            random_seed=random_seed,
            atol=atol,
            rtol=rtol,
            validate_metadata=validate_metadata,
            validate_layer_grads=True,
            layer_grad_atol=atol,
            layer_grad_rtol=rtol,
        )
    assert len(reports) == 1
    # A failing layer-grad report must sink the shipped verdict. The converse is
    # deliberately NOT asserted: the shipped path can still fail on parameter
    # gradients after an accepting layer report.
    if not bool(reports[0]):
        assert passed is False
    return reports[0]


def _synthetic_call(address: str, call_index: int, output_layers: list[str]) -> Any:
    """Build a synthetic module-call log."""

    return SimpleNamespace(
        address=address,
        call_index=call_index,
        call_label=f"{address}:{call_index}",
        output_layers=output_layers,
    )


def _synthetic_layer(
    label: str,
    grad: torch.Tensor | None,
    modules: list[str] | None = None,
) -> Any:
    """Build a synthetic layer log."""

    return SimpleNamespace(
        layer_label=label,
        grad=grad,
        has_grad=grad is not None,
        modules=modules if modules is not None else ["m:1"],
    )


def test_stock_module_grad_collector_captures_module_output_grad() -> None:
    """The stock collector captures gradients from module outputs."""

    model = TinyMLP()
    collector = _StockModuleGradCollector()
    collector.install(model)
    try:
        loss = model(torch.randn(2, 3)).sum()
        loss.backward()
        collector.collect_grads_after_backward()
    finally:
        collector.cleanup()
    assert ("l1", 1, 0) in collector.stock_module_output_grads
    assert ("l2", 1, 0) in collector.stock_module_output_grads


def test_stock_module_grad_collector_does_not_retain_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stock oracle captures via tensor hooks rather than ``retain_grad``."""

    def fail_retain_grad(tensor: torch.Tensor) -> None:
        """Fail if the stock oracle tries to retain output grads."""

        raise AssertionError("stock layer oracle must not call retain_grad")

    monkeypatch.setattr(torch.Tensor, "retain_grad", fail_retain_grad)
    model = TinyMLP()
    collector = _StockModuleGradCollector()
    collector.install(model)
    try:
        loss = model(torch.randn(2, 3)).sum()
        loss.backward()
        collector.collect_grads_after_backward()
    finally:
        collector.cleanup()
    assert ("l1", 1, 0) in collector.stock_module_output_grads


def test_stock_module_grad_collector_captures_all_output_slots() -> None:
    """The stock collector captures non-first module outputs."""

    model = Pair()
    collector = _StockModuleGradCollector()
    collector.install(model)
    try:
        first, second = model(torch.randn(2, 3, requires_grad=True))
        loss = first.sum() + second.sum()
        loss.backward()
        collector.collect_grads_after_backward()
    finally:
        collector.cleanup()

    assert ("", 1, 0) in collector.stock_module_output_grads
    assert ("", 1, 1) in collector.stock_module_output_grads


def test_tensor_leaf_helpers_traverse_supported_containers() -> None:
    """Tensor leaf discovery handles lists, dicts, and dataclasses."""

    first = torch.randn(1)
    second = torch.randn(1)
    assert _first_leaf_tensor({"a": None, "b": [first, second]}) is first
    assert _first_leaf_tensor(TensorBox(ignored=1, tensor=second)) is second
    assert _tensor_leaves({"a": None, "b": [first, second]}) == [first, second]


def test_pass_index_from_layer_modules_parses_string_and_tuple() -> None:
    """Module pass-index parsing supports current layer module formats."""

    assert _pass_index_from_layer_modules(SimpleNamespace(modules=["a.b:3"])) == 3
    assert _pass_index_from_layer_modules(SimpleNamespace(modules=[("a.b", 4)])) == 4


def test_candidate_module_lookup_helpers_use_pass_dict() -> None:
    """Candidate helper lookups resolve root and pass-qualified module calls."""

    call = _synthetic_call("block", 2, ["x"])
    root = _synthetic_call("self", 1, ["root"])
    trace = SyntheticTrace([call, root], {})
    assert _candidate_module_call_for(trace, "block", 2) is call
    assert _candidate_root_module(trace) is root


def test_compare_module_output_grads_reports_mismatched_bucket() -> None:
    """Failed allclose comparisons are mismatched, not covered."""

    grad = torch.ones(2, 2)
    trace = SyntheticTrace(
        [_synthetic_call("linear", 1, ["linear_out"])],
        {"linear_out": _synthetic_layer("linear_out", grad)},
    )
    report = _compare_module_output_grads(trace, {("linear", 1, 0): grad + 1}, set())
    assert report.coverage["linear:1"] == "mismatched"
    assert report.mismatched_count == 1
    assert not report.overall_passed


def test_overall_passed_requires_skipped_no_grad_zero() -> None:
    """One skipped-no-grad entry fails both 1+4 and 4+1 synthetic shapes."""

    for covered, skipped in ((1, 4), (4, 1)):
        calls = []
        layers = {}
        stock = {}
        for index in range(covered + skipped):
            address = f"m{index}"
            label = f"layer{index}"
            grad = torch.ones(1) if index < covered else None
            calls.append(_synthetic_call(address, 1, [label]))
            layers[label] = _synthetic_layer(label, grad)
            if grad is not None:
                stock[(address, 1, 0)] = grad.clone()
        report = _compare_module_output_grads(SyntheticTrace(calls, layers), stock, set())
        assert report.covered_count == covered
        assert report.skipped_no_grad_count == skipped
        assert not report.overall_passed


def test_compare_excludes_root_and_identity_from_denominator() -> None:
    """Root and identity outputs are classified but do not block passing."""

    grad = torch.ones(1)
    trace = SyntheticTrace(
        [
            _synthetic_call("self", 1, ["root"]),
            _synthetic_call("id", 1, ["id_out"]),
            _synthetic_call("linear", 1, ["linear_out"]),
        ],
        {
            "root": _synthetic_layer("root", grad),
            "id_out": _synthetic_layer("id_out", grad),
            "linear_out": _synthetic_layer("linear_out", grad),
        },
    )
    report = _compare_module_output_grads(
        trace,
        {("linear", 1, 0): grad},
        {("id", 1, 0)},
    )
    assert report.coverage["self:1"] == "skipped_root_module"
    assert report.coverage["id:1"] == "skipped_identity_output"
    assert report.overall_passed


def test_compare_counts_no_tensor_output() -> None:
    """Module calls with no captured tensor output are a classified exclusion."""

    report = _compare_module_output_grads(
        SyntheticTrace([_synthetic_call("empty", 1, [])], {}),
        {},
        set(),
    )
    assert report.skipped_no_tensor_output_count == 1
    assert report.coverage["empty:1"] == "skipped_no_tensor_output"


def test_unresolved_output_label_fails_closed() -> None:
    """A module call naming an unresolvable output layer sinks the verdict."""

    grad = torch.ones(1)
    trace = SyntheticTrace(
        [
            _synthetic_call("linear", 1, ["linear_out"]),
            _synthetic_call("ghost", 1, ["missing_label"]),
        ],
        {"linear_out": _synthetic_layer("linear_out", grad)},
    )
    report = _compare_module_output_grads(trace, {("linear", 1, 0): grad}, set())
    # Positive control: the resolvable output is covered...
    assert report.coverage["linear:1"] == "covered"
    # ...but the unresolvable label is an internal inconsistency, not an
    # exclusion, and no ratio tolerance can absorb it.
    assert report.coverage["ghost:1"] == "unresolved_output_label"
    assert report.unresolved_output_label_count == 1
    assert not report.overall_passed


def test_compare_counts_module_less_layers_diagnostically() -> None:
    """Module-less layers are tracked separately from module-call coverage."""

    grad = torch.ones(1)
    trace = SyntheticTrace(
        [_synthetic_call("linear", 1, ["linear_out"])],
        {
            "linear_out": _synthetic_layer("linear_out", grad, ["linear:1"]),
            "top": _synthetic_layer("top", grad, []),
        },
    )
    report = _compare_module_output_grads(trace, {("linear", 1, 0): grad}, set())
    assert report.skipped_module_less_count == 1
    assert report.overall_passed


def test_per_module_output_oracle_basic() -> None:
    """TinyMLP passes the PATH E oracle with required coverage."""

    torch.manual_seed(0)
    report = _run_public_layer_grad_validation(
        TinyMLP(),
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    _assert_acceptance(report)


def test_validate_backward_pass_validate_layer_grads_public_flag() -> None:
    """The public backward validator can include the PATH E oracle."""

    torch.manual_seed(0)
    assert backward_validation.validate_backward_pass(
        TinyMLP(),
        torch.randn(2, 3),
        random_seed=42,
        validate_layer_grads=True,
    )


def test_zero_parameter_grads_fail_independently_of_layer_grad_flag() -> None:
    """An unverifiable parameter-gradient census never reports success.

    ``validate_layer_grads`` selects how much EVIDENCE is gathered; it must not
    select the VERDICT. Before this was fixed the two settings returned opposite
    booleans for the same model, same input and same warning text.
    """

    class DetachedParameterModel(nn.Module):
        """Model declaring a parameter disconnected from its output."""

        def __init__(self) -> None:
            """Initialize the deliberately unused parameter."""

            super().__init__()
            self.weight = nn.Parameter(torch.ones(3))
            self.relu = nn.ReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Return an output that never consumes ``weight``."""

            return self.relu(inputs * 2.0)

    model = DetachedParameterModel()
    x = torch.randn(2, 3)
    verdicts = []
    for validate_layer_grads in (False, True):
        with pytest.warns(RuntimeWarning, match="zero parameter gradients"):
            verdicts.append(
                backward_validation.validate_backward_pass(
                    model,
                    x,
                    random_seed=42,
                    validate_layer_grads=validate_layer_grads,
                )
            )
    assert verdicts == [False, False]


@pytest.mark.smoke
def test_layer_grad_default_runs_captured_grad_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default public validator runs the captured-gradient oracle."""

    calls = 0

    def count_comparison(*args: Any, **kwargs: Any) -> LayerGradReport:
        """Count and delegate captured-gradient comparisons."""

        nonlocal calls
        calls += 1
        return _compare_module_output_grads(*args, **kwargs)

    monkeypatch.setattr(
        "torchlens.validation._layer_grad_report._compare_module_output_grads",
        count_comparison,
    )
    assert backward_validation.validate_backward_pass(TinyMLP(), torch.randn(2, 3), random_seed=42)
    assert calls == 1


def test_nested_module_parent_and_child_outputs_are_covered() -> None:
    """Nested child and parent module calls both appear in PATH E coverage."""

    model = nn.Sequential(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), nn.Linear(4, 2))
    report = _run_public_layer_grad_validation(
        model,
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    assert report.coverage["0.1:1"] == "covered"
    assert report.coverage["0:1"] == "covered"
    _assert_acceptance(report)


def test_identity_output_modules_are_skipped() -> None:
    """Identity-output modules are skipped instead of falsely covered."""

    report = _run_public_layer_grad_validation(
        IdentityWrapper(),
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    assert report.coverage["identity:1"] == "skipped_identity_output"
    assert report.skipped_identity_output_count == 1
    assert report.overall_passed


def test_near_identity_output_modules_are_covered() -> None:
    """Numerically close outputs are audited unless they are true identity."""

    report = _run_public_layer_grad_validation(
        NearIdentityWrapper(),
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )

    assert report.coverage["near:1"] == "covered"
    _assert_acceptance(report)


def test_multi_output_module_grad_comparison_checks_all_outputs() -> None:
    """A mismatch on a non-first module output is reported."""

    grad = torch.ones(2, 3)
    trace = SyntheticTrace(
        [_synthetic_call("pair", 1, ["pair_a", "pair_b"])],
        {
            "pair_a": _synthetic_layer("pair_a", grad),
            "pair_b": _synthetic_layer("pair_b", grad * 999),
        },
    )
    report = _compare_module_output_grads(
        trace,
        {("pair", 1, 0): grad, ("pair", 1, 1): grad},
        set(),
    )

    assert report.coverage["pair:1[0]"] == "covered"
    assert report.coverage["pair:1[1]"] == "mismatched"
    assert report.mismatched_labels == ("pair:1[1]",)
    assert not report.overall_passed


def test_weight_tied_module_call_indices_are_separate() -> None:
    """Repeated calls to the same module address are compared separately."""

    class Tied(nn.Module):
        """Use one module instance twice."""

        def __init__(self) -> None:
            """Initialize shared layer."""

            super().__init__()
            self.shared = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the shared layer twice."""

            return self.shared(self.shared(x))

    report = _run_public_layer_grad_validation(
        Tied(),
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    assert report.coverage["shared:1"] == "covered"
    assert report.coverage["shared:2"] == "covered"
    _assert_acceptance(report)


def test_oracle_rnn_3_step() -> None:
    """Three-step RNN fixture passes the PATH E oracle."""

    report = _run_public_layer_grad_validation(
        TinyRNN(),
        torch.randn(2, 3, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    _assert_acceptance(report)


@pytest.mark.slow
def test_oracle_resnet50_eval() -> None:
    """ResNet-style eval fixture passes the PATH E oracle."""

    try:
        from torchvision.models import resnet50
    except Exception:
        model = TinyResNet().eval()
        x = torch.randn(2, 4)
    else:
        model = resnet50(weights=None).eval()
        x = torch.randn(1, 3, 32, 32)
    report = _run_public_layer_grad_validation(
        model,
        x,
        {},
        lambda output: output.float().sum(),
        atol=1e-4,
        rtol=1e-3,
        random_seed=42,
        # Metadata invariants are OFF here only to hold coverage exactly where
        # the deleted private copy had it. The shipped path DOES run them, and
        # on this model they fail for a PRE-EXISTING backward-metadata capture
        # bug that has nothing to do with the layer-grad oracle -- pinned by
        # ``test_shipped_backward_path_resnet50_metadata_invariant_is_broken``.
        validate_metadata=False,
    )
    _assert_acceptance(report)


@pytest.mark.slow
def test_shipped_backward_path_resnet50_metadata_invariant_is_broken() -> None:
    """Pin a PRE-EXISTING backward-metadata bug the shipped path trips.

    Surfaced by repointing the layer-grad oracle at the shipped
    ``validate_backward_pass``: a real ResNet backward capture leaves a layer
    whose ``grad_fn_handle`` has no reciprocal GradFn backpointer. Reproduced
    unchanged at base commit ``e7f036fe``, so this is a capture bug in the
    backward backend, NOT a validation defect -- the invariant is doing its job.
    Delete this test (and the ``validate_metadata=False`` opt-outs above) once
    the capture bug is fixed.
    """

    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet50(weights=None).eval()
    with pytest.raises(MetadataInvariantError, match="missing its GradFn backpointer"):
        backward_validation.validate_backward_pass(
            model,
            torch.randn(1, 3, 32, 32),
            random_seed=42,
            validate_metadata=True,
            validate_layer_grads=False,
        )


@pytest.mark.slow
def test_oracle_gpt2_small_forward_backward() -> None:
    """GPT-2-small-style fixture passes the PATH E oracle when available."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_positions=8,
        n_ctx=8,
        vocab_size=32,
    )
    model = transformers.GPT2Model(config).eval()
    input_ids = torch.randint(0, 32, (1, 4))

    def gpt2_loss(output: Any) -> torch.Tensor:
        """Return a scalar GPT-2 loss from stock or reconstructed output."""

        if isinstance(output, list):
            return output[0].float().sum()
        return output.last_hidden_state.float().sum()

    report = _run_public_layer_grad_validation(
        model,
        input_ids,
        {},
        gpt2_loss,
        atol=1e-4,
        rtol=1e-3,
        random_seed=42,
        # Same PRE-EXISTING backward-metadata capture bug as the ResNet fixture.
        validate_metadata=False,
    )
    _assert_acceptance(report)


def test_per_operation_oracle_deferred_per_ad_50() -> None:
    """Shipping code and tests do not contain PATH B implementation symbols."""

    repo = Path(__file__).resolve().parents[1]
    deleted_identifiers = [
        "_Stock" + "GradCaptureMode",
        "align" + "_stock_to_candidate",
    ]
    for identifier in deleted_identifiers:
        result = subprocess.run(
            ["git", "grep", "-w", identifier, "--", "torchlens", "tests"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stdout
    stock_side_result = subprocess.run(
        ["git", "grep", "-w", "_normalize" + "_func_name", "--", "torchlens/validation"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert stock_side_result.returncode == 1, stock_side_result.stdout


def test_path_e_module_exports_expected_surface() -> None:
    """The PATH E helper and report modules expose the expected names."""

    import torchlens.validation._stock_layer_grads as stock_module

    assert hasattr(stock_module, "_StockModuleGradCollector")
    assert hasattr(stock_module, "_first_leaf_tensor")
    assert hasattr(stock_module, "_stock_layer_grads")
    assert LayerGradReport(
        mode="module_output",
        overall_passed=True,
        coverage={},
        covered_count=1,
        skipped_no_tensor_output_count=0,
        unresolved_output_label_count=0,
        skipped_module_less_count=0,
        skipped_no_grad_count=0,
        skipped_identity_output_count=0,
        skipped_root_module_count=0,
        mismatched_count=0,
        unexpected_count=0,
        candidate_grad_count=1,
        atol=1e-5,
        rtol=1e-4,
    )
