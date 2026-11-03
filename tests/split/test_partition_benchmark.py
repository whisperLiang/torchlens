"""The isolated acceptance driver must never turn incomplete evidence into success."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, prepare_graph

pytestmark = pytest.mark.usefixtures("fp64_functional")

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "benchmarks" / "split_graph_partition.py"


def test_sgd_factory_keeps_execution_parameters_and_updates_after_rebinding() -> None:
    """The acceptance optimizer must use plain SGD on the current execution leaves."""
    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    parameter = nn.Parameter(torch.tensor([0.2, -0.4]))
    reference = nn.Parameter(parameter.detach().clone())
    frozen = nn.Parameter(torch.ones(2), requires_grad=False)
    optimizer = namespace["make_optimizer"]([parameter, parameter, frozen])
    expected = torch.optim.SGD([reference], lr=0.01)
    assert type(optimizer) is torch.optim.SGD
    assert optimizer.param_groups[0]["params"] == [parameter]
    assert optimizer.param_groups[0]["momentum"] == 0
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert not optimizer.param_groups[0]["nesterov"]
    parameter.grad = torch.tensor([0.3, -0.1])
    reference.grad = parameter.grad.clone()
    optimizer.step()
    expected.step()
    leaf = parameter.detach().clone().requires_grad_(True)
    optimizer = namespace["make_optimizer"]([leaf])
    assert optimizer.param_groups[0]["params"][0] is leaf
    leaf.grad = torch.tensor([-0.2, 0.4])
    reference.grad = leaf.grad.clone()
    optimizer.step()
    expected.step()
    torch.testing.assert_close(leaf, reference, atol=0, rtol=0)
    assert not optimizer.state


def test_fixed_float_tolerance_rejects_out_of_budget_values() -> None:
    """SGD acceptance keeps its original fixed numerical budget."""
    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    expected = torch.zeros(2, dtype=torch.float64)
    assert namespace["finite_close"](torch.full_like(expected, 5e-9), expected) == 5e-9
    with pytest.raises(AssertionError):
        namespace["finite_close"](torch.full_like(expected, 5e-8), expected)


@pytest.mark.parametrize("case", ["integer", "boolean", "dtype", "shape", "nan", "inf", "gradient"])
def test_fixed_tolerance_keeps_non_numeric_contracts(case: str) -> None:
    """Strict comparisons also reject categorical, structural and nonfinite errors."""
    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    pairs = {
        "integer": (torch.tensor([1_000_000_001]), torch.tensor([1_000_000_000])),
        "boolean": (torch.tensor([True]), torch.tensor([False])),
        "dtype": (torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float64)),
        "shape": (torch.zeros(1), torch.zeros(2)),
        "nan": (torch.tensor([float("nan")]), torch.zeros(1)),
        "inf": (torch.tensor([float("inf")]), torch.zeros(1)),
        "gradient": (None, torch.zeros(1)),
    }
    with pytest.raises(AssertionError):
        namespace["finite_close"](*pairs[case])


def test_state_only_candidate_is_refused_without_failing_the_search() -> None:
    """A lone BN counter has no real compute frontier and cannot attest a DAG cut."""
    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    model = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2), nn.ReLU(inplace=True)).train()
    context = prepare_graph(
        model,
        torch.ones(1, 2, 3, 3),
        features=SplitFeatures(training=True, retain_trace=False),
    )
    counter = next(n for n in context.trace_graph.compute_nodes if n.op_type == "add")
    partition = GraphPartition(frozenset({counter.canonical_id}))
    analyze = context.analyze(partition)
    assert not set(analyze.plan.boundary_node_ids) & partition.prefix_node_ids
    args = SimpleNamespace(
        prefix_device="cpu",
        suffix_device="cpu",
        candidate_strategy="ancestors",
        candidate_census=True,
    )
    validate = namespace["validate_kind"]
    with patch.dict(validate.__globals__, candidates=lambda *a, **kw: [partition]):
        row = validate(context, model, model.state_dict(), (2, 3, 3), "noncontiguous_dag", args)
    assert row["status"] == "unsupported"
    assert row["attempts"][0]["reason"] == "candidate_compute_frontier_absent"
    assert row["legal_candidate_count"] == 0
    assert row["checks"]["backward"] == "not_run"


@pytest.mark.parametrize("model", ["rfdetr", "yolov8"])
def test_detection_targets_preserve_input_precision(model: str) -> None:
    """Double-precision controls need double boxes while categorical labels stay integral."""
    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    targets = namespace["training_targets"](model, torch.ones(2, 3, 8, 8, dtype=torch.float64))
    boxes = targets[0]["boxes"] if model == "rfdetr" else targets["bboxes"]
    assert boxes.dtype == torch.float64
    if model == "rfdetr":
        assert targets[0]["labels"].dtype == torch.int64
    else:
        assert targets["cls"].dtype == torch.int64
        assert targets["batch_idx"].dtype == torch.int64


def test_official_loss_factory_scope_follows_output_dtype_and_restores_ambient() -> None:
    """Native loss temporaries must use the selected oracle precision, without leaking it."""
    pytest.importorskip("ultralytics")
    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")

    class NativeStyleCriterion(nn.Module):
        def forward(self, output: torch.Tensor, targets: dict) -> tuple:
            """Match the official preprocess allocation and indexed assignment pattern."""
            result = torch.zeros(output.shape, device=output.device)
            result[torch.ones(output.shape, device=output.device, dtype=torch.bool)] = (
                output.flatten()
            )
            return result.sum().unsqueeze(0), {}

    class LossModel(nn.Module):
        def init_criterion(self) -> nn.Module:
            """Provide the native model's criterion construction interface."""
            return NativeStyleCriterion()

    original = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        loss_fn = namespace["training_objective"](
            LossModel(), SimpleNamespace(model="yolov8", model_mode="train"), {}
        )
        output = torch.ones(1, dtype=torch.float64, requires_grad=True)
        loss = loss_fn(output, {})
        assert loss.dtype == torch.float64
        assert torch.get_default_dtype() == torch.float32
        loss.backward()
        torch.testing.assert_close(output.grad, torch.ones_like(output))
    finally:
        torch.set_default_dtype(original)


def test_native_frontier_observer_uses_post_prefix_alias_values_and_gradients() -> None:
    """A prefix write may rebase an earlier view even when its own output is internal."""
    from copy import deepcopy

    class AliasedBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(4, 4, bias=False)
            self.right = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Keep the view frontier and the inplace writer on the selected branch."""
            a = self.left(x)
            alias = a.view_as(x)
            b = self.right(x)
            a.relu_()
            return (a * 2).sin() + b + alias

    model = AliasedBranch()
    with torch.no_grad():
        model.left.weight.copy_(torch.eye(4))
    reference = deepcopy(model)
    context = prepare_graph(
        model, torch.ones(1, 4), features=SplitFeatures(training=True, retain_trace=False)
    )
    selected = frozenset(
        n.canonical_id
        for n in context.trace_graph.compute_nodes
        if n.module_path == "left" or n.op_type in {"viewas", "relu", "mul", "sin"}
    )
    runtime = context.materialize(context.analyze(GraphPartition(selected)))
    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    errors = namespace["training_step"](
        runtime,
        reference,
        torch.tensor([[-1.0, 0.5, 1.0, -2.0], [-0.2, 2.0, -3.0, 4.0]], requires_grad=True),
        namespace["make_optimizer"](namespace["effective_parameters"](runtime).values()),
        namespace["make_optimizer"](reference.parameters()),
    )
    assert errors["boundary_values_checked"] > 0
    assert errors["boundary_value_error"] < 1e-10
    assert errors["connected_boundary_gradient_error"] < 1e-10


@pytest.mark.parametrize("retain_trace", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_native_observer_keeps_state_writes_and_augmented_assignment_in_order(
    retain_trace: bool,
    device: str,
) -> None:
    """BatchNorm counters and floating add_ facades must not share wrong call indexes."""
    from copy import deepcopy

    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("A real CUDA device is required")

    class ResidualNorm(nn.Module):
        """Use counter add_ writes and residual __iadd__ in one native forward."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            value = self.norm(self.proj(x))
            value += x
            return value.tanh()

    model = ResidualNorm().to(device).train()
    reference = deepcopy(model)
    graph = prepare_graph(
        model,
        torch.randn(3, 4, device=device),
        features=SplitFeatures(training=True, retain_trace=retain_trace, batch_axes={}),
    )
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    actual = namespace["make_optimizer"](namespace["effective_parameters"](runtime).values())
    expected = namespace["make_optimizer"](reference.parameters())
    nodes = graph.trace_graph.compute_nodes
    partitions = [
        GraphPartition(frozenset(n.canonical_id for n in nodes[:count]))
        for count in (0, 1, len(nodes))
    ]
    for step, partition in enumerate(partitions, 1):
        runtime = runtime.repartition(partition)
        errors = namespace["training_step"](
            runtime,
            reference,
            torch.randn(3, 4, device=device, requires_grad=True),
            actual,
            expected,
        )
        assert errors["buffer_error"] == 0
        assert (
            model.norm.num_batches_tracked.item()
            == reference.norm.num_batches_tracked.item()
            == step
        )
        assert errors["parameter_gradient_error"] < 1e-4


def test_buffer_observer_resolves_unregistered_native_tensor_attributes() -> None:
    """Capture inventories may name native attributes absent from named_buffers()."""
    from copy import deepcopy

    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")

    class Offset(nn.Module):
        """Consume a plain tensor as the official detection models do."""

        def __init__(self) -> None:
            super().__init__()
            self.offset = torch.ones(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (x + self.offset).sin()

    model = Offset()
    reference = deepcopy(model)
    graph = prepare_graph(model, torch.ones(2, 4))
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    runtime.replay(torch.ones(2, 4))
    assert namespace["buffer_error"](runtime, reference) == 0


def test_native_observer_matches_both_tensor_split_overloads() -> None:
    """Size-list and integer splits share a native facade but retain distinct calls and leaves."""
    from copy import deepcopy

    namespace = runpy.run_path(str(DRIVER), run_name="benchmark_test")

    class MixedSplits(nn.Module):
        """Interleave split-with-sizes and fixed-size native calls."""

        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(4, 6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a, b, c = self.project(x).split([2, 2, 2], dim=1)
            d, e = (a + b).split(1, dim=1)
            return c * 3 + torch.cat((d * 2, e * 4), dim=1)

    model = MixedSplits()
    reference = deepcopy(model)
    graph = prepare_graph(
        model, torch.randn(2, 4), features=SplitFeatures(training=True, retain_trace=False)
    )
    partition = GraphPartition(
        frozenset(node.canonical_id for node in graph.trace_graph.compute_nodes[:4])
    )
    runtime = graph.materialize(graph.analyze(partition))
    actual = namespace["make_optimizer"](namespace["effective_parameters"](runtime).values())
    expected = namespace["make_optimizer"](reference.parameters())
    errors = namespace["training_step"](
        runtime, reference, torch.randn(2, 4, requires_grad=True), actual, expected
    )
    assert errors["connected_boundary_gradient_error"] < 1e-4


@pytest.mark.heavy
def test_default_training_cli_runs_only_sgd_across_three_cuts(tmp_path: Path) -> None:
    """Default training must validate four SGD batches for each partition kind."""
    output = tmp_path / "sgd.json"
    code = """
import runpy
import sys
import torch
from unittest.mock import patch

driver = sys.argv[1]
sys.argv = [driver, *sys.argv[2:]]
namespace = runpy.run_path(driver, run_name='benchmark_test')
main = namespace['main']

class Branches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def branch():
            return torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1),
                torch.nn.BatchNorm2d(2), torch.nn.ReLU(inplace=True))
        self.left, self.right = branch(), branch()
    def forward(self, x):
        return self.left(x) + self.right(x)

with patch.dict(main.__globals__, load_model=lambda *a, **kw: (Branches(), (2, 3, 3), {})):
    main()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(DRIVER),
            "--model",
            "resnet18",
            "--model-mode",
            "train",
            "--strict",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["optimizer"] == {"name": "SGD", "lr": 0.01}
    assert report["recut_capture_count"] == 0
    for row in report["results"]:
        assert row["status"] == "passed"
        assert row["distinct_partitions"] == 3
        assert [(step["optimizer"], step["batch"]) for step in row["training_steps"]] == [
            ("sgd", 1),
            ("sgd", 2),
            ("sgd", 4),
            ("sgd", 8),
        ]


@pytest.mark.heavy
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("status", ["passed", "failed", "unsupported", "skipped", "not_validated"])
def test_acceptance_exit_status_and_complete_json(
    tmp_path: Path, strict: bool, status: str
) -> None:
    """Exercise the CLI in a child process with controlled validation outcomes."""
    output = tmp_path / "report.json"
    code = """
import runpy
import sys
import torch
from unittest.mock import patch

driver, status = sys.argv[1:3]
sys.argv = [driver, *sys.argv[3:]]
namespace = runpy.run_path(driver, run_name='benchmark_test')
main = namespace['main']
def load_model(name, weights, **kwargs):
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()), (4,), {}
def validate(*args):
    return {'partition_kind': args[-2], 'status': status, 'reason': 'controlled outcome'}
with patch.dict(main.__globals__, load_model=load_model, validate_kind=validate):
    main()
"""
    command = [
        sys.executable,
        "-c",
        code,
        str(DRIVER),
        status,
        "--model",
        "resnet18",
        "--output",
        str(output),
    ]
    if strict:
        command.extend(("--strict", "--deterministic"))
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert output.is_file(), completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == status
    assert report["execution_context"]["deterministic_algorithms"] is strict
    assert report["tolerance"] == {
        "atol": 1e-8,
        "rtol": 1e-7,
        "integer_boolean": "exact",
    }
    assert len(report["results"]) == 2
    assert all(row["reason"] == "controlled outcome" for row in report["results"])
    assert json.loads(completed.stdout) == report
    assert (completed.returncode == 0) == (status == "passed" or not strict and status != "failed")


@pytest.mark.heavy
def test_missing_trusted_weights_is_skipped_before_strict_exit(tmp_path: Path) -> None:
    """An absent RF-DETR checkpoint must be disclosed and fail strict acceptance."""
    output = tmp_path / "rfdetr.json"
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "--model", "rfdetr", "--strict", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert output.is_file(), completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "skipped"
    assert "trusted local checkpoint" in report["reason"]
    assert completed.returncode != 0


@pytest.mark.heavy
def test_repartition_reports_actual_state_copies_for_equivalent_cpu_aliases(tmp_path: Path) -> None:
    """Different placement spellings must not attest an unobserved device copy."""
    output = tmp_path / "repartition.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "split_repartition.py"),
            "--prefix-device",
            "cpu",
            "--suffix-device",
            "cpu:0",
            "--repeats",
            "1",
            "--memory-iterations",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["phases_separate_instrumented_run"]["device_transfer"]["n"] == 0
    assert report["device_transfer"].startswith("not_run")
