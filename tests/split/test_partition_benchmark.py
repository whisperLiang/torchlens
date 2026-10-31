"""The isolated acceptance driver must never turn incomplete evidence into success."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, prepare_graph

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "benchmarks" / "split_graph_partition.py"


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
    actual = namespace["make_optimizer"](
        "adam", namespace["effective_parameters"](runtime).values()
    )
    expected = namespace["make_optimizer"]("adam", reference.parameters())
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
    actual = namespace["make_optimizer"]("sgd", namespace["effective_parameters"](runtime).values())
    expected = namespace["make_optimizer"]("sgd", reference.parameters())
    errors = namespace["training_step"](
        runtime, reference, torch.randn(2, 4, requires_grad=True), actual, expected
    )
    assert errors["connected_boundary_gradient_error"] < 1e-4


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
        command.append("--strict")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert output.is_file(), completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == status
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
