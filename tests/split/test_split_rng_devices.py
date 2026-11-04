"""Placed stochastic calls preserve the captured generator's device semantics."""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, PlacementPlan, SplitFeatures, prepare_graph


class Stochastic(nn.Module):
    """Exercise both an activation mask and an independent random factory."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Keep the official native random operators in the reference forward."""
        value = self.dropout(self.linear(x))
        return (value * torch.rand_like(value)).tanh()


@pytest.mark.parametrize("source,destination", [("cpu", "cuda"), ("cuda", "cpu")])
@pytest.mark.parametrize("retain_trace", [False, True])
def test_stochastic_training_preserves_masks_values_and_gradients_across_devices(
    source: str, destination: str, retain_trace: bool
) -> None:
    """Independent native execution agrees across B=1/2/4/8 and A/B/A recuts."""
    if not torch.cuda.is_available():
        pytest.skip("Real CPU/CUDA generators and transport are required")
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "benchmarks/split_graph_partition.py"),
        run_name="rng_reference",
    )
    model = Stochastic().to(source).double().train()
    reference = deepcopy(model)
    graph = prepare_graph(
        model,
        torch.randn(1, 4, dtype=torch.float64, device=source),
        features=SplitFeatures(training=True, retain_trace=retain_trace),
        placement=PlacementPlan.across(source, destination),
    )
    assert graph.batch_validation["status"] == "passed", graph.batch_validation["reason"]
    nodes = graph.trace_graph.compute_nodes
    first = GraphPartition(frozenset({nodes[0].canonical_id}))
    full = GraphPartition(frozenset(node.canonical_id for node in nodes))
    runtime = graph.materialize(graph.analyze(first))
    actual_opt = namespace["make_optimizer"](namespace["effective_parameters"](runtime).values())
    expected_opt = namespace["make_optimizer"](reference.parameters())
    for batch, partition in zip((1, 2, 4, 8), (first, full, first, first), strict=True):
        runtime = runtime.repartition(partition)
        errors = namespace["training_step"](
            runtime,
            reference,
            torch.randn(batch, 4, dtype=torch.float64, device=source, requires_grad=True),
            actual_opt,
            expected_opt,
        )
        assert errors["forward_error"] < 1e-10
        assert errors["input_gradient_error"] < 1e-10
        assert errors["optimizer_state_error"] < 1e-10
