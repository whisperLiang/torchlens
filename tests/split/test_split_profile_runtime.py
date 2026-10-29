"""Public graph costs are static and reusable across partitions."""

from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, SplitRequest, after, prepare


def test_profile_and_partition_metrics_need_no_execution() -> None:
    """Expose batch-scaled frontier costs without replaying or materializing."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    runtime = prepare(
        model,
        torch.ones(2, 4),
        SplitRequest(after("1"), features=SplitFeatures(training=True, retain_trace=False)),
    )
    with (
        patch.object(model, "forward", side_effect=AssertionError("no recapture")),
        patch.object(
            runtime.adapter, "build_segments", side_effect=AssertionError("no materialize")
        ),
    ):
        profile = runtime.graph_profile(batch_size=8)
        assert profile.graph_identity == runtime.graph_identity
        assert profile.batch_size == 8
        assert sum(item.bytes for item in profile.parameters) == sum(
            param.numel() * param.element_size() for param in model.parameters()
        )
        assert {item.node_id for item in profile.nodes} == {
            node.canonical_id for node in runtime.trace_graph.compute_nodes
        }
        selected = frozenset(
            node.canonical_id
            for node in runtime.trace_graph.compute_nodes
            if node.module_path in {"0", "1"}
        )
        analysis = runtime.analyze_partition(GraphPartition(selected))
        assert analysis.metrics is not None
        assert analysis.metrics.unique_boundary_tensors == 1
        assert analysis.metrics.forward_boundary_bytes == runtime.traced_batch_size * 8 * 4


def test_unproven_batch_costs_are_unknown() -> None:
    """Cost inspection discloses a failed batch probe without guessing tensor sizes."""

    class BatchBranch(nn.Module):
        """Change the taken path across batch sizes."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return distinct compute graphs for the two witness batches."""
            return torch.sigmoid(torch.relu(x)) if x.shape[0] > 1 else torch.relu(x)

    runtime = prepare(BatchBranch(), torch.ones(2, 4), SplitRequest(after("relu")))
    assert runtime.batch_validation["status"] == "failed"
    profile = runtime.graph_profile(batch_size=8)
    assert "batch_shape:extrapolation_unproven" in profile.unknown_cost_diagnostics
    assert all(tensor.bytes is None for tensor in profile.tensors)
