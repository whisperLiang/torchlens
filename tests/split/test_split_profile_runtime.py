"""Public graph costs are static and reusable across partitions."""

from __future__ import annotations

from unittest.mock import patch

import torch
from dag_helpers import call_ids, close_call_ids
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, prepare_graph


def test_profile_and_partition_metrics_need_no_execution() -> None:
    """Expose batch-scaled frontier costs without replaying or materializing."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    graph = prepare_graph(
        model, torch.ones(2, 4), features=SplitFeatures(training=True, retain_trace=False)
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"1"}), ops=frozenset({"1"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
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
        analysis = runtime.analyze(GraphPartition(selected))
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

    graph = prepare_graph(BatchBranch(), torch.ones(2, 4))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    assert runtime.batch_validation["status"] == "failed"
    profile = runtime.graph_profile(batch_size=8)
    assert "batch_shape:extrapolation_unproven" in profile.unknown_cost_diagnostics
    assert all(tensor.bytes is None for tensor in profile.tensors)
