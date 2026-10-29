"""Effect proofs include source ownership and transport, even without a GPU."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, PlacementPlan, prepare_graph
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.planner import plan_partition


def test_input_write_cannot_cross_a_device_boundary() -> None:
    """Transported storage cannot reproduce an observable write to the caller's input."""

    class InputWrite(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.add_(1) * 2

    graph = prepare_graph(InputWrite(), torch.ones(2, 4))
    request = replace(graph.request, placement=PlacementPlan.across("cpu", "cuda:0"))
    with pytest.raises(SplitUnsupportedError) as caught:
        plan_partition(graph.trace_graph, request)
    assert caught.value.context.reason == "partition_effect_order_unproven"
    node = graph.trace_graph.compute_nodes[0]
    prefix_write = replace(request, partition=GraphPartition(frozenset({node.canonical_id})))
    assert plan_partition(graph.trace_graph, prefix_write).ordered_mutation_node_ids
