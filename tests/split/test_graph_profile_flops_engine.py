"""GraphProfile uses capture costs on metadata, including live custom registrations."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from torchlens.capture import flops
from torchlens.split import GraphPartition, SplitFeatures, prepare_graph


class FunctionModel(nn.Module):
    """Use a native callable as a real module's forward body."""

    def __init__(self, function: object) -> None:
        super().__init__()
        self.function = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_profile_reuses_capture_flops_without_tensor_payloads(compact: bool, batch: int) -> None:
    """Conv, normalization, elementwise and linear costs agree with the capture engine."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(16, 2)
    ).eval()
    graph = prepare_graph(
        model, torch.ones(2, 3, 4, 4), features=SplitFeatures(retain_trace=not compact)
    )
    with patch.object(model, "forward", side_effect=AssertionError("profile cannot replay")):
        profile = graph.graph_profile(batch_size=batch)
    for cost in profile.nodes:
        assert cost.forward_flops is not None
        name = graph.trace_graph.node_by_id[cost.node_id].cost_name
        assert cost.backward_flops == flops.compute_backward_flops(name, cost.forward_flops)
    conv = next(n for n in profile.nodes if n.op_type == "conv2d")
    assert conv.forward_flops == 2 * batch * 4 * 2 * 2 * 3 * 3 * 3 + batch * 4 * 2 * 2
    assert profile.backward_flops_estimated
    if batch == graph.traced_batch_size:
        for cost, _node in zip(profile.nodes, graph.trace_graph.compute_nodes, strict=True):
            assert cost.captured_forward_flops == cost.forward_flops


def test_registered_rule_is_used_after_profile_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration has one authority and invalidates cached profiles and analysis metrics."""
    graph = prepare_graph(FunctionModel(lambda x: x.sin()), torch.ones(2, 4))
    graph.graph_profile(batch_size=8)
    partition = GraphPartition(frozenset(n.canonical_id for n in graph.trace_graph.compute_nodes))
    graph.analyze(partition)
    monkeypatch.setattr(flops, "_CUSTOM_OP_RULES", dict(flops._CUSTOM_OP_RULES))
    flops.register_op_rule("sin", lambda shape, params, args, kwargs: 123 * args[0].shape[0])
    assert graph.graph_profile(batch_size=8).nodes[0].forward_flops == 123 * 8
    assert graph.analyze(partition).metrics.forward_flops == 123


@pytest.mark.parametrize("operation", ["matmul", "einsum", "attention", "multioutput", "unknown"])
def test_shape_only_argument_adapter(operation: str) -> None:
    """Native argument shapes support attention/einsum and deduplicate multi-output work."""
    functions = {
        "matmul": lambda x: x @ x.transpose(-1, -2),
        "einsum": lambda x: torch.einsum("bik,bjk->bij", x, x),
        "attention": lambda x: nn.functional.scaled_dot_product_attention(x, x, x),
        "multioutput": lambda x: torch.max(x, dim=-1),
        "unknown": lambda x: torch.special.gammaln(x),
    }
    graph = prepare_graph(FunctionModel(functions[operation]), torch.ones(2, 3, 4))
    profile = graph.graph_profile(batch_size=8)
    if operation == "unknown":
        assert any(n.forward_flops is None for n in profile.nodes)
        assert profile.unknown_cost_diagnostics
    else:
        assert all(n.forward_flops is not None for n in profile.nodes)
    for call in graph.trace_graph.replay_calls:
        rows = [n for n in profile.nodes if n.call_id == call.call_id]
        assert all(n.forward_flops == n.backward_flops == 0 for n in rows[1:])


def test_batch_reduction_is_recomputed_from_arguments_without_allocations() -> None:
    """A reduction's output stays fixed while its input work changes with batch."""
    graph = prepare_graph(FunctionModel(lambda x: torch.sum(x, dim=0)), torch.ones(2, 4))
    with (
        patch.object(torch, "empty", side_effect=AssertionError("no tensor allocations")),
        patch.object(torch, "tensor", side_effect=AssertionError("no tensor allocations")),
    ):
        profile = graph.graph_profile(batch_size=8)
    assert profile.nodes[0].activation_bytes == 4 * 4
    assert profile.nodes[0].forward_flops == 8 * 4
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    assert runtime.graph_profile(batch_size=8) is profile
