"""A narrow native anchor initialization proof must preserve out= safety."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, SplitRequest, after, percent, prepare
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.planner import plan_split
from torchlens.split.shape_program import DimExpr


class AnchorInitialization(nn.Module):
    """Keep a native new_full/arange(out=...) pair in an ordinary trained model."""

    def __init__(self) -> None:
        """Construct parameters before and after the constant anchor calculation."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.cut = nn.Tanh()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize anchor coordinates using the official YOLOv8 call pattern."""

        value = self.cut(self.left(x))
        anchors = torch.arange(4, out=value.new_full((4,), 0))
        return self.head(value + anchors)


@pytest.mark.parametrize("retain_trace", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_anchor_initialization_preserves_training_and_recuts(
    retain_trace: bool, device: str
) -> None:
    """Compare outputs, input/parameter gradients and momentum updates at four batches."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for native out= initialization regression coverage")
    torch.manual_seed(428)
    model = AnchorInitialization().to(device)
    reference = deepcopy(model)
    runtime = prepare(
        model,
        torch.randn(1, 4, device=device),
        SplitRequest(
            after("cut"), features=SplitFeatures(training=True, retain_trace=retain_trace)
        ),
    )
    graph = runtime.trace_graph
    initial = GraphPartition(
        frozenset(
            node.canonical_id
            for node in graph.compute_nodes
            if node.canonical_id in runtime.plan.prefix_node_ids
        )
    )
    entire = GraphPartition(frozenset(node.canonical_id for node in graph.compute_nodes))
    runtime = runtime.partition(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    for index, batch in enumerate((1, 2, 4, 8)):
        assert len(runtime.plan.ordered_mutation_node_ids) == 1
        x = torch.randn(batch, 4, device=device, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        target = torch.randn(batch, 2, device=device)
        with torch.no_grad():
            torch.testing.assert_close(runtime.replay(x), reference(expected_x))
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        expected_loss = nn.functional.mse_loss(reference(expected_x), target)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, target)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, expected_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(optimizer.state[actual], expected_optimizer.state[expected])
        runtime = runtime.partition(entire if index % 2 == 0 else initial)
        assert runtime.trace_graph is graph


@pytest.mark.parametrize("destination", ["input", "view", "resize", "generic_out"])
def test_uncertified_out_writes_stay_refused(destination: str) -> None:
    """The anchor exception must not admit source/view/resize or other out= calls."""

    class UnprovenOut(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Exercise one unsupported destination or write target."""

            if destination == "input":
                return torch.arange(4, out=x)
            if destination == "view":
                out = x.new_full((2, 2), 0).view(4)
            elif destination == "resize":
                out = x.new_full((1,), 0)
            else:
                out = x.new_full((4,), 0)
            if destination == "generic_out":
                return torch.add(x, 1, out=out)
            return torch.arange(4, out=out)

    runtime = prepare(UnprovenOut(), torch.ones(4), SplitRequest(percent(50)))
    partition = GraphPartition(
        frozenset(node.canonical_id for node in runtime.trace_graph.compute_nodes)
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.partition(partition)
    assert caught.value.context.reason == "partition_effect_order_unproven"


def test_anchor_destination_cannot_cross_the_boundary() -> None:
    """The allocation and write must remain in the same segment."""

    runtime = prepare(AnchorInitialization(), torch.randn(2, 4), SplitRequest(after("cut")))
    selected: set[str] = set()
    for node in runtime.trace_graph.compute_nodes:
        selected.add(node.canonical_id)
        if node.op_type == "newfull":
            break
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.partition(GraphPartition(frozenset(selected)))
    assert caught.value.context.reason == "partition_effect_order_unproven"


def test_anchor_initialization_keeps_the_complete_effect_order_barrier() -> None:
    """An independent computation cannot move across the certified out= write."""

    class Interleaved(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Put an independent read between fresh allocation and initialization."""

            destination = x.new_full((4,), 0)
            independent = x * 2
            anchors = torch.arange(4, out=destination)
            return independent + anchors

    runtime = prepare(Interleaved(), torch.randn(2, 4), SplitRequest(percent(50)))
    selected = frozenset(
        node.canonical_id
        for node in runtime.trace_graph.compute_nodes
        if node.op_type in {"newfull", "arange"}
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.partition(GraphPartition(selected))
    assert caught.value.context.reason == "partition_effect_order_unproven"


def test_anchor_requires_matching_symbolic_destination_extents() -> None:
    """Matching B=1 metadata alone cannot certify a dynamically resizing destination."""

    runtime = prepare(AnchorInitialization(), torch.randn(1, 4), SplitRequest(after("cut")))
    graph = runtime.trace_graph
    allocation = next(node for node in graph.compute_nodes if node.op_type == "newfull")
    program = graph.shape_program
    shapes = dict(program.value_shapes)
    shapes[allocation.canonical_id] = replace(
        shapes[allocation.canonical_id],
        dims=(DimExpr("mul", args=(DimExpr.const(4), DimExpr.symbol(program.batch_symbol))),),
    )
    changed = replace(graph, shape_program=replace(program, value_shapes=shapes))
    request = replace(runtime.request, point=GraphPartition(frozenset()))
    with pytest.raises(SplitUnsupportedError) as caught:
        plan_split(changed, request)
    assert caught.value.context.reason == "partition_effect_order_unproven"
