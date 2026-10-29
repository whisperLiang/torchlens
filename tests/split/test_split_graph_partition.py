"""Explicit DAG partition legality, frontier and replay contracts."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest
import torch
from test_split_planner import _node
from torch import nn
from v2_helpers import (
    plan_fixture,
    prepare_fixture,
    request_fixture,
)

from torchlens.intervention.types import CapturedArgTemplate, LiteralValue
from torchlens.split import GraphPartition, SplitFeatures
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.graph import SplitTraceGraph
from torchlens.split.placement import PlacementPlan
from torchlens.utils.display import identity


def _diamond() -> SplitTraceGraph:
    """Return an interleaved two-branch graph with input and output placeholders."""
    edges = {
        "A": ("input",),
        "B": ("A",),
        "C": ("A",),
        "D": ("B",),
        "E": ("C",),
        "F": ("D", "E", "A"),
    }
    return SplitTraceGraph(
        backend="torch",
        nodes=(
            _node("input", is_input=True),
            *(
                _node(name, parents=parents, raw_index=i)
                for i, (name, parents) in enumerate(edges.items(), 1)
            ),
            _node("output", parents=("F",), is_output=True, raw_index=7),
        ),
        input_node_ids=("input",),
        output_node_ids=("output",),
        graph_shape_hash="diamond",
        traced_batch_size=2,
    )


def test_exhaustive_small_dag_partitions() -> None:
    """Every one of 64 node sets agrees with an independent closure oracle."""
    graph = _diamond()
    ids = tuple(node.canonical_id for node in graph.compute_nodes)
    for size in range(len(ids) + 1):
        for members in combinations(ids, size):
            selected = frozenset(members)
            legal = all(
                set(node.parents) <= selected | {"input"}
                for node in graph.compute_nodes
                if node.canonical_id in selected
            )
            request = request_fixture(GraphPartition(selected))
            if not legal:
                with pytest.raises(SplitUnsupportedError) as caught:
                    plan_fixture(graph, request)
                assert caught.value.context.reason == "partition_reverse_dependency"
                continue
            plan = plan_fixture(graph, request)
            assert plan.prefix_node_ids == selected | {"input"}
            assert plan.suffix_node_ids == frozenset(graph.node_by_id) - plan.prefix_node_ids
            expected = {
                parent
                for node in graph.nodes
                if node.canonical_id in plan.suffix_node_ids
                for parent in node.parents
                if parent in plan.prefix_node_ids
            }
            assert set(plan.boundary_node_ids) == expected


def test_noncontiguous_frontier_deduplicates_consumers() -> None:
    """A shared branch source crosses once, independent of its consumer count."""
    graph = _diamond()
    partition = GraphPartition(frozenset({"A", "B", "D"}))
    plan = plan_fixture(graph, request_fixture(partition))
    assert plan.boundary_node_ids == ("A", "D")
    assert all(item.role == "activation" for item in plan.boundary_spec.values())
    assert partition.as_boundary() == GraphPartition(frozenset({"D", "B", "A"})).as_boundary()


@pytest.mark.parametrize("bad_id", ["input", "output", "missing", "raw_A"])
def test_partition_rejects_noncanonical_compute_ids(bad_id: str) -> None:
    """Sources, sinks and display aliases cannot substitute for canonical IDs."""
    with pytest.raises(SplitUnsupportedError) as caught:
        plan_fixture(_diamond(), request_fixture(GraphPartition(frozenset({bad_id}))))
    assert caught.value.context.reason == "partition_node_invalid"
    assert caught.value.context.split_point.startswith("partition:")


def test_native_multioutput_call_is_atomic() -> None:
    """A partition cannot execute only half of a multi-output call."""
    graph = _diamond()
    graph = replace(
        graph,
        nodes=tuple(
            replace(n, func_call_id=2) if n.canonical_id == "C" else n for n in graph.nodes
        ),
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        plan_fixture(graph, request_fixture(GraphPartition(frozenset({"A", "B"}))))
    assert caught.value.context.reason == "partition_call_split"


@pytest.mark.parametrize("kind", ["inplace", "out", "batchnorm", "metadata", "custom"])
def test_partition_refuses_unproven_effects(kind: str) -> None:
    """Normalized literal flags and captured mutation evidence fail closed."""
    graph = _diamond()
    node = graph.node_by_id["B"]
    if kind == "inplace":
        node = replace(
            node, args_template=CapturedArgTemplate(kwargs=(("inplace", LiteralValue(True)),))
        )
    elif kind == "out":
        node = replace(
            node, args_template=CapturedArgTemplate(kwargs=(("out", LiteralValue(object())),))
        )
    elif kind == "batchnorm":
        node = replace(
            node,
            op_type="batch_norm",
            args_template=CapturedArgTemplate(kwargs=(("training", LiteralValue(True)),)),
        )
    elif kind == "metadata":
        node = replace(node, op=SimpleNamespace(is_inplace=True))
    else:
        node = replace(node, target=lambda x: x)
    graph = replace(graph, nodes=tuple(node if n.canonical_id == "B" else n for n in graph.nodes))
    with pytest.raises(SplitUnsupportedError) as caught:
        plan_fixture(graph, request_fixture(GraphPartition(frozenset({"A", "B", "D"}))))
    assert caught.value.context.reason == (
        "partition_custom_effects_unproven"
        if kind == "custom"
        else "partition_effect_order_unproven"
    )


class MultiOutput(nn.Module):
    """Read-only aliased chunks, several consumers and nested model outputs."""

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        """Keep every chunk visible in a nested output tree."""
        a, b = torch.chunk(x, 2, dim=1)
        return {"sum": a + b, "nested": (a * 2, [b.relu(), x])}


def test_multioutput_nested_alias_replay() -> None:
    """Every native output travels with its container metadata and replays correctly."""
    x = torch.randn(2, 8)
    model = MultiOutput()
    runtime = prepare_fixture(model, x, request_fixture("through:" + "chunk_1_1"))
    ids = frozenset(
        n.canonical_id for n in runtime.trace_graph.compute_nodes if "chunk" in n.op_type
    )
    assert len(ids) == 2
    cut = runtime.repartition(GraphPartition(ids))
    assert len(cut.plan.boundary_node_ids) == 3  # two chunks and direct input output
    torch.testing.assert_close(cut.replay(x), model(x))
    with pytest.raises(SplitUnsupportedError, match="all outputs"):
        runtime.repartition(GraphPartition(frozenset({next(iter(ids))})))


def test_batchnorm_partition_preserves_ordered_training_state() -> None:
    """BatchNorm preserves eval outputs and ordered training gradients and buffers."""
    x = torch.randn(2, 4)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU()).eval()
    from copy import deepcopy

    reference = deepcopy(model)
    runtime = prepare_fixture(model, x, request_fixture("through:" + "linear"))
    cut = runtime.repartition(GraphPartition(frozenset()))
    torch.testing.assert_close(cut.replay(x), model(x))
    model.train()
    training = prepare_fixture(
        model, x, request_fixture("through:" + "linear", features=SplitFeatures(training=True))
    )
    cut = training.repartition(GraphPartition(frozenset()))
    reference.train()
    actual = cut.run_suffix(cut.run_training_prefix(x))
    expected = reference(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    for a, b in zip(model.buffers(), reference.buffers(), strict=True):
        torch.testing.assert_close(a, b)


def test_random_partition_preserves_rng() -> None:
    """Per-call RNG restoration survives scheduling branches in different segments."""

    class RandomBranches(nn.Module):
        """Two stochastic branches around a deterministic prefix selection."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Sum independently sampled captures with the deterministic branch."""
            a = torch.rand_like(x)
            b = x.sin()
            return a + b + torch.rand_like(x)

    x = torch.randn(2, 4)
    runtime = prepare_fixture(RandomBranches(), x, request_fixture("through:" + "sin"))
    cut = runtime.repartition(
        GraphPartition(
            frozenset(
                n.canonical_id for n in runtime.trace_graph.compute_nodes if n.op_type == "sin"
            )
        )
    )
    rng = torch.random.get_rng_state().clone()
    torch.testing.assert_close(cut.replay(x), runtime.replay(x))
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_shared_trainable_cross_device_partition_refuses_without_allocation() -> None:
    """Tied weights require a gradient merge contract before heterogeneous training."""
    layer = nn.Linear(4, 4)
    runtime = prepare_fixture(
        nn.Sequential(layer, nn.ReLU(), layer),
        torch.ones(2, 4),
        request_fixture("through:" + "1", features=SplitFeatures(training=True)),
    )
    selected = frozenset(
        node.canonical_id
        for node in runtime.trace_graph.compute_nodes
        if node.canonical_id in runtime.plan.prefix_node_ids
    )
    request = replace(
        runtime.request,
        partition=GraphPartition(selected),
        placement=PlacementPlan.across("cpu", "cuda:0"),
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        plan_fixture(runtime.trace_graph, request)
    assert caught.value.context.reason == "partition_shared_trainable_cross_device"
    assert plan_fixture(runtime.trace_graph, replace(request, placement=PlacementPlan.on("cpu")))


def test_exact_internal_identity_is_safe_but_namesakes_refuse() -> None:
    """Only TorchLens's audited no-op placeholder gains the custom-target exemption."""
    graph = _diamond()
    graph = replace(
        graph,
        nodes=tuple(
            replace(n, target=identity) if n.canonical_id == "B" else n for n in graph.nodes
        ),
    )
    request = request_fixture(GraphPartition(frozenset({"A", "B", "D"})))
    assert plan_fixture(graph, request)

    def namesake(value: torch.Tensor) -> torch.Tensor:
        """An arbitrary callable cannot inherit the known no-op's certification."""
        return value

    namesake.__name__ = "identity"
    graph = replace(
        graph,
        nodes=tuple(
            replace(n, target=namesake) if n.canonical_id == "B" else n for n in graph.nodes
        ),
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        plan_fixture(graph, request)
    assert caught.value.context.reason == "partition_custom_effects_unproven"
