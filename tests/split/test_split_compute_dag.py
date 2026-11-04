"""Call-level DAG projection, native partition conversion and execution contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from graphlib import TopologicalSorter
from itertools import combinations
from unittest.mock import patch

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, SplitGraph, prepare_graph
from torchlens.split.errors import SplitUnsupportedError


class Branches(nn.Module):
    """Interleave independent branches before joining their outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retain a residual path and two independent compute branches."""
        left = x.sin()
        right = x.cos()
        return left.relu() + right


@pytest.fixture
def branch_graph() -> SplitGraph:
    """Capture one small diamond for partition and projection checks."""
    return prepare_graph(Branches(), torch.ones(2, 4), features=SplitFeatures(batch_axes={}))


@pytest.mark.parametrize("model", [nn.Sequential(nn.Linear(4, 4), nn.ReLU()), Branches()])
def test_projection_reuses_native_calls_and_preserves_directed_data(model: nn.Module) -> None:
    """Every compute value retains its producer, consumers and native call identity."""
    graph = prepare_graph(model, torch.ones(2, 4), features=SplitFeatures(batch_axes={}))
    dag = graph.compute_dag()
    native = graph.trace_graph
    assert dag.graph_identity == graph.graph_identity
    assert dag.nodes == native.replay_calls
    assert all(a is b for a, b in zip(dag.nodes, native.replay_calls, strict=True))
    assert all(
        native.node_by_id[v].target is not None for c in dag.nodes for v in c.output_node_ids
    )
    owners = {v: c.call_id for c in dag.nodes for v in c.output_node_ids}
    expected = {
        (owners[p], call.call_id, p)
        for call in dag.nodes
        for value in call.output_node_ids
        for p in native.node_by_id[value].parents
        if p in owners and owners[p] != call.call_id
    }
    assert set(dag.data_edges) == expected
    assert len(dag.data_edges) == len(expected)
    assert not dag.effect_edges
    assert not dag.must_colocate


def test_residual_and_multiple_input_branches_keep_all_communications() -> None:
    """A shared input and independent branch outputs do not become serial dependencies."""

    class MultiInput(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Expose a residual output and a separate branch result."""
            left, middle, right = x.sin(), y.cos(), x.tanh()
            return left + middle + x, right

    graph = prepare_graph(
        MultiInput(), (torch.ones(2, 4), torch.zeros(2, 4)), features=SplitFeatures(batch_axes={})
    )
    dag = graph.compute_dag()
    types = {
        c.call_id: graph.trace_graph.node_by_id[c.output_node_ids[0]].op_type for c in dag.nodes
    }
    assert len(dag.nodes) == 5
    assert len(dag.data_edges) == 3
    assert not any(
        types[a] in {"sin", "cos", "tanh"} and types[b] in {"sin", "cos", "tanh"}
        for a, b, _ in dag.data_edges
    )
    partition = dag.to_partition(c.call_id for c in dag.nodes if types[c.call_id] == "tanh")
    runtime = graph.materialize(graph.analyze(partition))
    torch.testing.assert_close(
        runtime.replay(torch.ones(2, 4), torch.zeros(2, 4)),
        MultiInput()(torch.ones(2, 4), torch.zeros(2, 4)),
    )


def test_multioutput_call_is_atomic_and_keeps_each_tensor_value() -> None:
    """Two chunk outputs share one native call and carry separate communication IDs."""

    class Chunks(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Consume both outputs, including twice through one native add."""
            left, right = x.chunk(2, dim=1)
            return left + right, left.sin()

    x = torch.arange(8.0).reshape(2, 4)
    graph = prepare_graph(Chunks(), x, features=SplitFeatures(batch_axes={}))
    dag = graph.compute_dag()
    chunk = next(c for c in dag.nodes if len(c.output_node_ids) == 2)
    assert len(dag.nodes) == 3
    assert {v for a, _, v in dag.data_edges if a == chunk.call_id} == set(chunk.output_node_ids)
    partition = dag.to_partition([chunk.call_id, chunk.call_id])
    assert partition == GraphPartition(frozenset(chunk.output_node_ids))
    torch.testing.assert_close(graph.materialize(graph.analyze(partition)).replay(x), Chunks()(x))
    profile = graph.graph_profile()
    rows = [n for n in profile.nodes if n.call_id == chunk.call_id]
    assert len(rows) == 2
    assert rows[1].forward_flops == rows[1].backward_flops == 0
    with pytest.raises(ValueError, match="Call ID"):
        dag.to_partition([chunk.output_node_ids[0]])


def test_sources_are_excluded_but_real_state_reads_remain() -> None:
    """Parameter/buffer provenance stays available without becoming compute vertices."""

    class StateReads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.register_buffer("offset", torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Perform real computation on both state families."""
            return x * self.weight.sin() + self.offset.cos() + torch.ones_like(x)

    graph = prepare_graph(StateReads(), torch.zeros(2, 4), features=SplitFeatures(batch_axes={}))
    # Also exercise explicit source-only rows, which some backends retain.
    native = graph.trace_graph
    example = native.compute_nodes[0]
    sources = tuple(
        replace(
            example,
            canonical_id=name,
            label=name,
            raw_label=name,
            target=None,
            parents=(),
            func_call_id=None,
            is_param_source=name == "parameter",
        )
        for name in ("parameter", "constant")
    )
    graph.trace_graph = replace(native, nodes=(*native.nodes, *sources))
    dag = graph.compute_dag()
    ids = {v for c in dag.nodes for v in c.output_node_ids}
    assert not ids & {"parameter", "constant", *native.input_node_ids, *native.output_node_ids}
    assert not any(native.node_by_id[v].is_buffer for v in ids)
    assert {native.node_by_id[v].op_type for v in ids} >= {"sin", "cos", "oneslike"}
    assert graph.graph_profile().parameters


def test_raw_war_waw_dependencies_and_local_alias_constraints() -> None:
    """Storage dependencies preserve native order without inventing tensor transfers."""

    class Mutations(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Read, write, read and write one local allocation through an alias."""
            fresh = x + 1
            before = fresh.square()
            alias = fresh.view_as(x)
            alias.add_(1)
            after = alias.clone()
            alias.add_(2)
            return before + after + alias

    graph = prepare_graph(
        Mutations(), torch.ones(2, 4), features=SplitFeatures(training=True, batch_axes={})
    )
    dag = graph.compute_dag()
    native = graph.trace_graph
    by_type: dict[str, list[str]] = {}
    for c in dag.nodes:
        n = native.node_by_id[c.output_node_ids[0]]
        by_type.setdefault(getattr(n.target, "__name__", ""), []).append(c.call_id)
    first, second = by_type["add_"]
    square, clone = by_type["square"][0], by_type["clone"][0]
    assert {(square, first), (first, clone), (first, second), (clone, second)} <= set(
        dag.effect_edges
    )
    assert not any(a == square and b == first for a, b, _ in dag.data_edges)
    assert any(first in group and by_type["view_as"][0] in group for group in dag.must_colocate)
    assert all(len(group) >= 2 for group in dag.must_colocate)
    assert not set(dag.effect_edges) - {
        (native.replay_call_by_output_id[a].call_id, native.replay_call_by_output_id[b].call_id)
        for b, parents in native._partition_effect_index.predecessors.items()
        for a in parents
    }


def test_shared_mutable_state_keeps_effect_order_and_native_refusal() -> None:
    """A DAG's successful construction does not license reversed BatchNorm writes."""

    class SharedNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Use one mutable state on otherwise independent branches."""
            return self.norm(x.sin()) + self.norm(x.cos())

    graph = prepare_graph(
        SharedNorm().train(), torch.ones(2, 4), features=SplitFeatures(training=True, batch_axes={})
    )
    dag = graph.compute_dag()
    norms = [
        c
        for c in dag.nodes
        if graph.trace_graph.node_by_id[c.output_node_ids[0]].op_type == "batchnorm"
    ]
    assert len(norms) == 2
    assert (norms[0].call_id, norms[1].call_id) in dag.effect_edges
    right = next(
        c for c in dag.nodes if graph.trace_graph.node_by_id[c.output_node_ids[0]].op_type == "cos"
    )
    partition = dag.to_partition([right.call_id, norms[1].call_id])
    with pytest.raises(SplitUnsupportedError) as caught:
        graph.analyze(partition)
    assert caught.value.context.reason in {
        "partition_effect_order_unproven",
        "partition_reverse_dependency",
    }


def test_small_dag_subsets_agree_with_native_analysis(branch_graph: SplitGraph) -> None:
    """An independent downward-closure oracle checks every subset of a small DAG."""
    dag = branch_graph.compute_dag()
    ids = tuple(c.call_id for c in dag.nodes)
    for size in range(len(ids) + 1):
        for selected in combinations(ids, size):
            prefix = set(selected)
            partition = dag.to_partition(selected)
            legal = all(b not in prefix or a in prefix for a, b, _ in dag.data_edges)
            if not legal:
                with pytest.raises(SplitUnsupportedError) as caught:
                    branch_graph.analyze(partition)
                assert caught.value.context.reason == "partition_reverse_dependency"
            else:
                analysis = branch_graph.analyze(partition)
                assert analysis.request.partition == partition
                torch.testing.assert_close(
                    branch_graph.materialize(analysis).replay(torch.ones(2, 4)),
                    Branches()(torch.ones(2, 4)),
                )


def test_projection_is_immutable_stable_acyclic_and_never_executes(
    branch_graph: SplitGraph,
) -> None:
    """Repeated projections and cost inspection reuse the capture without a forward."""
    with (
        patch.object(branch_graph.model, "forward", side_effect=AssertionError("no forward")),
        patch(
            "torchlens.split.pipeline.capture_canonical_model",
            side_effect=AssertionError("no capture"),
        ),
        patch.object(
            branch_graph.adapter, "build_segments", side_effect=AssertionError("no segments")
        ),
    ):
        dag = branch_graph.compute_dag()
        assert branch_graph.compute_dag() == dag
        assert branch_graph.graph_profile().graph_identity == dag.graph_identity
        branch_graph.analyze(dag.to_partition([dag.nodes[0].call_id]))
    dependencies = {c.call_id: set() for c in dag.nodes}
    for a, b in (*((a, b) for a, b, _ in dag.data_edges), *dag.effect_edges):
        assert a != b
        dependencies[b].add(a)
    assert len(tuple(TopologicalSorter(dependencies).static_order())) == len(dag.nodes)
    with pytest.raises(FrozenInstanceError):
        dag.nodes = ()


@pytest.mark.parametrize("selection", ["call", b"call", [None], [1]])
def test_to_partition_rejects_invalid_id_representations(
    branch_graph: SplitGraph, selection: object
) -> None:
    """Call IDs are supplied as a collection of nonempty strings."""
    with pytest.raises(TypeError):
        branch_graph.compute_dag().to_partition(selection)


def test_to_partition_rejects_unknown_and_source_ids(branch_graph: SplitGraph) -> None:
    """Conversion accepts native calls only and leaves dependency validation to analyze."""
    dag = branch_graph.compute_dag()
    for invalid in (
        "absent",
        *branch_graph.trace_graph.input_node_ids,
        *branch_graph.trace_graph.output_node_ids,
    ):
        with pytest.raises(ValueError, match="Call ID"):
            dag.to_partition([invalid])
    assert dag.to_partition(()) == GraphPartition(frozenset())
    assert dag.to_partition(c.call_id for c in reversed(dag.nodes)) == dag.to_partition(
        c.call_id for c in dag.nodes
    )
