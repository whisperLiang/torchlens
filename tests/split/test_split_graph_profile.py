"""Static graph cost metadata and deduplicated partition metrics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import pytest
import torch
from dag_helpers import call_ids, close_call_ids
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, prepare_graph
from torchlens.split.graph_profile import build_graph_profile, partition_metrics


class SharedParameters(nn.Module):
    """One projection reused by independent branches."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.alias = self.projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Join two projections consuming the same weight and input."""
        return self.projection(x) + self.alias(x * 2)


def test_static_profile_shapes_flops_state_aliases_and_no_execution() -> None:
    """Profiles never run the model and count tied state once with stable names."""
    model = SharedParameters()
    graph = prepare_graph(model, torch.ones(2, 4))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"linear_1_1:1"}), ops=frozenset({"linear"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    with patch.object(model, "forward", side_effect=AssertionError("must not execute")):
        profile = runtime.graph_profile(batch_size=8)
    assert profile.batch_size == 8
    linear = [node for node in profile.nodes if node.op_type == "linear"]
    assert len(linear) == 2
    assert all(node.forward_flops == 8 * 4 * (2 * 4 + 1) for node in linear)
    assert all(node.activation_bytes == 8 * 4 * 4 for node in linear)
    assert sum(param.bytes for param in profile.parameters) == (4 * 4 + 4) * 4
    assert len(profile.parameters) == 2
    weight = next(param for param in profile.parameters if len(param.shape) == 2)
    assert weight.aliases == ("alias.weight", "projection.weight")
    assert len(weight.consumer_ids) == 2
    assert weight.state_id == "parameter:alias.weight"
    with pytest.raises(FrozenInstanceError):
        profile.batch_size = 4
    other_graph = prepare_graph(SharedParameters(), torch.ones(2, 4))
    other_dag = other_graph.compute_dag()
    other_partition = other_dag.to_partition(
        close_call_ids(
            other_dag,
            call_ids(other_graph, modules=frozenset({"linear_1_1:1"}), ops=frozenset({"linear"})),
        )
    )
    other = other_graph.materialize(other_graph.analyze(other_partition))
    assert [p.state_id for p in profile.parameters] == [
        p.state_id for p in other.graph_profile().parameters
    ]


def test_partition_metrics_count_unique_boundary_and_parameters() -> None:
    """Costs use the value frontier rather than the number of outgoing edges."""
    graph = prepare_graph(
        SharedParameters(), torch.ones(2, 4), features=SplitFeatures(training=True)
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"linear_1_1:1"}), ops=frozenset({"linear"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    profile = runtime.graph_profile(batch_size=8)
    first = next(node for node in profile.nodes if node.op_type == "linear")
    analysis = runtime.analyze(GraphPartition(frozenset({first.node_id})))
    metrics = partition_metrics(profile, analysis.plan, training=True)
    assert metrics.unique_boundary_tensors == 2  # first projection plus shared input
    assert metrics.forward_boundary_bytes == 2 * 8 * 4 * 4
    assert metrics.parameter_bytes == 80
    assert metrics.prefix_parameter_bytes == metrics.suffix_parameter_bytes == 80
    assert metrics.backward_gradient_bytes is None
    assert metrics.forward_flops == 640  # shared engine includes multiplication and addition
    assert metrics.known_forward_flops > 0
    assert "backward_gradient_bytes:loss_path_not_observed" in metrics.unknown_cost_diagnostics


def test_unknown_dtype_shape_and_invalid_batch() -> None:
    """Unsupported static facts remain None; invalid batch inputs refuse."""
    graph = prepare_graph(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.ones(2, 4))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    graph = runtime.trace_graph
    graph = replace(
        graph, nodes=tuple(replace(n, dtype="unknown128") for n in graph.nodes), shape_program=None
    )
    profile = build_graph_profile(graph, batch_size=8)
    assert all(tensor.bytes is None and tensor.shape is None for tensor in profile.tensors)
    assert profile.unknown_cost_diagnostics
    for batch in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            build_graph_profile(graph, batch_size=batch)


def test_matmul_cost_and_multioutput_atomic_metadata() -> None:
    """Matmul cost is shape-based and each chunk advertises its whole call group."""

    class Matrix(nn.Module):
        """Chunk the final matrix result into two output values."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Return two views of the matrix product."""
            return torch.chunk(a @ b, 2, dim=-1)

    graph = prepare_graph(
        Matrix(),
        (torch.randn(2, 3, 4), torch.randn(4, 6)),
        features=SplitFeatures(batch_axes={"/args/0": 0}),
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"matmul"}), ops=frozenset({"matmul"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    profile = runtime.graph_profile(batch_size=2)
    matmul = next(node for node in profile.nodes if node.op_type == "matmul")
    assert matmul.forward_flops == 2 * (2 * 3 * 6) * 4
    chunks = [node for node in profile.nodes if node.op_type == "chunk"]
    assert len(chunks) == 2
    assert all(set(node.output_node_ids) == {item.node_id for item in chunks} for node in chunks)
    assert chunks[0].forward_flops == chunks[1].forward_flops == 0


def test_tinygrad_dtype_costs_match_native_scalar_widths() -> None:
    """Native tinygrad scalar spellings retain exact dense-byte costs."""
    tinygrad = pytest.importorskip("tinygrad")
    from torchlens.split.graph_profile import dtype_bytes

    for name in (
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ):
        native = getattr(tinygrad.dtypes, name)
        assert dtype_bytes(str(native)) == native.itemsize
    assert dtype_bytes("dtypes.future_scalar") is None
    assert dtype_bytes("dtypes.float.ptr(16)") is None
