"""Public split contracts needed by independent federated executors."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, prepare_graph
from torchlens.split.graph_profile import dtype_bytes

pytestmark = pytest.mark.usefixtures("fp64_functional")


def test_returned_compute_values_remain_partitionable() -> None:
    """A real output producer reused by downstream calls is still a compute node."""
    graph = prepare_graph(
        nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2)), torch.ones(2, 4)
    )
    node = graph.trace_graph.compute_nodes[0]
    nodes = tuple(
        replace(item, is_output=True) if item is node else item for item in graph.trace_graph.nodes
    )
    graph.trace_graph = replace(
        graph.trace_graph,
        nodes=nodes,
        output_node_ids=(*graph.trace_graph.output_node_ids, node.canonical_id),
    )
    assert node.canonical_id in {item.canonical_id for item in graph.trace_graph.compute_nodes}
    analysis = graph.analyze(GraphPartition(frozenset({node.canonical_id})))
    assert node.canonical_id in analysis.plan.boundary_node_ids
    assert not set(node.parents).intersection(analysis.plan.boundary_node_ids)


def test_tensorflow_dense_dtype_spelling_has_known_width() -> None:
    """Native TF repr strings must not turn known dense wire costs into unknowns."""
    assert dtype_bytes("<dtype: 'float32'>") == 4
    assert dtype_bytes("<dtype: 'float64'>") == 8
    assert dtype_bytes("<dtype: 'resource'>") is None


def test_native_boundary_only_gradients_and_phase_hooks() -> None:
    """Calibration VJPs use native root/state handling and leave parameter grads alone."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
    graph = prepare_graph(
        model, torch.ones(2, 4), features=SplitFeatures(training=True, retain_trace=False)
    )
    partition = graph.partition_candidates()[1]
    runtime = graph.materialize(graph.analyze(partition))
    boundary = runtime.run_training_prefix(torch.randn(2, 4))
    phases = []

    @contextmanager
    def timer(phase: str) -> Iterator[None]:
        phases.append(phase)
        yield

    loss, gradients = runtime.train_suffix(
        boundary,
        torch.zeros(2, 2),
        loss_fn=nn.functional.mse_loss,
        boundary_gradients_only=True,
        phase_timer=timer,
    )
    assert torch.isfinite(loss)
    assert gradients
    assert all(parameter.grad is None for parameter in model.parameters())
    assert phases == ["suffix_forward", "suffix_loss", "suffix_backward"]
    runtime.backward_prefix(boundary, gradients)
    assert model[0].weight.grad is not None
    assert model[2].weight.grad is None


def test_transported_source_buffers_bind_to_server_owned_state() -> None:
    """Independent executors retain two real BatchNorm updates across wire copies."""
    import copy

    from torchlens.split import ReplayBoundary

    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    client, oracle = copy.deepcopy(model), copy.deepcopy(model)
    features = SplitFeatures(training=True, retain_trace=False)
    graph = prepare_graph(model, torch.ones(2, 4), features=features)
    other = prepare_graph(client, torch.ones(2, 4), features=features)
    partition = graph.partition_candidates()[1]
    server = graph.materialize(graph.analyze(partition))
    prefix = other.materialize(other.analyze(partition))
    for _ in range(2):
        inputs = torch.randn(2, 4)
        oracle(inputs)
        boundary = prefix.run_training_prefix(inputs)
        remote = ReplayBoundary(
            backend=boundary.backend,
            tensors={key: value.detach().clone() for key, value in boundary.tensors.items()},
            spec=boundary.spec,
            metadata=boundary.metadata,
        )
        local = server.bind_boundary_state(remote)
        server.train_suffix(local, torch.zeros(2, 2), loss_fn=nn.functional.mse_loss)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, dict(oracle.named_buffers())[name])


def test_paddle_nested_outputs_keep_paths_and_reused_producers() -> None:
    """Native Paddle output containers survive a split through a returned intermediate."""
    paddle = pytest.importorskip("paddle")

    class Model(paddle.nn.Layer):
        def forward(self, inputs):
            hidden = inputs * 2
            return {
                "logits": paddle.nn.functional.relu(hidden),
                "statistics": {"mean": hidden.mean()},
            }

    model = Model()
    inputs = paddle.ones([2, 4], dtype="float64")
    graph = prepare_graph(model, inputs, backend="paddle", features=SplitFeatures(batch_axes={}))
    partition = graph.partition_candidates()[1]
    runtime = graph.materialize(graph.analyze(partition))
    actual = runtime.replay(inputs)
    expected = model(inputs)
    assert set(actual) == set(expected)
    for left, right in (
        (actual["logits"], expected["logits"]),
        (actual["statistics"]["mean"], expected["statistics"]["mean"]),
    ):
        assert bool(paddle.allclose(left, right).item())
