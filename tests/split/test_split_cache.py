"""Boundary cache tests."""

from __future__ import annotations

import pytest
import torch
from dag_helpers import call_ids, close_call_ids
from torch import nn

from torchlens.split import SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError


class CacheModel(nn.Module):
    """Small model for cache roundtrip tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_boundary_cache_roundtrip(tmp_path) -> None:
    """A saved boundary validates and drives suffix replay after load."""

    torch.manual_seed(0)
    model = CacheModel().eval()
    x = torch.randn(2, 3)
    graph = prepare_graph(model, x)
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    boundary = runtime.run_prefix(x)

    runtime.save_boundary(boundary, tmp_path / "boundary")
    loaded = runtime.load_boundary(tmp_path / "boundary")

    runtime.validate_boundary(loaded)
    assert torch.allclose(runtime.run_suffix(loaded), model(x), atol=1e-5, rtol=1e-4)


def test_training_boundary_cache_is_suffix_only(tmp_path) -> None:
    """Caching a graph-connected boundary strips prefix-backward metadata."""

    torch.manual_seed(0)
    model = CacheModel().train()
    x = torch.randn(2, 3)
    target = torch.randn(2, 2)
    graph = prepare_graph(model, x, features=SplitFeatures(training=True))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    boundary = runtime.run_training_prefix(x)

    runtime.save_boundary(boundary, tmp_path / "training_boundary")
    loaded = runtime.load_boundary(tmp_path / "training_boundary")

    assert loaded.metadata.get("supports_prefix_backward") is False
    assert "prefix_boundary_tensors" not in loaded.metadata
    _loss, grads = runtime.train_suffix(loaded, target)
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(loaded, grads)
