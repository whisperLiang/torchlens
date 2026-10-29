"""Frozen state crossing a DAG boundary must not become an autograd root."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, SplitRequest, after, prepare


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("microbatch", [None, 2])
@pytest.mark.parametrize("retain_trace", [False, True])
def test_eval_batchnorm_state_stays_frozen_across_partition_and_recut(
    device: str, microbatch: int | None, retain_trace: bool
) -> None:
    """Compare two momentum updates with BatchNorm buffers in the frontier."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for native cuDNN BatchNorm regression coverage")
    torch.manual_seed(902)
    model = (
        nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Linear(4, 2),
        )
        .to(device)
        .eval()
    )
    reference = deepcopy(model)
    runtime = prepare(
        model,
        torch.randn(4, 4, device=device),
        SplitRequest(after("2"), features=SplitFeatures(training=True, retain_trace=retain_trace)),
    )
    selected = frozenset(
        node.canonical_id
        for node in runtime.trace_graph.compute_nodes
        if node.canonical_id in runtime.plan.prefix_node_ids
    )
    partition = GraphPartition(selected)
    runtime = runtime.partition(partition)
    buffer_keys = {
        key
        for key, node_id in runtime.plan.boundary_bindings.items()
        if runtime.trace_graph.node_by_id[node_id].is_buffer
    }
    assert buffer_keys, "This regression requires frozen running statistics in the frontier"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    for _ in range(2):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        targets = torch.randn(4, 2, device=device)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        expected_loss = nn.functional.mse_loss(reference(expected_x), targets)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, targets, microbatch_size=microbatch)
        assert not buffer_keys.intersection(gradients)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, expected_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(model.buffers(), reference.buffers(), strict=True):
            torch.testing.assert_close(actual, expected)
        runtime = runtime.partition(partition)
