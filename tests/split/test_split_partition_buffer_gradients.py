"""Frozen state crossing a DAG boundary must not become an autograd root."""

from __future__ import annotations

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from v2_helpers import (
    prepare_fixture,
    request_fixture,
    select_node_prefix,
)

from torchlens.split import GraphPartition, PlacementPlan, SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError

pytestmark = pytest.mark.usefixtures("fp64_functional")


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
    runtime = prepare_fixture(
        model,
        torch.randn(4, 4, device=device),
        request_fixture(
            partial(select_node_prefix, node_id="2"),
            features=SplitFeatures(training=True, retain_trace=retain_trace),
        ),
    )
    selected = frozenset(
        node.canonical_id
        for node in runtime.trace_graph.compute_nodes
        if node.canonical_id in runtime.plan.prefix_node_ids
    )
    partition = GraphPartition(selected)
    runtime = runtime.repartition(partition)
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
        runtime = runtime.repartition(partition)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("microbatch", [None, 2])
@pytest.mark.parametrize("retain_trace", [False, True])
def test_training_batchnorm_updates_current_state_through_repartition(
    device: str,
    microbatch: int | None,
    retain_trace: bool,
) -> None:
    """Native suffix buffer updates persist for full and sequential chunk training."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("A real CUDA device is required")
    torch.manual_seed(41)
    model = (
        nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.BatchNorm2d(4), nn.Tanh())
        .to(device)
        .double()
        .train()
    )
    reference = deepcopy(model)
    graph = prepare_graph(
        model,
        torch.randn(4, 4, 2, 2, device=device, dtype=torch.float64),
        features=SplitFeatures(training=True, retain_trace=retain_trace),
    )
    first = GraphPartition(frozenset({graph.trace_graph.compute_nodes[0].canonical_id}))
    runtime = graph.materialize(graph.analyze(first))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    for step, partition in enumerate((first, GraphPartition(frozenset()), first), 1):
        runtime = runtime.repartition(partition)
        optimizer.zero_grad(set_to_none=True)
        expected_optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4, 4, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        targets = torch.randn_like(x)
        size = microbatch or 4
        expected_loss = sum(
            nn.functional.mse_loss(
                reference(expected_x[start : start + size]), targets[start : start + size]
            )
            * size
            / 4
            for start in range(0, 4, size)
        )
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, targets, microbatch_size=microbatch)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, expected_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(model.state_dict(), reference.state_dict())
        assert model[1].num_batches_tracked.item() == step * (4 // size)


def test_mutable_buffer_training_refuses_disconnected_boundary() -> None:
    """A detached buffer snapshot cannot silently discard authoritative state writes."""
    model = nn.Sequential(nn.BatchNorm2d(4), nn.Tanh()).train()
    x = torch.randn(2, 4, 2, 2)
    graph = prepare_graph(model, x, features=SplitFeatures(training=True, retain_trace=False))
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    before = deepcopy(model.state_dict())
    boundary = runtime.run_prefix(x)
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.train_suffix(boundary, torch.zeros_like(x))
    torch.testing.assert_close(model.state_dict(), before)


def test_readonly_buffer_operand_does_not_require_connected_training_boundary() -> None:
    """A local activation write may read frozen state without mutating that state."""

    class Offset(nn.Module):
        """Add a read-only offset into a fresh activation allocation."""

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("offset", torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            value = x.clone()
            value.add_(self.offset)
            return value.tanh()

    model = Offset()
    x = torch.randn(2, 4)
    graph = prepare_graph(model, x, features=SplitFeatures(training=True, retain_trace=False))
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    reference_x = x.detach().clone().requires_grad_(True)
    target = torch.zeros_like(x)
    expected_loss = nn.functional.mse_loss(model(reference_x), target)
    expected_loss.backward()
    boundary = runtime.run_prefix(x)
    loss, gradients = runtime.train_suffix(boundary, target)
    torch.testing.assert_close(loss, expected_loss)
    input_key = next(
        key
        for key, value in runtime.plan.boundary_bindings.items()
        if value in graph.trace_graph.input_node_ids
    )
    torch.testing.assert_close(gradients[input_key], reference_x.grad)
    torch.testing.assert_close(model.offset, torch.ones(4))


@pytest.mark.parametrize("prefix,suffix", [("cpu", "cuda"), ("cuda", "cpu")])
@pytest.mark.parametrize("retain_trace", [False, True])
@pytest.mark.parametrize("microbatch", [None, 2])
def test_transported_mutable_buffers_survive_training_and_recuts(
    prefix: str, suffix: str, retain_trace: bool, microbatch: int | None
) -> None:
    """Effective state, not a temporary device copy, retains every native buffer update."""
    if not torch.cuda.is_available():
        pytest.skip("Real CPU/CUDA buffer transport is required")
    from torchlens.split.ownership import _inventory

    model = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.BatchNorm2d(4), nn.Tanh())
    model = model.to(prefix).double().train()
    reference = deepcopy(model)
    graph = prepare_graph(
        model,
        torch.randn(4, 4, 2, 2, dtype=torch.float64, device=prefix),
        features=SplitFeatures(training=True, retain_trace=retain_trace),
        placement=PlacementPlan.across(prefix, suffix),
    )
    nodes = graph.trace_graph.compute_nodes
    partial = GraphPartition(frozenset({nodes[0].canonical_id}))
    full = GraphPartition(frozenset(node.canonical_id for node in nodes))
    runtime = graph.materialize(graph.analyze(partial))
    for partition in (partial, full, partial):
        runtime = runtime.repartition(partition)
        reference.zero_grad(set_to_none=True)
        for owners in runtime.parameter_bindings().values():
            for parameter in owners.values():
                parameter.grad = None
        x = torch.randn(4, 4, 2, 2, dtype=torch.float64, device=prefix, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)
        # Prefix execution is one logical forward; only suffix BatchNorm is chunked.
        chunk = 4 if partition == full else microbatch or 4
        expected = sum(
            reference(ref_x[start : start + chunk]).square().mean() * chunk / 4
            for start in range(0, 4, chunk)
        )
        expected.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(
            boundary, None, loss_fn=lambda out, _: out.square().mean(), microbatch_size=microbatch
        )
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss.cpu(), expected.cpu())
        torch.testing.assert_close(x.grad, ref_x.grad)
        expected_params = dict(reference.named_parameters())
        for name, owners in runtime.parameter_bindings().items():
            for parameter in owners.values():
                torch.testing.assert_close(
                    parameter.grad.cpu(),
                    expected_params[name.removeprefix("parameter:")].grad.cpu(),
                )
        rows, values = _inventory(runtime, runtime._segment_state_snapshots())
        expected_buffers = dict(reference.named_buffers())
        for row in rows:
            if row.kind == "buffer":
                for value in values[row.state_id].values():
                    torch.testing.assert_close(value.cpu(), expected_buffers[row.aliases[0]].cpu())


@pytest.mark.parametrize("prefix,suffix", [("cpu", "cpu"), ("cpu", "cuda"), ("cuda", "cpu")])
def test_shared_buffer_writes_preserve_prefix_autograd_storage(prefix: str, suffix: str) -> None:
    """Publication must not overwrite running statistics saved by prefix backward."""
    if "cuda" in (prefix, suffix) and not torch.cuda.is_available():
        pytest.skip("Real CPU/CUDA buffer transport is required")
    from torchlens.split.ownership import _inventory

    class ReusedNorm(nn.Module):
        """Use one mutable state source on both sides without shared trainable weights."""

        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.BatchNorm2d(4, affine=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Exercise two native writes and two connected gradient paths."""
            return self.norm(self.norm(x).tanh()).sin()

    model = ReusedNorm().to(prefix).double().train()
    reference = deepcopy(model)
    graph = prepare_graph(
        model,
        torch.randn(2, 4, 2, 2, dtype=torch.float64, device=prefix),
        features=SplitFeatures(training=True, retain_trace=False),
        placement=PlacementPlan.across(prefix, suffix),
    )
    nodes = graph.trace_graph.compute_nodes
    cut = next(index + 1 for index, node in enumerate(nodes) if node.op_type == "tanh")
    partition = GraphPartition(frozenset(node.canonical_id for node in nodes[:cut]))
    runtime = graph.materialize(graph.analyze(partition))
    for _step in range(2):
        x = torch.randn(2, 4, 2, 2, dtype=torch.float64, device=prefix, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)
        expected = reference(ref_x).square().mean()
        expected.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(
            boundary, None, loss_fn=lambda out, _: out.square().mean()
        )
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss.cpu(), expected.cpu())
        torch.testing.assert_close(x.grad, ref_x.grad)
        rows, values = _inventory(runtime, runtime._segment_state_snapshots())
        for row in rows:
            if row.kind == "buffer":
                for value in values[row.state_id].values():
                    torch.testing.assert_close(
                        value.cpu(), dict(reference.named_buffers())[row.aliases[0]].cpu()
                    )
        runtime = runtime.repartition(partition)
