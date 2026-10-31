"""Static boundary layout and caller-owned payload isolation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch import nn
from v2_helpers import (
    analyze_fixture,
    prepare_fixture,
    request_fixture,
    select_node_prefix,
)

from torchlens.split import GraphPartition, SplitFeatures
from torchlens.split.boundary_layout import build_boundary_layout
from torchlens.split.graph_profile import partition_metrics


class ChunkBranches(nn.Module):
    """Expose multiple views and a shared-input passthrough boundary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Join two half-width outputs and two uses of one half."""
        left, right = torch.chunk(x, 2, dim=1)
        return left + right + left * 2 + x[:, :4]


def test_layout_uses_canonical_bindings_and_static_shapes() -> None:
    """ABI keys with paths map to canonical values without executing a model."""
    model = ChunkBranches()
    runtime = prepare_fixture(
        model, torch.randn(2, 8), request_fixture(partial(select_node_prefix, node_id="chunk_1_1"))
    )
    partition = GraphPartition(
        frozenset(n.canonical_id for n in runtime.trace_graph.compute_nodes if n.op_type == "chunk")
    )
    runtime = runtime.repartition(partition)
    with patch.object(model, "forward", side_effect=AssertionError("must not execute")):
        layout = build_boundary_layout(
            runtime.graph_profile(batch_size=8), runtime.plan, training=True
        )
    assert layout.unique_tensor_count == 3
    assert layout.forward_payload_bytes == 8 * (8 + 4 + 4) * 4
    assert layout.backward_payload_bytes is None
    assert all(row.value_id == runtime.plan.boundary_bindings[row.key] for row in layout.tensors)
    assert {
        tuple(getattr(part, "index", part) for part in row.container_path) for row in layout.tensors
    } >= {(0,), (1,)}
    assert all(row.alias_group is None for row in layout.tensors)
    assert layout.unknown_cost_diagnostics
    with pytest.raises(FrozenInstanceError):
        layout.training = False


def test_unknown_layout_bytes_and_inference_backward_requirement() -> None:
    """Unresolved tensor sizes stay unknown instead of pretending to cost zero."""
    runtime = prepare_fixture(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        torch.randn(2, 4),
        request_fixture(partial(select_node_prefix, node_id="linear")),
    )
    profile = runtime.graph_profile()
    profile = replace(
        profile, tensors=tuple(replace(t, bytes=None, shape=None) for t in profile.tensors)
    )
    layout = build_boundary_layout(profile, runtime.plan)
    assert layout.forward_payload_bytes is None
    assert layout.backward_payload_bytes == 0
    assert not layout.training


class MutableIndices(nn.Module):
    """A legal point split whose suffix mutates an integer index tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 3)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Create an isolated prefix value, mutate it, then use it as indices."""
        copied = ids.clone()
        copied.add_(1)
        return self.embedding(copied)


def test_training_integer_boundary_isolation() -> None:
    """Repeated suffix training must not modify the caller's index payload."""
    model = MutableIndices()
    inputs = torch.tensor([[0, 1], [2, 3]])
    runtime = prepare_fixture(
        model,
        inputs,
        request_fixture(
            partial(select_node_prefix, node_id="clone"), features=SplitFeatures(training=True)
        ),
    )
    boundary = runtime.run_training_prefix(inputs)
    before = {key: value.clone() for key, value in boundary.tensors.items()}
    targets = torch.ones(2, 2, 3)
    first, _ = runtime.train_suffix(boundary, targets)
    second, _ = runtime.train_suffix(boundary, targets)
    torch.testing.assert_close(first, second)
    for key, value in boundary.tensors.items():
        assert torch.equal(value, before[key])
    layout = build_boundary_layout(runtime.graph_profile(), runtime.plan, training=True)
    assert all(row.backward_required is False for row in layout.tensors)
    assert layout.backward_payload_bytes == 0
    metrics = partition_metrics(runtime.graph_profile(), runtime.plan, training=True)
    assert metrics.backward_gradient_bytes == layout.backward_payload_bytes
    assert "backward_gradient_bytes:loss_path_not_observed" not in metrics.unknown_cost_diagnostics


@pytest.mark.parametrize("dtype", ["torch.float32", "torch.complex64", "unknown", None])
def test_training_costs_do_not_treat_capture_grad_flags_as_a_loss_contract(
    dtype: str | None,
) -> None:
    """Floating and unknown boundary types retain unknown loss-dependent gradient costs."""

    runtime = prepare_fixture(
        nn.Sequential(nn.ReLU(), nn.Sigmoid()),
        torch.randn(2, 4),
        request_fixture(
            partial(select_node_prefix, node_id="relu"), features=SplitFeatures(training=True)
        ),
    )
    profile = replace(
        runtime.graph_profile(),
        tensors=tuple(
            replace(t, dtype=dtype, requires_grad=False) for t in runtime.graph_profile().tensors
        ),
    )
    plan = replace(
        runtime.plan,
        boundary_spec={
            key: replace(schema, dtype=dtype, requires_grad=False)
            for key, schema in runtime.plan.boundary_spec.items()
        },
    )
    layout = build_boundary_layout(profile, plan, training=True)
    metrics = partition_metrics(profile, plan, training=True)
    assert layout.backward_payload_bytes is metrics.backward_gradient_bytes is None
    assert all(row.backward_required is None for row in layout.tensors)


def test_empty_frontier_has_zero_payload_even_for_training() -> None:
    """A source-only suffix transfers no tensors and requires no boundary gradients."""

    class Constant(nn.Module):
        """Produce a tensor independent of the supplied input."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Allocate the complete output in the suffix."""

            return torch.ones(2, 4)

    runtime = prepare_fixture(
        Constant(),
        torch.randn(2, 4),
        request_fixture(
            partial(select_node_prefix, node_id="ones"), features=SplitFeatures(training=True)
        ),
    )
    analysis = analyze_fixture(runtime, GraphPartition(frozenset()))
    layout = build_boundary_layout(runtime.graph_profile(), analysis.plan, training=True)
    assert layout.tensors == ()
    assert analysis.metrics.unique_boundary_tensors == 0
    assert layout.forward_payload_bytes == analysis.metrics.forward_boundary_bytes == 0
    assert layout.backward_payload_bytes == analysis.metrics.backward_gradient_bytes == 0


def test_boundary_same_device_move_and_noncontiguous_clone() -> None:
    """Same-device movement avoids allocation; explicit clone isolates view storage."""
    runtime = prepare_fixture(
        nn.Sequential(nn.Identity(), nn.ReLU()),
        torch.randn(2, 4),
        request_fixture(partial(select_node_prefix, node_id="relu")),
    )
    boundary = runtime.run_prefix(torch.randn(2, 8)[:, ::2])
    moved = boundary.to("cpu")
    cloned = boundary.clone()
    for key, value in boundary.tensors.items():
        assert moved.tensors[key] is value
        assert cloned.tensors[key].data_ptr() != value.data_ptr()
        torch.testing.assert_close(cloned.tensors[key], value)
