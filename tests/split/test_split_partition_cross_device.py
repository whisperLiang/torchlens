"""Noncontiguous DAG partitions preserve training across devices and recuts."""

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

from torchlens.split import (
    GraphPartition,
    PlacementPlan,
    SplitFeatures,
    SplitRuntime,
)


class InterleavedBranches(nn.Module):
    """Independent parameterized branches with a direct input skip."""

    def __init__(self) -> None:
        """Create two branches whose parameters can move independently."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.left_end = nn.ReLU()
        self.right_end = nn.Tanh()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interleave the branch calls so neither selected branch is an ordered cut."""
        left = self.left(x)
        right = self.right(x)
        return self.head(self.left_end(left) + self.right_end(right) + x)


def _branch_partition(runtime: SplitRuntime, branch: str) -> GraphPartition:
    """Select one complete independent branch and assert its noncontiguous order."""
    nodes = runtime.trace_graph.compute_nodes
    positions = [
        index for index, node in enumerate(nodes) if node.module_path in {branch, f"{branch}_end"}
    ]
    assert len(positions) == 2 and positions[-1] - positions[0] > 1
    return GraphPartition(frozenset(nodes[index].canonical_id for index in positions))


def _effective_parameters(runtime: SplitRuntime) -> dict[str, torch.Tensor]:
    """Read the exact tensors used by both segments through stable model IDs."""
    bindings = runtime.parameter_bindings()
    assert all(len(owners) == 1 for owners in bindings.values())
    return {
        state_id.removeprefix("parameter:"): next(iter(owners.values()))
        for state_id, owners in bindings.items()
    }


@pytest.mark.parametrize("retain_trace", [False, True])
@pytest.mark.parametrize(
    ("prefix_device", "suffix_device"),
    [("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cpu")],
    ids=["cpu-control", "cpu-to-cuda", "cuda-to-cpu"],
)
def test_noncontiguous_cross_device_training_and_external_optimizer_migration(
    retain_trace: bool, prefix_device: str, suffix_device: str
) -> None:
    """Compare B1/2/4/8 gradients and momentum updates while swapping branch ownership."""
    if "cuda:0" in (prefix_device, suffix_device) and not torch.cuda.is_available():
        pytest.skip("CUDA is required for heterogeneous DAG partition training")
    torch.manual_seed(319)
    # Double precision keeps this contract independent of CPU/CUDA float32 kernel
    # choices; no global TF32 setting or numerical tolerance is changed.
    model = InterleavedBranches().double()
    reference = deepcopy(model)
    runtime = prepare_fixture(
        model,
        torch.randn(1, 4, dtype=torch.float64),
        request_fixture(
            partial(select_node_prefix, node_id="left_end"),
            features=SplitFeatures(training=True, retain_trace=retain_trace),
        ),
    )
    partitions = (_branch_partition(runtime, "left"), _branch_partition(runtime, "right"))
    runtime = runtime.repartition(partitions[0]).with_placement(
        PlacementPlan.across(prefix_device, suffix_device)
    )
    graph = runtime.trace_graph
    effective = _effective_parameters(runtime)
    optimizer = torch.optim.SGD(effective.values(), lr=0.01, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    reference_parameters = dict(reference.named_parameters())
    assert effective.keys() == reference_parameters.keys()

    for step, batch in enumerate((1, 2, 4, 8)):
        x = torch.randn(batch, 4, dtype=torch.float64, requires_grad=True)
        split_x = x.detach().to(prefix_device).clone().requires_grad_(True)
        target = torch.randn(batch, 2, dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        expected_output = reference(x)
        actual_output = runtime.replay(split_x.detach())
        assert str(actual_output.device) == suffix_device
        torch.testing.assert_close(actual_output.cpu(), expected_output.detach())
        expected_loss = nn.functional.mse_loss(expected_output, target)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(split_x)
        assert any(
            graph.node_by_id[node_id].is_input
            for node_id in runtime.plan.boundary_bindings.values()
        ), "The direct input skip must exercise boundary gradient transport"
        loss, gradients = runtime.train_suffix(boundary, target.to(suffix_device))
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss.detach().cpu(), expected_loss.detach())
        torch.testing.assert_close(split_x.grad.cpu(), x.grad)
        for name, parameter in effective.items():
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad.cpu(), reference_parameters[name].grad)
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in effective.items():
            expected = reference_parameters[name]
            torch.testing.assert_close(parameter.detach().cpu(), expected.detach())
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"].cpu(),
                reference_optimizer.state[expected]["momentum_buffer"],
            )

        # Recut changes the device ownership of both branches. Optimizer history
        # remains caller-owned: copy it explicitly onto the newly bound tensors.
        selected = partitions[(step + 1) % 2]
        migration = runtime.plan_state_migration(selected)
        assert migration.supported
        moved = {entry.state_id: entry for entry in migration.entries}
        expected_action = (
            "reassociate" if prefix_device == suffix_device else "copy_and_reassociate"
        )
        for branch in ("left", "right"):
            for field in ("weight", "bias"):
                assert moved[f"parameter:{branch}.{field}"].optimizer_action == expected_action
        previous, previous_optimizer = effective, optimizer
        runtime = runtime.repartition(selected)
        assert runtime.trace_graph is graph
        effective = _effective_parameters(runtime)
        optimizer = torch.optim.SGD(effective.values(), lr=0.01, momentum=0.9)
        for name, parameter in effective.items():
            optimizer.state[parameter]["momentum_buffer"] = (
                previous_optimizer.state[previous[name]]["momentum_buffer"]
                .to(parameter.device)
                .clone()
            )
        torch.testing.assert_close(
            runtime.replay(split_x.detach()).cpu(), reference(x.detach()).detach()
        )
