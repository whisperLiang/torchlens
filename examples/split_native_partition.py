"""Run a compact split-training example with an external DAG partition policy.

Run from the repository root with ``python examples/split_native_partition.py``
after installing TorchLens. No model downloads or GPU are required.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from torchlens.split import SplitFeatures, prepare_graph


class BranchModel(nn.Module):
    """Interleaved small and wide branches followed by one shared classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 8)
        self.left_relu = nn.ReLU()
        self.right_relu = nn.ReLU()
        self.head = nn.Linear(12, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the branches in an interleaved capture order."""
        left = self.left(inputs)
        right = self.right(inputs)
        return self.head(torch.cat((self.left_relu(left), self.right_relu(right)), dim=1))


def main() -> None:
    """Check a full-model Adam update against compact split training and recut."""
    torch.manual_seed(41)
    model = BranchModel()
    reference = deepcopy(model)
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 2)
    graph = prepare_graph(model, inputs, features=SplitFeatures(training=True, retain_trace=False))
    dag = graph.compute_dag()
    profile = graph.graph_profile(batch_size=4)
    # An external algorithm can consume dag and profile. For this known model,
    # use the left branch as a fixed noncontiguous workload.
    prefix_call_ids = [
        call.call_id
        for call in dag.nodes
        if graph.trace_graph.node_by_id[call.output_node_ids[0]].module_path
        in {"left", "left_relu"}
    ]
    partition = dag.to_partition(prefix_call_ids)
    analysis = graph.analyze(partition)
    runtime = graph.materialize(analysis)
    ordered = [node.canonical_id for node in graph.trace_graph.compute_nodes]
    assert partition.prefix_node_ids != frozenset(ordered[: len(partition.prefix_node_ids)])
    alternate = dag.to_partition(
        call.call_id
        for call in dag.nodes
        if graph.trace_graph.node_by_id[call.output_node_ids[0]].module_path
        in {"right", "right_relu"}
    )
    migration = runtime.plan_state_migration(partition)
    print("Selected prefix nodes:", sorted(partition.prefix_node_ids))
    print("Preflight:", analysis.capability_report.preflight_ok)
    print("Static costs:", analysis.metrics)
    print("Forward FLOPs:", profile.forward_flops)
    print("State migration:", migration)
    print("Boundary layout:", runtime.boundary_layout(batch_size=4, training=True))
    torch.testing.assert_close(runtime.replay(inputs), reference(inputs))

    # This same-device example retains the original parameter objects, so one
    # caller-owned optimizer is valid across recuts. Cross-device callers must
    # inspect parameter_bindings() and the external optimizer migration contract.
    bindings = runtime.parameter_bindings()
    parameters = {id(value): value for owners in bindings.values() for value in owners.values()}
    optimizer = torch.optim.Adam(parameters.values(), lr=0.01)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
    for selected in (alternate, partition):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        expected_loss = nn.functional.mse_loss(reference(inputs), targets)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(inputs)
        actual_loss, boundary_gradients = runtime.train_suffix(boundary, targets)
        runtime.backward_prefix(boundary, boundary_gradients)
        torch.testing.assert_close(actual_loss, expected_loss)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
        migration = runtime.plan_state_migration(selected)
        assert migration.supported
        runtime = runtime.repartition(selected)
        rebound = {
            id(value)
            for owners in runtime.parameter_bindings().values()
            for value in owners.values()
        }
        assert rebound == parameters.keys(), (
            "Cross-device copies require explicit optimizer reassociation"
        )
        torch.testing.assert_close(runtime.replay(inputs), reference(inputs))
    print("Forward, parameter gradients and two Adam updates agree; recuts reused the capture.")


if __name__ == "__main__":
    main()
