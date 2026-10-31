"""Run a compact split-training example with an external DAG partition policy.

Run from the repository root with ``python examples/split_native_partition.py``
after installing TorchLens. No model downloads or GPU are required.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from torchlens.split import GraphPartition, GraphProfile, SplitFeatures, prepare_graph


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


def choose_small_ready_calls(profile: GraphProfile, max_calls: int = 2) -> GraphPartition:
    """Choose dependency-ready calls with the smallest logical output sizes.

    This is an illustrative external policy, not a placement optimizer. It keeps
    native multi-output calls atomic and never infers latency from tensor bytes.
    Unknown sizes sort last. Sources are available on the client implicitly.
    """

    nodes = {node.node_id: node for node in profile.nodes}
    sources = {tensor.value_id for tensor in profile.tensors} - nodes.keys()
    remaining = {node.call_id: node.output_node_ids for node in profile.nodes}
    selected: set[str] = set()
    for _ in range(max_calls):
        ready = [
            call_id
            for call_id, outputs in remaining.items()
            if all(set(nodes[node_id].input_node_ids) <= selected | sources for node_id in outputs)
        ]
        if not ready:
            break

        def cost(call_id: str) -> tuple[float, str]:
            """Order eligible calls by known logical bytes, with deterministic ties."""
            sizes = [nodes[node_id].activation_bytes for node_id in remaining[call_id]]
            return (
                float("inf")
                if any(size is None for size in sizes)
                else float(sum(size for size in sizes if size is not None)),
                call_id,
            )

        chosen = min(ready, key=cost)
        selected.update(remaining.pop(chosen))
    return GraphPartition(frozenset(selected))


def main() -> None:
    """Check a full-model Adam update against compact split training and recut."""
    torch.manual_seed(41)
    model = BranchModel()
    reference = deepcopy(model)
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 2)
    graph = prepare_graph(model, inputs, features=SplitFeatures(training=True, retain_trace=False))
    profile = graph.graph_profile(batch_size=4)
    partition = choose_small_ready_calls(profile)
    analysis = graph.analyze(partition)
    runtime = graph.materialize(analysis)
    ordered = [node.canonical_id for node in graph.trace_graph.compute_nodes]
    assert partition.prefix_node_ids != frozenset(ordered[: len(partition.prefix_node_ids)])
    alternate = GraphPartition(
        frozenset(
            node.canonical_id
            for node in graph.trace_graph.compute_nodes
            if node.module_path in {"right", "right_relu"}
        )
    )
    migration = runtime.plan_state_migration(partition)
    print("Selected prefix nodes:", sorted(partition.prefix_node_ids))
    print("Preflight:", analysis.capability_report.preflight_ok)
    print("Static costs:", analysis.metrics)
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
