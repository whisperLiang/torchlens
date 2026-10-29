"""Stable ownership metadata and external optimizer migration contracts."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.split import GraphPartition, PlacementPlan, SplitFeatures, SplitRequest, after
from torchlens.split.ownership import parameter_bindings, parameter_ownership, plan_state_migration
from torchlens.split.state import SegmentState


class SharedState(torch.nn.Module):
    """Use one parameter through two names and retain a registered buffer."""

    def __init__(self) -> None:
        """Create tied sources whose names survive capture and materialization."""

        super().__init__()
        self.left = torch.nn.Linear(4, 4)
        self.right = torch.nn.Linear(4, 4)
        self.right.weight = self.left.weight
        self.middle = torch.nn.Tanh()
        self.register_buffer("offset", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume tied weights on opposite sides of the cut."""

        return self.right(self.middle(self.left(x))) + self.offset


def _runtime(model: torch.nn.Module, *, retain_trace: bool = False) -> Any:
    """Prepare live Torch state with either supported retention mode."""

    return tl.split.prepare(
        model,
        torch.randn(2, 4),
        SplitRequest(
            after("middle"),
            features=SplitFeatures(training=True, retain_trace=retain_trace),
        ),
    )


@pytest.mark.parametrize("retain_trace", [False, True])
def test_ownership_deduplicates_tied_names_and_buffers_without_lazy_mutation(
    retain_trace: bool,
) -> None:
    """Metadata is immutable, tensor-free, and independent of Python object addresses."""

    model = SharedState()
    other_model = copy.deepcopy(model)
    runtime = _runtime(model, retain_trace=retain_trace)
    states = [runtime.segments.training_prefix._state, runtime.segments.suffix._state]
    before = [dict(state._entries) for state in states]
    rows = {row.state_id: row for row in parameter_ownership(runtime)}
    assert [state._entries for state in states] == before
    weight = rows["parameter:left.weight"]
    assert weight.aliases == ("left.weight", "right.weight")
    assert weight.owners == ("prefix", "suffix")
    assert weight.trainable and weight.optimizer_managed
    assert rows["buffer:offset"].kind == "buffer"
    assert not rows["buffer:offset"].optimizer_managed
    other = _runtime(other_model, retain_trace=retain_trace)
    assert {row.state_id for row in parameter_ownership(other)} == set(rows)
    if retain_trace:
        runtime.trace.cleanup()
        other.trace.cleanup()


def test_parameter_bindings_exposes_referenced_and_owned_effective_parameters() -> None:
    """Optimizer mappings include live referenced parameters and deduplicate named aliases."""

    model = SharedState()
    runtime = _runtime(model)
    bindings = parameter_bindings(runtime)
    assert set(bindings) == {"parameter:left.weight", "parameter:left.bias", "parameter:right.bias"}
    assert bindings["parameter:left.weight"]["prefix"] is model.left.weight
    assert bindings["parameter:left.weight"]["suffix"] is model.left.weight
    with pytest.raises(TypeError):
        bindings["parameter:left.weight"]["prefix"] = torch.zeros(4, 4)  # type: ignore[index]


def test_migration_describes_directions_without_building_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis describes parameter and buffer movement without materializing state."""

    runtime = _runtime(SharedState())
    states = [runtime.segments.training_prefix._state, runtime.segments.suffix._state]
    before = [dict(state._entries) for state in states]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Fail if metadata accidentally builds an executable."""

        raise AssertionError("migration planning must not construct segments")

    monkeypatch.setattr(runtime.adapter, "build_segments", forbidden)
    partition = GraphPartition(
        frozenset(node.canonical_id for node in runtime.trace_graph.compute_nodes)
    )
    migration = plan_state_migration(runtime, partition)
    assert migration.supported
    assert migration.optimizer_owner == "caller"
    rows = {row.state_id: row for row in migration.entries}
    assert rows["parameter:right.bias"].from_owners == ("suffix",)
    assert rows["parameter:right.bias"].to_owners == ("prefix",)
    assert rows["parameter:right.bias"].optimizer_action == "reassociate"
    assert rows["parameter:left.weight"].identity_preserved
    assert [state._entries for state in states] == before


def test_migration_refuses_ambiguous_shared_optimizer_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal tied replicas do not prove that their independent optimizer histories agree."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    runtime = _runtime(SharedState()).with_placement(PlacementPlan.on("cpu"))
    bindings = parameter_bindings(runtime)["parameter:left.weight"]
    assert bindings["prefix"] is not bindings["suffix"]
    migration = plan_state_migration(
        runtime, after(runtime.trace_graph.compute_nodes[-1].canonical_id)
    )
    assert not migration.supported
    assert "shared_optimizer_state_ambiguous" in migration.unsupported_reasons
    shared = next(row for row in migration.entries if row.state_id == "parameter:left.weight")
    assert shared.optimizer_action == "unsupported"
    assert shared.identity_preserved is None


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam"])
def test_optimizer_bindings_remain_valid_after_repartition(kind: str) -> None:
    """Same-device recuts preserve object identity and accumulated external optimizer state."""

    reference = SharedState()
    model = copy.deepcopy(reference)
    runtime = _runtime(model)

    def unique_parameters(bound: Any) -> list[torch.Tensor]:
        """Deduplicate shared parameter objects when constructing one optimizer."""

        return list(
            {id(value): value for owners in bound.values() for value in owners.values()}.values()
        )

    def optimizer(parameters: Any) -> torch.optim.Optimizer:
        """Use identical hyperparameters for both execution paths."""

        if kind == "adam":
            return torch.optim.Adam(parameters, lr=0.002)
        return torch.optim.SGD(parameters, lr=0.002, momentum=0.9 if kind == "momentum" else 0)

    bindings = parameter_bindings(runtime)
    actual_optimizer = optimizer(unique_parameters(bindings))
    expected_optimizer = optimizer(reference.parameters())
    for step in range(3):
        actual_optimizer.zero_grad(set_to_none=True)
        expected_optimizer.zero_grad(set_to_none=True)
        inputs, targets = torch.randn(4, 4), torch.randn(4, 4)
        expected_loss = (reference(inputs) - targets).square().mean()
        expected_loss.backward()
        boundary = runtime.run_training_prefix(inputs)
        loss, gradients = runtime.train_suffix(boundary, targets)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        actual_optimizer.step()
        expected_optimizer.step()
        for name, parameter in model.named_parameters():
            expected = dict(reference.named_parameters())[name]
            torch.testing.assert_close(parameter, expected)
            torch.testing.assert_close(
                actual_optimizer.state[parameter], expected_optimizer.state[expected]
            )
        chosen = (
            frozenset(node.canonical_id for node in runtime.trace_graph.compute_nodes)
            if step % 2 == 0
            else frozenset()
        )
        partition = GraphPartition(chosen)
        assert plan_state_migration(runtime, partition).supported
        runtime = runtime.partition(partition)
        updated = parameter_bindings(runtime)
        for state_id, owners in updated.items():
            assert all(
                value is next(iter(bindings[state_id].values())) for value in owners.values()
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_migration_reports_parameter_and_buffer_device_moves() -> None:
    """A new cut reports CUDA state moving to a CPU prefix and external Adam work."""

    class Buffered(torch.nn.Module):
        """Keep a suffix-only buffer next to an independent suffix parameter."""

        def __init__(self) -> None:
            """Construct a sequential graph with state on both segments."""

            super().__init__()
            self.left = torch.nn.Linear(4, 4)
            self.middle = torch.nn.Tanh()
            self.right = torch.nn.Linear(4, 4)
            self.register_buffer("offset", torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Read the buffer only after the partition point."""

            return self.right(self.middle(self.left(x))) + self.offset

    model = Buffered()
    runtime = _runtime(model).with_placement(PlacementPlan.across("cpu", "cuda:0"))
    old_bindings = runtime.parameter_bindings()
    migration = runtime.plan_state_migration(
        after(runtime.trace_graph.compute_nodes[-1].canonical_id)
    )
    assert migration.supported
    rows = {row.state_id: row for row in migration.entries}
    moved_parameter = rows["parameter:right.weight"]
    assert moved_parameter.optimizer_action == "copy_and_reassociate"
    assert not moved_parameter.identity_preserved
    assert moved_parameter.source_devices == (("suffix", "cuda:0"),)
    assert moved_parameter.destination_devices == (("prefix", "cpu"),)
    moved_buffer = rows["buffer:offset"]
    assert moved_buffer.source_devices == (("suffix", "cuda:0"),)
    assert moved_buffer.destination_devices == (("prefix", "cpu"),)
    assert moved_buffer.optimizer_action == "none"
    recut = runtime.at(after(runtime.trace_graph.compute_nodes[-1].canonical_id))
    new_bindings = recut.parameter_bindings()
    torch.testing.assert_close(
        new_bindings["parameter:right.weight"]["prefix"],
        old_bindings["parameter:right.weight"]["suffix"].cpu(),
    )
    assert new_bindings["parameter:right.weight"]["prefix"] is not model.right.weight
