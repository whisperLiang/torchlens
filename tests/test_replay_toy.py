"""Toy-model tests for execution-plan compilation and replay."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from torchlens import compile_execution_plan, enumerate_frontier_splits, replay_forward
from torchlens import replay_partitioned
from torchlens import validate_replay_equivalence


@dataclass
class ToyOutput:
    logits: torch.Tensor
    aux: dict


class MultiOutputToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> ToyOutput:
        left = torch.relu(self.left(x))
        right = torch.sigmoid(self.right(x))
        merged = torch.cat([left, right], dim=-1)
        logits = self.head(merged)
        return ToyOutput(logits=logits, aux={"left": left, "right": right})


class BatchReshapeToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.reshape(x.size(0), -1)
        hidden = torch.relu(self.hidden(flattened))
        return self.head(hidden)


class BatchFactoryToy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_bias = torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)
        return x[:, :1] + batch_bias


class BufferInplaceOutputToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bias", torch.randn(4))

    def forward(self, x: torch.Tensor) -> dict:
        biased = x + self.bias
        mutated = biased.clone()
        mutated.add_(1)
        return {"main": mutated * 2, "aux": (biased, mutated)}


class DropoutToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class ViewAliasInplaceBoundaryToy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundary = x * 2
        alias = boundary.view_as(boundary)
        alias.add_(1)
        return alias * 3


@pytest.mark.smoke
def test_compile_execution_plan_and_replay_forward_toy() -> None:
    model = MultiOutputToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs)

    replay_output = replay_forward(plan, example_inputs)
    direct_output = model(example_inputs)

    assert torch.allclose(replay_output.logits, direct_output.logits)
    assert torch.allclose(replay_output.aux["left"], direct_output.aux["left"])
    assert torch.allclose(replay_output.aux["right"], direct_output.aux["right"])


@pytest.mark.smoke
def test_validate_replay_equivalence_toy() -> None:
    model = MultiOutputToy().eval()
    example_inputs = torch.randn(3, 4)
    plan = compile_execution_plan(model, example_inputs)

    assert validate_replay_equivalence(plan, model, example_inputs)


@pytest.mark.smoke
def test_replay_forward_allows_batch_size_change_for_reshape() -> None:
    model = BatchReshapeToy().eval()
    example_inputs = torch.randn(2, 2, 2)
    new_inputs = torch.randn(5, 2, 2)
    plan = compile_execution_plan(model, example_inputs)

    replay_output = replay_forward(plan, new_inputs)
    direct_output = model(new_inputs)

    assert torch.allclose(replay_output, direct_output)


@pytest.mark.smoke
def test_replay_forward_allows_batch_size_change_for_factory_sizes() -> None:
    model = BatchFactoryToy().eval()
    example_inputs = torch.randn(2, 3)
    new_inputs = torch.randn(5, 3)
    plan = compile_execution_plan(model, example_inputs)

    replay_output = replay_forward(plan, new_inputs)
    direct_output = model(new_inputs)

    assert torch.allclose(replay_output, direct_output)


@pytest.mark.smoke
def test_plan_trace_matches_modellog_compile_for_buffer_inplace_outputs() -> None:
    model = BufferInplaceOutputToy().eval()
    example_inputs = torch.randn(2, 4)
    plan_trace = compile_execution_plan(model, example_inputs, trace_mode="plan")
    modellog_trace = compile_execution_plan(model, example_inputs, trace_mode="modellog")

    plan_output = replay_forward(plan_trace, example_inputs)
    modellog_output = replay_forward(modellog_trace, example_inputs)
    direct_output = model(example_inputs)

    assert torch.allclose(plan_output["main"], direct_output["main"])
    assert torch.allclose(plan_output["aux"][0], direct_output["aux"][0])
    assert torch.allclose(plan_output["aux"][1], direct_output["aux"][1])
    assert torch.allclose(modellog_output["main"], direct_output["main"])


@pytest.mark.smoke
def test_execution_plan_and_splits_include_flops_metadata() -> None:
    model = MultiOutputToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs)
    split = enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=8)[0]

    assert plan.meta["total_flops_forward"] > 0
    assert plan.meta["total_flops_backward"] > 0
    assert plan.meta["total_flops_total"] == (
        plan.meta["total_flops_forward"] + plan.meta["total_flops_backward"]
    )
    assert all("flops_forward" in node.meta for node in plan.nodes)
    assert split.meta["prefix_flops_total"] == (
        split.meta["prefix_flops_forward"] + split.meta["prefix_flops_backward"]
    )
    assert split.meta["suffix_flops_total"] == (
        split.meta["suffix_flops_forward"] + split.meta["suffix_flops_backward"]
    )
    assert split.meta["boundary_node_count"] == len(split.boundary_indices)


@pytest.mark.smoke
def test_compile_execution_plan_rng_capture_is_opt_in() -> None:
    model = DropoutToy().train()
    example_inputs = torch.randn(4, 4)

    fast_plan = compile_execution_plan(model, example_inputs)
    deterministic_plan = compile_execution_plan(model, example_inputs, preserve_rng=True)

    assert all(node.rng_state is None for node in fast_plan.nodes)
    assert any(node.rng_state for node in deterministic_plan.nodes)


@pytest.mark.smoke
def test_boundary_snapshot_auto_detects_alias_inplace_suffix() -> None:
    model = ViewAliasInplaceBoundaryToy().eval()
    example_inputs = torch.randn(2, 4)
    plan = compile_execution_plan(model, example_inputs)
    boundary_idx = next(
        node.idx for node in plan.nodes if node.meta.get("func_name") == "__mul__"
    )
    split = next(
        split
        for split in enumerate_frontier_splits(plan, max_frontier_size=1, max_splits=8)
        if split.boundary_indices == [boundary_idx]
    )

    result = replay_partitioned(plan, example_inputs, split=split, return_boundary=True)
    boundary = result["boundary"]["tensors"][split.boundary_labels[0]]
    direct_output = model(example_inputs)
    boundary_output = replay_partitioned(plan, result["boundary"], split=split)

    assert torch.allclose(boundary, example_inputs * 2)
    assert torch.allclose(result["output"], direct_output)
    assert torch.allclose(boundary_output, direct_output)
