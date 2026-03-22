"""End-to-end demo for execution-plan replay, graph splitting, and split backward."""

from __future__ import annotations

import torch
from torch import nn

from torchlens import (
    backward_prefix_from_boundary,
    benchmark_replay,
    compile_execution_plan,
    enumerate_frontier_splits,
    replay_forward,
    replay_partitioned,
    train_partitioned,
    validate_gradient_equivalence,
    validate_replay_equivalence,
    validate_split_equivalence,
)


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = torch.relu(self.left(x))
        right = torch.sigmoid(self.right(x))
        return self.head(torch.cat([left, right], dim=-1))


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


def main() -> None:
    model = DemoNet().train()
    example_inputs = torch.randn(4, 4)
    targets = torch.randn(4, 2)

    plan = compile_execution_plan(model, example_inputs)
    print(plan)

    full_output = replay_forward(plan, example_inputs)
    print("replay_forward:", tuple(full_output.shape))

    splits = enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=8)
    split = splits[0]
    print("frontier split:", split)

    partitioned = replay_partitioned(plan, example_inputs, split=split, return_boundary=True)
    print("replay_partitioned output:", tuple(partitioned["output"].shape))
    print("boundary labels:", partitioned["boundary"]["labels"])

    train_result = train_partitioned(
        plan,
        example_inputs,
        split=split,
        targets=targets,
        loss_fn=mse_loss,
        return_boundary=True,
        return_boundary_grad=True,
        step_optimizer=False,
    )
    print("train_partitioned loss:", float(train_result["loss"].item()))

    prefix_backward = backward_prefix_from_boundary(
        plan,
        example_inputs,
        split,
        train_result["boundary_grad"],
        step_optimizer=False,
        match_boundary=True,
        cached_boundary=train_result["boundary"],
    )
    print("backward_prefix_from_boundary:", prefix_backward["boundary"]["labels"])

    print("validate_replay_equivalence:", validate_replay_equivalence(plan, model, example_inputs))
    print(
        "validate_split_equivalence:",
        validate_split_equivalence(plan, split, model, example_inputs),
    )
    print(
        "validate_gradient_equivalence:",
        validate_gradient_equivalence(
            plan,
            split,
            model,
            example_inputs,
            targets=targets,
            loss_fn=mse_loss,
            atol=1e-5,
            rtol=1e-4,
        ),
    )
    print(
        "benchmark_replay:",
        benchmark_replay(plan, example_inputs, split=split, iterations=3, warmup=1),
    )


if __name__ == "__main__":
    main()
