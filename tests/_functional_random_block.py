"""Functional execution of the existing random-graph blocks for op-count stress."""

import torch
from example_models import _AttentionBlock, _BranchMergeBlock, _ResidualBlock, _SequentialBlock
from torch import nn
from torch.nn import functional as F


def run_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Execute the original block's tensor DAG without extra module invocations.

    Parameters
    ----------
    block:
        One of RandomGraphModel's four existing block families, with its real weights.
    x:
        The original full-width input tensor.

    Returns
    -------
    torch.Tensor
        The same output as the module form, computed through public Torch operations.
    """

    def linear(layer: nn.Linear, value: torch.Tensor) -> torch.Tensor:
        """Use the same learned affine operation as Linear.forward."""
        return F.linear(value, layer.weight, layer.bias)

    if isinstance(block, (_SequentialBlock, _ResidualBlock)):
        result = linear(block.fc2, F.relu(linear(block.fc1, x)))
        return result + x if isinstance(block, _ResidualBlock) else result
    if isinstance(block, _BranchMergeBlock):
        branches = (
            F.relu(linear(block.branch1[0], x)),
            torch.tanh(linear(block.branch2[0], x)),
            torch.sigmoid(linear(block.branch3[0], x)),
        )
        return linear(block.merge, torch.cat(branches, dim=-1))
    if isinstance(block, _AttentionBlock):
        q, k, v = linear(block.q_proj, x), linear(block.k_proj, x), linear(block.v_proj, x)
        attention = torch.matmul(q, k.transpose(-2, -1)) / block.scale
        attention = torch.softmax(attention, dim=-1)
        return linear(block.out_proj, torch.matmul(attention, v))
    raise TypeError(f"Unsupported random-graph block: {type(block).__name__}")
