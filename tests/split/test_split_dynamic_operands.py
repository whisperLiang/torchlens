"""Dynamic split sizes are operand values, not axis numbers or arbitrary literals."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, SplitFeatures, prepare_graph


class Grouped(nn.Module):
    """Group queries into the batch axis, then restore samples without randomness."""

    def __init__(self, spelling: str) -> None:
        super().__init__()
        self.spelling = spelling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve axes and fixed query chunk sizes even when they equal B."""
        batch = x.shape[0]
        grouped = torch.cat(x.split(2, dim=1), dim=0) * 2
        if self.spelling == "keyword":
            parts = grouped.split(split_size=batch, dim=0)
        elif self.spelling == "sections":
            parts = torch.split(grouped, [batch] * 3, dim=0)
        elif self.spelling == "uneven_sections":
            first, rest = torch.split_with_sizes(grouped, split_sizes=[batch, 2 * batch], dim=0)
            parts = (first, *rest.split(batch, dim=0))
        elif self.spelling == "chunk":
            parts = torch.chunk(grouped, 3, dim=0)
        elif self.spelling == "negative_axis":
            parts = grouped.split(batch, dim=-3)
        else:
            parts = grouped.split(batch, dim=0)
        return x + torch.cat(parts, dim=1)


@pytest.mark.parametrize(
    "spelling", ["method", "keyword", "sections", "uneven_sections", "negative_axis", "chunk"]
)
@pytest.mark.parametrize("retain_trace", [False, True])
def test_dynamic_split_grouping_matches_values_and_gradients(
    spelling: str, retain_trace: bool
) -> None:
    """B=1/2/4/8 preserve sample membership and backward across recuts."""
    model = Grouped(spelling)
    graph = prepare_graph(
        model,
        torch.randn(1, 6, 8, dtype=torch.float64),
        features=SplitFeatures(training=True, retain_trace=retain_trace),
    )
    assert graph.batch_validation["status"] == "passed", graph.batch_validation
    runtime = graph.materialize(graph.analyze(GraphPartition(frozenset())))
    for batch in (1, 2, 4, 8):
        runtime = runtime.repartition(GraphPartition(frozenset()))
        x = torch.randn(batch, 6, 8, dtype=torch.float64, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        expected = model(expected_x)
        expected.square().mean().backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(
            boundary, None, loss_fn=lambda out, _: out.square().mean()
        )
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected.square().mean())
        torch.testing.assert_close(x.grad, expected_x.grad)
        torch.testing.assert_close(runtime.replay(x.detach()), expected.detach())


def test_dynamic_split_does_not_authorize_a_changed_scalar_branch() -> None:
    """Operand lowering must not replace unrelated batch-valued integers."""

    class Branch(Grouped):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Keep topology and split semantics but change a scalar by batch."""
            return super().forward(x) * (1 if x.shape[0] == 1 else 2)

    graph = prepare_graph(Branch("method"), torch.randn(1, 6, 8))
    assert graph.batch_validation["status"] == "failed"
    assert "numeric mismatch" in graph.batch_validation["reason"]
