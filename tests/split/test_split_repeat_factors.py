"""Dynamic repeat counts are distinct from repeated tensor output dimensions."""

from __future__ import annotations

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from v2_helpers import (
    prepare_fixture,
    repartition_fixture,
    request_fixture,
    select_node_prefix,
    select_prefix,
)

from torchlens.split import SplitFeatures
from torchlens.split.errors import SplitBoundaryError


class LearnedQueries(nn.Module):
    """Replicate fixed learned state over the input batch like decoder queries."""

    def __init__(self, spelling: str) -> None:
        """Keep non-singleton source extents to test factor-versus-extent lowering."""

        super().__init__()
        self.queries = nn.Parameter(torch.randn(2, 4))
        self.projection = nn.Linear(4, 4)
        self.spelling = spelling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat the two query rows B times and combine them with both input rows."""

        if self.spelling == "varargs":
            queries = self.queries.repeat(x.shape[0], 1)
        elif self.spelling == "tuple":
            queries = self.queries.repeat((x.shape[0], 1))
        elif self.spelling == "tile":
            queries = torch.tile(self.queries, (x.shape[0], 1))
        elif self.spelling == "keyword_repeat":
            queries = self.queries.repeat(repeats=(x.shape[0], 1))
        elif self.spelling == "keyword_tile":
            queries = torch.tile(input=self.queries, dims=(x.shape[0], 1))
        elif self.spelling == "tile_short":
            queries = torch.tile(self.queries.unsqueeze(0), (x.shape[0], 1)).flatten(0, 1)
        elif self.spelling == "view":
            queries = self.queries.unsqueeze(0).repeat(x.shape[0], 1, 1).flatten(0, 1)
        else:
            queries = self.queries.repeat(x.shape[0], 1, 1).flatten(0, 1)
        return self.projection(x.flatten(0, 1) + queries).tanh()


@pytest.mark.parametrize(
    "spelling",
    [
        "varargs",
        "tuple",
        "tile",
        "leading_axis",
        "view",
        "tile_short",
        "keyword_repeat",
        "keyword_tile",
    ],
)
@pytest.mark.parametrize("retain_trace", [False, True])
def test_learned_query_repeat_training_across_batches_and_recuts(
    spelling: str, retain_trace: bool
) -> None:
    """Batch-dependent factors preserve gradients, Adam state, and graph reuse."""

    torch.manual_seed(913)
    model = LearnedQueries(spelling).eval()
    reference = deepcopy(model)
    runtime = prepare_fixture(
        model,
        torch.randn(1, 2, 4),
        request_fixture(
            partial(select_node_prefix, node_id="projection"),
            features=SplitFeatures(training=True, retain_trace=retain_trace),
        ),
    )
    assert runtime.batch_validation["status"] == "passed"
    program = runtime.trace_graph.shape_program
    assert program is not None
    repeat = next(
        node for node in runtime.trace_graph.compute_nodes if node.op_type in {"repeat", "tile"}
    )
    assert program.proof_sources[repeat.canonical_id] == "sampled_shape_witness"
    assert program.recipes[repeat.canonical_id][0].recipe_id.startswith("repeat:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.002)
    for batch in (1, 2, 4, 8):
        x = torch.randn(batch, 2, 4, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(True)
        target = torch.randn(2 * batch, 4)
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        expected_loss = nn.functional.mse_loss(reference(expected_x), target)
        expected_loss.backward()
        boundary = runtime.run_training_prefix(x)
        loss, gradients = runtime.train_suffix(boundary, target)
        runtime.backward_prefix(boundary, gradients)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(x.grad, expected_x.grad)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(optimizer.state[actual], expected_optimizer.state[expected])
        graph = runtime.trace_graph
        point = (
            partial(select_node_prefix, node_id=str(repeat.canonical_id))
            if batch % 2
            else partial(select_node_prefix, node_id="projection")
        )
        runtime = repartition_fixture(runtime, point)
        assert runtime.trace_graph is graph


def test_fixed_repeat_of_dynamic_receiver_keeps_its_factor() -> None:
    """A fixed count equal to a tested batch remains fixed when the input changes."""

    class FixedRepeat(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Expand a dynamic receiver using constant repeat factors."""

            return x.repeat(2, 1).sin()

    model = FixedRepeat().eval()
    runtime = prepare_fixture(
        model,
        torch.randn(1, 4),
        request_fixture(
            lambda g, count=50: select_prefix(g, (len(g.compute_nodes) - 1) * count // 100 + 1)
        ),
    )
    program = runtime.trace_graph.shape_program
    assert program is not None
    repeat = next(node for node in runtime.trace_graph.compute_nodes if node.op_type == "repeat")
    assert repeat.canonical_id not in program.recipes
    assert program.proof_sources[repeat.canonical_id] == "semantic:repeat"
    for batch in (1, 2, 4, 8):
        x = torch.randn(batch, 4)
        torch.testing.assert_close(runtime.replay(x), model(x))


def test_repeat_with_dynamic_receiver_and_factor_refuses_unproven_batches() -> None:
    """A nonlinear B-squared extent is not silently treated as a fixed repeat."""

    class NonlinearRepeat(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Vary the repeat factor and the receiver extent simultaneously."""

            return x.repeat(x.shape[0], 1).sin()

    model = NonlinearRepeat().eval()
    runtime = prepare_fixture(
        model,
        torch.randn(1, 4),
        request_fixture(
            lambda g, count=50: select_prefix(g, (len(g.compute_nodes) - 1) * count // 100 + 1)
        ),
    )
    assert runtime.batch_validation["status"] == "failed"
    x = torch.randn(1, 4)
    torch.testing.assert_close(runtime.replay(x), model(x))
    for batch in (2, 4):
        with pytest.raises(SplitBoundaryError, match="probe did not pass"):
            runtime.replay(torch.randn(batch, 4))
