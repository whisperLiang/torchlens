"""Candidate diagnosis stays separate from executable segment construction."""

from __future__ import annotations

from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch import nn
from v2_helpers import (
    analyze_fixture,
    prepare_fixture,
    repartition_fixture,
    select_node_prefix,
    split_request,
)

from torchlens.split import (
    PartitionAnalysis,
    PlacementPlan,
    context as runtime_module,
    pipeline as pipeline_module,
    prepare_graph,
)
from torchlens.split.state import SegmentState


class PartitionModel(nn.Module):
    """Small model with several distinct compute boundaries."""

    def __init__(self) -> None:
        """Build two affine operations with an activation between them."""

        super().__init__()
        self.first = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.last = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the model output."""

        return self.last(self.relu(self.first(x)))


def test_analysis_never_builds_segments_or_recaptures() -> None:
    """DAG inspection and analysis defer all executable work to materialization."""

    x = torch.ones(2, 4)
    graph = prepare_graph(PartitionModel().eval(), x)
    with (
        patch.object(graph.adapter, "build_segments", wraps=graph.adapter.build_segments) as build,
        patch.object(
            runtime_module, "execute_split_runtime", wraps=runtime_module.execute_split_runtime
        ) as execute,
        patch.object(
            pipeline_module,
            "capture_canonical_model",
            side_effect=AssertionError("analysis must reuse the capture"),
        ),
    ):
        dag = graph.compute_dag()
        selected = dag.to_partition(c.call_id for c in dag.nodes[:2])
        analysis = graph.analyze(selected)
        assert build.call_count == execute.call_count == 0
        graph.materialize(analysis)
        assert build.call_count == execute.call_count == 1


@pytest.mark.parametrize("frozen", [False, True])
def test_partition_analysis_does_not_bind_lazy_owned_state(
    frozen: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostic inspection leaves later first-use state binding live."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = PartitionModel().eval().requires_grad_(not frozen)
    x = torch.ones(2, 4)
    runtime = prepare_fixture(
        model,
        x,
        split_request(
            partial(select_node_prefix, node_id="relu"),
            placement=PlacementPlan.on("cpu"),
            live_param_sources=True,
        ),
    )
    states = (
        runtime.segments.prefix._state,
        runtime.segments.training_prefix._state,
        runtime.segments.suffix._state,
    )
    entries_before = tuple(dict(state._entries) for state in states)
    pools_before = {
        id(state._replica_pool): dict(state._replica_pool._replicas)
        for state in states
        if state._replica_pool is not None
    }
    analyze_fixture(runtime, partial(select_node_prefix, node_id="last", include=False))
    assert tuple(state._entries for state in states) == entries_before
    for state in states:
        if state._replica_pool is not None:
            assert state._replica_pool._replicas == pools_before[id(state._replica_pool)]

    with torch.no_grad():
        model.last.weight.add_(1)
    torch.testing.assert_close(runtime.replay(x), model(x))


def test_explicit_analysis_and_repartition_materialize_equivalent_runtimes() -> None:
    """Explicit analysis and ``at`` share the same plan and replay contract."""

    model = PartitionModel().eval()
    x = torch.ones(2, 4)
    seed = prepare_fixture(
        model, x, split_request(partial(select_node_prefix, node_id="relu"), trainable=True)
    )
    point = partial(select_node_prefix, node_id="last", include=False)
    analysis = analyze_fixture(seed, point)
    assert isinstance(analysis, PartitionAnalysis)
    assert analysis.source_graph is seed.trace_graph
    assert analysis.graph_ir.values is seed.graph_ir.values
    explicit = seed.materialize(analysis)
    implicit = repartition_fixture(seed, point)
    assert explicit.split_id == implicit.split_id
    assert explicit.boundary_schema == implicit.boundary_schema
    assert explicit.capability_report == implicit.capability_report
    assert explicit.capability_report.training.supported
    first = explicit.run_prefix(x)
    second = implicit.run_prefix(x)
    assert first.spec == second.spec
    for key in first.tensors:
        torch.testing.assert_close(first.tensors[key], second.tensors[key])
    torch.testing.assert_close(explicit.run_suffix(first), implicit.run_suffix(second))
    torch.testing.assert_close(explicit.replay(x), model(x))


def test_analysis_cannot_be_materialized_on_another_capture() -> None:
    """A same-shaped foreign graph cannot supply a point analysis."""

    x = torch.ones(2, 4)
    first = prepare_fixture(
        PartitionModel().eval(), x, split_request(partial(select_node_prefix, node_id="relu"))
    )
    second = prepare_fixture(
        PartitionModel().eval(), x, split_request(partial(select_node_prefix, node_id="relu"))
    )
    with pytest.raises(ValueError, match="another captured graph"):
        second.materialize(
            analyze_fixture(first, partial(select_node_prefix, node_id="last", include=False))
        )
