"""Candidate diagnosis stays separate from executable segment construction."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import (
    PlacementPlan,
    SplitPointAnalysis,
    pipeline as pipeline_module,
    runtime as runtime_module,
)
from torchlens.split.state import SegmentState


class PointModel(nn.Module):
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


def test_candidate_diagnosis_never_builds_segments() -> None:
    """Only the selected point is materialized after candidate analysis."""

    x = torch.ones(2, 4)
    runtime = tl.split.prepare(PointModel().eval(), x, split_request("after:relu"))
    with (
        patch.object(
            runtime.adapter, "build_segments", wraps=runtime.adapter.build_segments
        ) as build,
        patch.object(
            runtime_module, "execute_split_runtime", wraps=runtime_module.execute_split_runtime
        ) as execute,
        patch.object(
            pipeline_module,
            "capture_canonical_model",
            side_effect=AssertionError("candidate analysis must reuse the capture"),
        ),
    ):
        report = runtime.split_points(diagnose=True)
        assert report.total == 2 * len(runtime.trace_graph.compute_nodes)
        assert build.call_count == execute.call_count == 0
        selected = report.supported[0].point
        runtime.at(selected)
        assert build.call_count == execute.call_count == 1


@pytest.mark.parametrize("frozen", [False, True])
def test_point_analysis_does_not_bind_lazy_owned_state(
    frozen: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostic inspection leaves later first-use state binding live."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = PointModel().eval().requires_grad_(not frozen)
    x = torch.ones(2, 4)
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:relu", placement=PlacementPlan.on("cpu"), live_param_sources=True),
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
    runtime.split_points(diagnose=True)
    runtime.analyze(tl.split.before("last"))
    assert tuple(state._entries for state in states) == entries_before
    for state in states:
        if state._replica_pool is not None:
            assert state._replica_pool._replicas == pools_before[id(state._replica_pool)]

    with torch.no_grad():
        model.last.weight.add_(1)
    torch.testing.assert_close(runtime.replay(x), model(x))


def test_analysis_and_at_materialize_equivalent_runtimes() -> None:
    """Explicit analysis and ``at`` share the same plan and replay contract."""

    model = PointModel().eval()
    x = torch.ones(2, 4)
    seed = tl.split.prepare(model, x, split_request("after:relu", trainable=True))
    point = tl.split.before("last")
    analysis = seed.analyze(point)
    assert isinstance(analysis, SplitPointAnalysis)
    assert analysis.source_graph is seed.trace_graph
    assert analysis.graph_ir.values is seed.graph_ir.values
    explicit = seed.materialize(analysis)
    implicit = seed.at(point)
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
    first = tl.split.prepare(PointModel().eval(), x, split_request("after:relu"))
    second = tl.split.prepare(PointModel().eval(), x, split_request("after:relu"))
    with pytest.raises(ValueError, match="another captured graph"):
        second.materialize(first.analyze(tl.split.before("last")))
