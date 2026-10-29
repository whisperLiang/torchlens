"""Bounded plan reuse never substitutes stale model state or mutable reports."""

from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn

from torchlens.split import SplitRequest, after, before, prepare, runtime as runtime_module


def test_cache_reuses_lowering_but_rechecks_state_and_isolates_reports() -> None:
    """Repeated analysis shares plans only; caller edits cannot poison later hits."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.ones(2, 4)
    runtime = prepare(model, x, SplitRequest(after("1")))
    with (
        patch.object(
            runtime_module, "lower_split_program", wraps=runtime_module.lower_split_program
        ) as lower,
        patch.object(
            runtime, "_validate_recut_state_for_plan", wraps=runtime._validate_recut_state_for_plan
        ) as validate,
    ):
        first = runtime.analyze(before("2"))
        first.plan.boundary_spec.clear()
        first.capability_report.features.clear()
        second = runtime.analyze(before("2"))
        assert second.plan.boundary_spec
        assert second.capability_report.features
        assert lower.call_count == 2
        assert validate.call_count == 2
        assert first.prefix_program is second.prefix_program
    with torch.no_grad():
        model[2].weight.add_(1)
    recut = runtime.materialize(second)
    torch.testing.assert_close(recut.replay(x), model(x))
    recut.analyze(before("2"))
    assert runtime.analysis_cache_info()["hits"] == 2
    recut.clear_analysis_cache()
    assert runtime.analysis_cache_info()["size"] == 0


def test_cache_is_bounded_and_not_shared_between_captures() -> None:
    """Each unrelated client starts empty and a rotating workload evicts plans."""
    model = nn.Sequential(*(nn.ReLU() for _ in range(35)))
    x = torch.randn(2, 4)
    first = prepare(model, x, SplitRequest(after("0")))
    second = prepare(model, x, SplitRequest(after("0")))
    for index in range(35):
        first.analyze(after(str(index)))
    assert first.analysis_cache_info()["size"] == 32
    assert first.analysis_cache_info()["misses"] == 35
    assert second.analysis_cache_info()["size"] == 0
