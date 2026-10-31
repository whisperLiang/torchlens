"""Bounded plan reuse never substitutes stale model state or mutable reports."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from unittest.mock import patch

import torch
from torch import nn
from v2_helpers import (
    analyze_fixture,
    prepare_fixture,
    repartition_fixture,
    request_fixture,
    select_node_prefix,
)

from torchlens.split import context as runtime_module


def test_cache_reuses_lowering_but_rechecks_state_and_isolates_reports() -> None:
    """Repeated analysis shares plans only; caller edits cannot poison later hits."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.ones(2, 4)
    runtime = prepare_fixture(model, x, request_fixture(partial(select_node_prefix, node_id="1")))
    runtime.clear_analysis_cache()
    with (
        patch.object(
            runtime_module, "lower_split_program", wraps=runtime_module.lower_split_program
        ) as lower,
        patch.object(
            runtime, "_validate_recut_state_for_plan", wraps=runtime._validate_recut_state_for_plan
        ) as validate,
    ):
        first = analyze_fixture(runtime, partial(select_node_prefix, node_id="2", include=False))
        first.plan.boundary_spec.clear()
        first.capability_report.features.clear()
        second = analyze_fixture(runtime, partial(select_node_prefix, node_id="2", include=False))
        assert second.plan.boundary_spec
        assert second.capability_report.features
        assert lower.call_count == 2
        assert validate.call_count == 2
        assert first.prefix_program is second.prefix_program
    with torch.no_grad():
        model[2].weight.add_(1)
    recut = runtime.materialize(second)
    torch.testing.assert_close(recut.replay(x), model(x))
    analyze_fixture(recut, partial(select_node_prefix, node_id="2", include=False))
    assert runtime.analysis_cache_info()["hits"] == 2
    recut.clear_analysis_cache()
    assert runtime.analysis_cache_info()["size"] == 0


def test_cache_is_bounded_and_not_shared_between_captures() -> None:
    """Each unrelated client starts empty and a rotating workload evicts plans."""
    model = nn.Sequential(*(nn.ReLU() for _ in range(35)))
    x = torch.randn(2, 4)
    first = prepare_fixture(model, x, request_fixture(partial(select_node_prefix, node_id="0")))
    second = prepare_fixture(model, x, request_fixture(partial(select_node_prefix, node_id="0")))
    for index in range(35):
        analyze_fixture(first, partial(select_node_prefix, node_id=str(index)))
    assert first.analysis_cache_info()["size"] == 32
    assert first.analysis_cache_info()["misses"] == 35
    assert second.analysis_cache_info()["size"] == 1


def test_replaced_graph_does_not_poison_sibling_profile_cache() -> None:
    """A recut that replaces its graph cannot change another runtime's costs."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    seed = prepare_fixture(
        model, torch.ones(2, 4), request_fixture(partial(select_node_prefix, node_id="1"))
    )
    child = repartition_fixture(seed, partial(select_node_prefix, node_id="2", include=False))
    original = seed.graph_profile()
    child.trace_graph = replace(
        child.trace_graph,
        nodes=tuple(replace(node, dtype="float64") for node in child.trace_graph.nodes),
    )
    changed = child.graph_profile()
    assert changed.tensors[0].bytes == 2 * original.tensors[0].bytes
    assert seed.graph_profile().tensors[0].bytes == original.tensors[0].bytes
    assert seed.analysis_cache_info()["size"] == 1
    assert child.analysis_cache_info()["size"] == 0
