"""Native DAG plan reuse cannot substitute stale state or mutable reports."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import torch
from dag_helpers import count_dag_partitions, iter_dag_partitions
from torch import nn

from torchlens.split import context as runtime_module, prepare_graph


def test_cache_reuses_lowering_but_rechecks_state_and_isolates_reports() -> None:
    """Repeated analysis shares plans; reports and live weights stay independent."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.ones(2, 4)
    graph = prepare_graph(model, x)
    dag = graph.compute_dag()
    partition = dag.to_partition(call.call_id for call in dag.nodes[:2])
    with (
        patch.object(
            runtime_module, "lower_split_program", wraps=runtime_module.lower_split_program
        ) as lower,
        patch.object(
            graph, "_validate_recut_state_for_plan", wraps=graph._validate_recut_state_for_plan
        ) as validate,
    ):
        first = graph.analyze(partition)
        first.plan.boundary_spec.clear()
        first.capability_report.features.clear()
        second = graph.analyze(partition)
        assert second.plan.boundary_spec
        assert second.capability_report.features
        assert lower.call_count == validate.call_count == 2
        assert first.prefix_program is second.prefix_program
    with torch.no_grad():
        model[2].weight.add_(1)
    runtime = graph.materialize(second)
    torch.testing.assert_close(runtime.replay(x), model(x))
    runtime.analyze(partition)
    assert graph.analysis_cache_info()["hits"] == 2
    runtime.clear_analysis_cache()
    assert graph.analysis_cache_info()["size"] == 0


def test_cache_is_bounded_and_not_shared_between_captures() -> None:
    """Every DAG assignment participates in eviction; captures own separate caches."""
    model = nn.Sequential(*(nn.ReLU() for _ in range(35)))
    x = torch.randn(2, 4)
    first, second = prepare_graph(model, x), prepare_graph(model, x)
    dag = first.compute_dag()
    for partition in iter_dag_partitions(dag):
        first.analyze(partition)
    assert first.analysis_cache_info()["size"] == 32
    assert first.analysis_cache_info()["misses"] == count_dag_partitions(dag) == 36
    assert second.analysis_cache_info()["size"] == 0


def test_replaced_graph_does_not_poison_sibling_profile_cache() -> None:
    """A recut that replaces its graph cannot change another runtime's costs."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    seed = prepare_graph(model, torch.ones(2, 4))
    dag = seed.compute_dag()
    partition = dag.to_partition(call.call_id for call in dag.nodes[:2])
    child = seed.materialize(seed.analyze(partition))
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
