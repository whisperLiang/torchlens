"""Capability evidence distinguishes static checks from numerical experiments."""

from __future__ import annotations

import json
from dataclasses import replace

import torch
from dag_helpers import call_ids, close_call_ids

from torchlens.split import (
    GraphPartition,
    PlacementPlan,
    SplitFeatures,
    prepare_graph,
)
from torchlens.split.program import _feature_evidence


def test_preflight_never_claims_unmeasured_training_or_placement_numerics() -> None:
    """An implemented feature is distinct from model, device, and numerical support."""

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    graph = prepare_graph(
        model, torch.randn(3, 4), features=SplitFeatures(training=True, retain_trace=False)
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"1"}), ops=frozenset({"1"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    evidence = runtime.capability_report.feature_evidence
    assert evidence is not None
    assert set(evidence) == {
        "arbitrary_dag_partition",
        "split_inference",
        "split_training",
        "compact_training",
        "dynamic_batch",
        "cross_device",
        "repartition",
        "state_migration",
        "microbatch",
        "boundary_cache",
    }
    assert evidence["compact_training"].backend_supported is True
    assert evidence["compact_training"].model_supported is True
    assert evidence["compact_training"].numerically_verified is None
    assert evidence["split_training"].numerically_verified is None
    assert evidence["cross_device"].model_supported is None
    assert evidence["cross_device"].device_supported is None
    assert evidence["arbitrary_dag_partition"].model_supported is True
    assert evidence["state_migration"].model_supported is None
    assert evidence["microbatch"].model_supported is None
    dynamic = evidence["dynamic_batch"]
    assert dynamic.numerically_verified is True
    assert "unplaced_forward_replay_only" in dynamic.evidence_scope
    assert "independent_batch_probe:B=2" in dynamic.evidence_scope
    # Inspecting a new partition establishes only its static analysis, not a
    # new numerical trial or a claim about all possible graph partitions.
    partition = GraphPartition(frozenset())
    changed = runtime.analyze(GraphPartition(frozenset())).capability_report.feature_evidence
    assert changed is not None
    assert changed["arbitrary_dag_partition"].model_supported is True
    assert changed["arbitrary_dag_partition"].numerically_verified is None
    serialized = json.loads(json.dumps(runtime.capability_report.as_dict()))
    assert serialized["feature_evidence"]["split_training"]["numerically_verified"] is None


def test_no_batch_axes_means_unknown_numerical_evidence() -> None:
    """A missing probe must not turn into a successful numerical assertion."""

    graph = prepare_graph(
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh()),
        torch.randn(3, 4),
        features=SplitFeatures(batch_axes={}),
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"0"}), ops=frozenset({"0"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    evidence = runtime.capability_report.feature_evidence
    assert evidence is not None
    assert evidence["dynamic_batch"].numerically_verified is None
    assert evidence["split_training"].backend_supported is True
    assert evidence["split_training"].model_supported is None


def test_explicit_device_availability_does_not_claim_execution_success() -> None:
    """Device discovery examines availability without constructing device replicas."""

    graph = prepare_graph(
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh()), torch.randn(3, 4)
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"0"}), ops=frozenset({"0"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    request = replace(runtime.request, placement=PlacementPlan.on("cpu"))
    evidence = _feature_evidence(runtime.adapter, runtime.trace_graph, request, True)
    assert evidence["cross_device"].device_supported is True
    assert evidence["cross_device"].numerically_verified is None
    placed = runtime.with_placement(PlacementPlan.on("cpu"))
    assert runtime.capability_report.feature_evidence["cross_device"].device_supported is None
    assert placed.capability_report.feature_evidence["cross_device"].device_supported is True
    request = replace(runtime.request, placement=PlacementPlan.across("cpu", "cuda:999999"))
    evidence = _feature_evidence(runtime.adapter, runtime.trace_graph, request, True)
    assert evidence["cross_device"].backend_supported is True
    assert evidence["cross_device"].device_supported is False
    assert evidence["split_inference"].numerically_verified is None


def test_failed_topology_probe_does_not_claim_numerical_comparison() -> None:
    """A probe that fails before replay cannot establish numerical success or failure."""

    class Branch(torch.nn.Module):
        """Change the graph across batch sizes so the topology check fails first."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run a different operation on the independent probe batch."""

            hidden = torch.relu(x)
            return torch.sin(hidden) if x.shape[0] == 1 else torch.cos(hidden)

    graph = prepare_graph(Branch(), torch.randn(3, 4))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    assert runtime.batch_validation["status"] == "failed"
    assert runtime.batch_validation["reason"].startswith("compare topology:")
    evidence = runtime.capability_report.feature_evidence
    assert evidence is not None
    assert evidence["dynamic_batch"].model_supported is False
    assert evidence["dynamic_batch"].numerically_verified is None
