"""Planner and frontier tests for split runtime."""

from __future__ import annotations

from dataclasses import replace

import pytest
from v2_helpers import (
    plan_fixture,
    split_request,
)

from torchlens.split.errors import SplitRequestError
from torchlens.split.graph import SplitTraceGraph, SplitTraceNode


def _node(
    label: str,
    *,
    canonical_id: str | None = None,
    parents: tuple[str, ...] = (),
    module_path: str | None = None,
    is_input: bool = False,
    is_output: bool = False,
    is_buffer: bool = False,
    raw_index: int = 0,
) -> SplitTraceNode:
    """Create a minimal split graph node."""

    return SplitTraceNode(
        label=label,
        raw_label=f"raw_{label}",
        canonical_id=label if canonical_id is None else canonical_id,
        backend="torch",
        raw_index=raw_index,
        op_type=label.split("_", 1)[0],
        target=None,
        func_call_id=raw_index,
        args_template=None,
        kwargs_template=None,
        parents=parents,
        children=(),
        output_ref=None,
        module_path=module_path,
        output_shape=(2, 3),
        symbolic_output_shape=None,
        dtype="torch.float32",
        requires_grad=True,
        output_container_path=(),
        output_container_spec=None,
        is_input=is_input,
        is_output=is_output,
        is_buffer=is_buffer,
        is_buffer_only_source=is_buffer and not parents,
        is_param_source=False,
        param_refs=(),
        replay_source_policy="constant",
        op=object(),
    )


def _graph(extra: tuple[SplitTraceNode, ...] = ()) -> SplitTraceGraph:
    """Create a small graph with a residual frontier."""

    nodes = (
        _node("input_1_1", is_input=True, raw_index=0),
        _node("linear_1_1", parents=("input_1_1",), module_path="proj", raw_index=1),
        _node("relu_1_1", parents=("linear_1_1",), module_path="relu", raw_index=2),
        _node("add_1_1", parents=("relu_1_1", "linear_1_1"), raw_index=3),
        _node("output_1_1", parents=("add_1_1",), is_output=True, raw_index=4),
        *extra,
    )
    return SplitTraceGraph(
        backend="torch",
        nodes=nodes,
        input_node_ids=("input_1_1",),
        output_node_ids=("output_1_1",),
        graph_shape_hash="abc",
        traced_batch_size=2,
    )


def test_after_boundary_frontier_includes_skip() -> None:
    """Splitting after ReLU includes both primary and residual skip tensors."""

    plan = plan_fixture(_graph(), split_request("through:relu"))

    assert "relu_1_1" in plan.prefix_node_ids
    assert set(plan.boundary_node_ids) == {"relu_1_1", "linear_1_1"}
    roles = {item.label: item.role for item in plan.boundary_spec.values()}
    assert roles["relu_1_1"] == "activation"
    assert roles["linear_1_1"] == "activation"


def test_before_boundary_uses_direct_parents_as_primary() -> None:
    """until: targets pass direct parents across the boundary."""

    plan = plan_fixture(_graph(), split_request("until:add_1_1"))

    roles = {item.label: item.role for item in plan.boundary_spec.values()}
    assert roles["relu_1_1"] == "activation"
    assert roles["linear_1_1"] == "activation"


def test_container_frontier_binds_schema_keys_to_canonical_values() -> None:
    """Structured output paths belong to the ABI key, not the graph value identity."""

    graph = _graph()
    graph = replace(
        graph,
        nodes=tuple(
            replace(node, output_container_path=("features", 1))
            if node.canonical_id == "linear_1_1"
            else node
            for node in graph.nodes
        ),
    )
    plan = plan_fixture(graph, split_request("through:relu"))

    assert plan.boundary_bindings == {
        "linear_1_1@'features'.1": "linear_1_1",
        "relu_1_1": "relu_1_1",
    }
    assert tuple(plan.boundary_spec) == tuple(plan.boundary_bindings)
    for key, value_id in plan.boundary_bindings.items():
        schema = plan.boundary_spec[key]
        assert schema.value_id == key
        assert schema.container_path == graph.node_by_id[value_id].output_container_path


def test_frontier_refuses_colliding_container_keys() -> None:
    """Distinct graph values must never silently overwrite a transport binding."""

    graph = SplitTraceGraph(
        backend="torch",
        nodes=(
            _node("input", is_input=True),
            replace(_node("value", parents=("input",), raw_index=1), output_container_path=(0,)),
            _node("value@0", parents=("input",), raw_index=2),
            _node("merge", parents=("value", "value@0"), raw_index=3),
            _node("output", parents=("merge",), is_output=True, raw_index=4),
        ),
        input_node_ids=("input",),
        output_node_ids=("output",),
        graph_shape_hash="container-key-collision",
        traced_batch_size=2,
    )

    with pytest.raises(SplitRequestError, match="duplicate canonical value bindings"):
        plan_fixture(graph, split_request("until:merge"))


def test_percent_boundary_selects_eligible_compute_node() -> None:
    """Percent split indexes eligible compute nodes only."""

    plan = plan_fixture(_graph(), split_request("50%"))

    assert "relu_1_1" in plan.prefix_node_ids


def test_after_terminal_compute_without_output_node_keeps_boundary() -> None:
    """Backend traces without explicit output nodes still expose terminal split output."""

    graph = SplitTraceGraph(
        backend="tf",
        nodes=(
            _node("input_1", is_input=True, raw_index=0),
            _node("dense_1", parents=("input_1",), raw_index=1),
        ),
        input_node_ids=("input_1",),
        output_node_ids=(),
        graph_shape_hash="abc",
        traced_batch_size=2,
    )

    plan = plan_fixture(graph, split_request("through:dense_1", backend="tf"))

    assert plan.suffix_node_ids == frozenset()
    assert plan.boundary_node_ids == ("dense_1",)


def test_before_first_compute_keeps_input_when_call_group_id_is_shared() -> None:
    """Input placeholders sharing a backend call ID are not part of the compute group."""

    graph = SplitTraceGraph(
        backend="paddle",
        nodes=(
            _node("input_1", is_input=True, raw_index=1),
            _node("linear_1", parents=("input_1",), raw_index=1),
            _node("output_1", parents=("linear_1",), is_output=True, raw_index=2),
        ),
        input_node_ids=("input_1",),
        output_node_ids=("output_1",),
        graph_shape_hash="abc",
        traced_batch_size=2,
    )

    plan = plan_fixture(graph, split_request("until:linear_1", backend="paddle"))

    assert "input_1" in plan.prefix_node_ids
    assert plan.boundary_node_ids == ("input_1",)


def test_invalid_canonical_partition_ids() -> None:
    """Missing, ambiguous display, input, output and buffer IDs all refuse typed."""
    from torchlens.split import GraphPartition, SplitRequest
    from torchlens.split.errors import SplitUnsupportedError
    from torchlens.split.planner import plan_partition

    graph = _graph((_node("buffer_1_1", is_buffer=True, raw_index=5),))
    for node_id in ("missing", "raw_", "input_1_1", "output_1_1", "buffer_1_1"):
        with pytest.raises(SplitUnsupportedError) as caught:
            plan_partition(graph, SplitRequest(GraphPartition(frozenset({node_id}))))
        assert caught.value.context.reason == "partition_node_invalid"


def test_repeated_layer_labels_require_canonical_target() -> None:
    """Pass-qualified canonical IDs keep repeated layer labels distinct."""

    nodes = (
        _node("input_1", is_input=True, raw_index=0),
        _node(
            "relu_1_1",
            canonical_id="relu_1_1:1",
            parents=("input_1",),
            raw_index=1,
        ),
        _node(
            "relu_1_1",
            canonical_id="relu_1_1:2",
            parents=("relu_1_1:1",),
            raw_index=2,
        ),
        _node("output_1", parents=("relu_1_1:2",), is_output=True, raw_index=3),
    )
    graph = SplitTraceGraph(
        backend="torch",
        nodes=nodes,
        input_node_ids=("input_1",),
        output_node_ids=("output_1",),
        graph_shape_hash="abc",
        traced_batch_size=2,
    )

    with pytest.raises(SplitRequestError, match="ambiguous"):
        plan_fixture(graph, split_request("through:relu_1_1"))

    plan = plan_fixture(graph, split_request("through:relu_1_1:2"))

    assert "relu_1_1:2" in plan.prefix_node_ids
    assert plan.boundary_node_ids == ("relu_1_1:2",)
