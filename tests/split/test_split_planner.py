"""Planner and frontier tests for split runtime."""

from __future__ import annotations

import pytest

from torchlens.split.errors import SplitSpecError
from torchlens.split.graph import SplitTraceGraph, SplitTraceNode
from torchlens.split.planner import plan_split
from torchlens.split.spec import SplitSpec


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

    plan = plan_split(_graph(), SplitSpec("after:relu"))

    assert plan.target_node_id == "relu_1_1"
    assert set(plan.boundary_node_ids) == {"relu_1_1", "linear_1_1"}
    roles = {item.label: item.role for item in plan.boundary_spec.values()}
    assert roles["relu_1_1"] == "primary"
    assert roles["linear_1_1"] == "skip"


def test_before_boundary_uses_direct_parents_as_primary() -> None:
    """before: targets pass direct parents across the boundary."""

    plan = plan_split(_graph(), SplitSpec("before:add_1_1"))

    roles = {item.label: item.role for item in plan.boundary_spec.values()}
    assert roles["relu_1_1"] == "primary"
    assert roles["linear_1_1"] == "primary"


def test_percent_boundary_selects_eligible_compute_node() -> None:
    """Percent split indexes eligible compute nodes only."""

    plan = plan_split(_graph(), SplitSpec("50%"))

    assert plan.target_node_id == "relu_1_1"
    assert plan.boundary_kind == "after"


def test_target_errors() -> None:
    """Missing, ambiguous, and source/sink targets raise SplitSpecError."""

    graph = _graph((_node("other_relu", module_path="other.relu", raw_index=5),))
    with pytest.raises(SplitSpecError, match="No split target"):
        plan_split(graph, SplitSpec("after:missing"))
    with pytest.raises(SplitSpecError, match="ambiguous"):
        plan_split(graph, SplitSpec("after:raw_"))
    with pytest.raises(SplitSpecError, match="input"):
        plan_split(_graph(), SplitSpec("after:input_1_1"))
    with pytest.raises(SplitSpecError, match="output"):
        plan_split(_graph(), SplitSpec("after:output_1_1"))
    with pytest.raises(SplitSpecError, match="buffer"):
        plan_split(
            _graph((_node("buffer_1_1", is_buffer=True, raw_index=5),)),
            SplitSpec("after:buffer_1_1"),
        )


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

    with pytest.raises(SplitSpecError, match="ambiguous"):
        plan_split(graph, SplitSpec("after:relu_1_1"))

    plan = plan_split(graph, SplitSpec("after:relu_1_1:2"))

    assert plan.target_node_id == "relu_1_1:2"
    assert plan.boundary_node_ids == ("relu_1_1:2",)
