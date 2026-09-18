"""Last-use schedules include references carried only by nested replay templates."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchlens.intervention.types import CapturedArgTemplate
from torchlens.split._torch_liveness import release_schedule
from torchlens.split.graph import ReplayValueRef, SplitTraceGraph, SplitTraceNode


def _node(node_id: str, *, parents: tuple[str, ...] = ()) -> SplitTraceNode:
    """Build a minimal tensor-producing node without running a capture."""

    return SplitTraceNode(
        label=node_id,
        raw_label=None,
        canonical_id=node_id,
        backend="torch",
        raw_index=None,
        op_type="sin",
        target=torch.sin,
        func_call_id=None,
        args_template=None,
        kwargs_template=None,
        parents=parents,
        children=(),
        output_ref=None,
        module_path=None,
        output_shape=(2, 4),
        symbolic_output_shape=None,
        dtype="torch.float32",
        requires_grad=False,
        output_container_path=(),
        output_container_spec=None,
        is_input=False,
        is_output=False,
        is_buffer=False,
        is_buffer_only_source=False,
        is_param_source=False,
        param_refs=(),
        replay_source_policy="constant",
        op=None,
    )


@pytest.mark.parametrize("template_slot", ["args_template", "kwargs_template"])
def test_nested_template_only_dependency_survives_until_its_consumer(template_slot: str) -> None:
    """An absent metadata edge must not release a value before its nested use."""

    producer = _node("producer")
    intermediate = _node("intermediate", parents=(producer.canonical_id,))
    consumer = _node("consumer", parents=(intermediate.canonical_id,))
    nested = ({"residual": [(ReplayValueRef(producer.canonical_id),)]},)
    template = (
        CapturedArgTemplate(args=(nested,))
        if template_slot == "args_template"
        else {"payload": nested}
    )
    consumer = replace(consumer, **{template_slot: template})
    graph = SplitTraceGraph(
        backend="torch",
        nodes=(producer, intermediate, consumer),
        input_node_ids=(),
        output_node_ids=(consumer.canonical_id,),
        graph_shape_hash=None,
        traced_batch_size=None,
    )
    node_ids = frozenset(node.canonical_id for node in graph.nodes)
    retained = frozenset(graph.output_node_ids)
    schedule = release_schedule(graph, node_ids, retained)
    assert producer.canonical_id not in consumer.parents
    assert producer.canonical_id not in schedule.get(intermediate.canonical_id, ())
    assert set(schedule[consumer.canonical_id]) == {
        producer.canonical_id,
        intermediate.canonical_id,
    }
    assert all(consumer.canonical_id not in values for values in schedule.values())

    # Removing the only late-use evidence must move the release earlier. This
    # control makes an implementation that ignores nested templates fail above.
    without_nested_use = replace(consumer, args_template=None, kwargs_template=None)
    control = replace(graph, nodes=(producer, intermediate, without_nested_use))
    assert (
        producer.canonical_id
        in release_schedule(control, node_ids, retained)[intermediate.canonical_id]
    )
