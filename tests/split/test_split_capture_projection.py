"""Exercise the upstream 3 capture boundary through public Split execution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.split import SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError

pytestmark = pytest.mark.usefixtures("fp64_functional")


@dataclass
class Output:
    """A nested output whose typed paths must survive generated replay."""

    values: dict[str, torch.Tensor]
    maximum: tuple[torch.Tensor, torch.Tensor]


class NestedOutputs(nn.Module):
    """Return shared leaves and both outputs of one native reduction."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Output:
        """Preserve native multi-output identity and a repeated output tensor."""

        hidden = self.linear(x).sin()
        maximum = hidden.max(dim=-1)
        return Output({"hidden": hidden, "same": hidden}, maximum)


@pytest.mark.smoke
@pytest.mark.parametrize("retain_trace", [False, True])
def test_nested_paths_and_live_parameters_execute_without_forward(
    retain_trace: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconstruct dataclass/dict/namedtuple outputs after live weight updates."""

    model = NestedOutputs()
    native = model.forward
    graph = prepare_graph(
        model, torch.ones(1, 4), features=SplitFeatures(retain_trace=retain_trace)
    )
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition([dag.nodes[0].call_id])))
    identities = tuple(call.call_id for call in dag.nodes)

    def forbidden_forward(*args: object, **kwargs: object) -> None:
        """Refuse any attempt to replace split replay with a native forward."""

        raise AssertionError("Split replay must execute its captured native calls")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    with torch.no_grad():
        model.linear.weight.add_(0.25)
    for batch in (1, 2, 4, 8):
        x = torch.randn(batch, 4)
        expected = native(x)
        actual = runtime.replay(x)
        assert type(actual) is Output
        assert type(actual.maximum) is type(expected.maximum)
        assert actual.values["hidden"] is actual.values["same"]
        torch.testing.assert_close(
            actual.values["hidden"], expected.values["hidden"], atol=1e-8, rtol=1e-7
        )
        torch.testing.assert_close(
            actual.maximum.values, expected.maximum.values, atol=1e-8, rtol=1e-7
        )
        torch.testing.assert_close(actual.maximum.indices, expected.maximum.indices, atol=0, rtol=0)
    runtime = runtime.repartition(dag.to_partition([]))
    assert tuple(call.call_id for call in runtime.compute_dag().nodes) == identities


@pytest.mark.smoke
def test_projection_keeps_upstream_public_record_contract() -> None:
    """Preparing a split leaves the sole public capture and grouped call API intact."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    capture = tl.trace(
        model,
        torch.ones(1, 4),
        capture=tl.options.CaptureOptions(replayable=True, save_args=True),
    )
    from torchlens.split.graph import split_graph_from_trace

    graph = split_graph_from_trace(capture)
    assert len(graph.input_node_ids) == len(capture.input_layers)
    assert len(graph.output_node_ids) == len(capture.output_layers)
    assert {node.func_call_id for node in graph.compute_nodes} == {
        op.call.call_id for op in capture.ops
    }
    assert not {node.canonical_id for node in graph.nodes}.intersection(
        op.label for op in capture._layer_list
    )
    assert all(node.op is not op for node, op in zip(graph.nodes, capture._layer_list, strict=True))
    capture.cleanup()


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_positional_inplace_functional_keeps_isolated_boundary_vjp(batch: int) -> None:
    """A newly visible functional write retains strict loss, parameter and input gradients."""

    from copy import deepcopy

    model = nn.Sequential(nn.Linear(4, 4), nn.Hardswish(inplace=True), nn.Linear(4, 2))
    oracle = deepcopy(model)
    graph = prepare_graph(
        model, torch.ones(1, 4), features=SplitFeatures(training=True, retain_trace=False)
    )
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition([dag.nodes[0].call_id])))
    x = torch.randn(batch, 4, requires_grad=True)
    native_x = x.detach().clone().requires_grad_(True)
    target = torch.randn(batch, 2)
    expected = nn.functional.mse_loss(oracle(native_x), target)
    expected.backward()
    boundary = runtime.run_training_prefix(x)
    original = {key: value.detach().clone() for key, value in boundary.tensors.items()}
    loss, gradients = runtime.train_suffix(boundary, target, loss_fn=nn.functional.mse_loss)
    runtime.backward_prefix(boundary, gradients)
    torch.testing.assert_close(loss, expected, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(x.grad, native_x.grad, atol=1e-8, rtol=1e-7)
    for actual, reference in zip(model.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-8, rtol=1e-7)
    for key, value in boundary.tensors.items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)


def test_positional_inplace_functional_refuses_preexisting_frontier_aliases() -> None:
    """The isolation proof cannot discard the storage relation of a second frontier view."""

    class AliasedWrite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Keep a view alive across a later native functional write."""

            value = self.linear(x)
            alias = value.view_as(value)
            return alias, nn.functional.hardswish(value, True)

    graph = prepare_graph(AliasedWrite(), torch.ones(1, 4), features=SplitFeatures(training=True))
    dag = graph.compute_dag()
    selected = [
        call.call_id
        for call in dag.nodes
        if graph.trace_graph.node_by_id[call.output_node_ids[0]].op_type != "hardswish"
    ]
    with pytest.raises(SplitUnsupportedError) as caught:
        graph.analyze(dag.to_partition(selected))
    assert caught.value.context.reason == "partition_effect_order_unproven"
