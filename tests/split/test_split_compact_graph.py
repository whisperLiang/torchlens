"""Compact Torch graphs retain executable state without diagnostic activations."""

from __future__ import annotations

import gc
import weakref
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from v2_helpers import split_request

from torchlens.ir.refs import TensorRef
from torchlens.split._torch_compact import compact_torch_graph
from torchlens.split.adapters.torch import TorchSplitAdapter
from torchlens.split.graph import SplitTraceGraph, split_graph_from_trace
from torchlens.split.pipeline import capture_model
from torchlens.split.planner import plan_split


class StatefulBranches(torch.nn.Module):
    """Mix tied parameters, buffers, aliased multi-output views and output trees."""

    def __init__(self) -> None:
        """Create replay state whose live identities must not change."""

        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return early and late values that depend on shared linear weights."""

        early = self.linear(x)
        normalized = self.norm(early)
        left, right = normalized.chunk(2, dim=1)
        return {"early": early, "late": self.linear(torch.cat((right, left), dim=1))}


def _capture_compact(model: torch.nn.Module, x: torch.Tensor) -> tuple[SplitTraceGraph, Any]:
    """Return compact execution metadata beside its untouched source capture."""

    trace = capture_model(model, (x,), split_request("50%", batch_axes={}))
    return compact_torch_graph(split_graph_from_trace(trace)), trace


@pytest.mark.parametrize("backend", ["tf", "jax", "fake"])
def test_compact_graph_rejects_non_torch_backends(backend: str) -> None:
    """The Torch-specific compactor refuses foreign graphs before inspecting nodes."""

    graph = SplitTraceGraph(
        backend=backend,
        nodes=(),
        input_node_ids=(),
        output_node_ids=(),
        graph_shape_hash=None,
        traced_batch_size=None,
    )
    with pytest.raises(ValueError, match="requires a normalized Torch graph"):
        compact_torch_graph(graph)


def test_compact_graph_releases_diagnostic_payloads_and_capture_objects() -> None:
    """The executable graph does not keep the discarded capture product alive."""

    model = StatefulBranches().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        graph, trace = _capture_compact(model, x)
    assert all(
        {field.name for field in fields(node.op)}
        == {"func_rng_states", "func_autocast_state", "multi_output_index", "out"}
        for node in graph.nodes
    )
    assert [node.module_path for node in graph.nodes] == [
        node.module_path for node in split_graph_from_trace(trace).nodes
    ]
    trace_ref = weakref.ref(trace)
    activation_refs = [
        weakref.ref(op.out)
        for op in trace
        if not (op.is_input or op.is_buffer) and isinstance(op.out, torch.Tensor)
    ]
    assert activation_refs
    assert all(node.op.out is None for node in graph.nodes if not node.is_buffer)
    assert all(
        node.output_ref.payload is None
        for node in graph.nodes
        if isinstance(node.output_ref, TensorRef)
    )
    trace.cleanup()
    del trace
    gc.collect()
    assert trace_ref() is None
    assert all(ref() is None for ref in activation_refs)


@pytest.mark.parametrize("live_sources", [False, True])
def test_compact_graph_replays_every_cut_after_original_capture_cleanup(
    live_sources: bool,
) -> None:
    """Captured and live state policies survive cleanup and preserve output trees."""

    model = StatefulBranches().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        graph, trace = _capture_compact(model, x)
        trace.cleanup()
        expected = model(x)
        for node in graph.compute_nodes:
            for location in ("before", "after"):
                spec = split_request(
                    f"{location}:{node.canonical_id}",
                    live_param_sources=live_sources,
                    batch_axes={},
                )
                segments = TorchSplitAdapter().build_segments(graph, plan_split(graph, spec), spec)
                actual = segments.suffix(segments.prefix(x, detach_boundary=True))
                assert actual.keys() == expected.keys()
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key])


def test_compact_graph_preserves_live_parameter_buffer_and_gradient_identity() -> None:
    """State snapshots are handle references, not detached or copied tensors."""

    model = StatefulBranches().eval()
    x = torch.randn(2, 4, requires_grad=True)
    graph, trace = _capture_compact(model, x)
    param_ids = {id(param) for param in model.parameters()}
    handles = [param.handle for node in graph.nodes for param in node.param_refs]
    assert handles and {id(handle) for handle in handles} == param_ids
    assert all(handle.requires_grad for handle in handles)
    buffer_ids = {id(buffer) for buffer in model.buffers()}
    buffer_handles = [buffer.handle for node in graph.nodes for buffer in node.buffer_refs]
    assert buffer_handles and {id(handle) for handle in buffer_handles} <= buffer_ids
    trace.cleanup()
    spec = split_request("50%", live_param_sources=True, batch_axes={})
    segments = TorchSplitAdapter().build_segments(graph, plan_split(graph, spec), spec)
    actual = segments.suffix(segments.training_prefix(x, detach_boundary=False))
    expected = model(x)
    parameters = tuple(model.parameters())
    actual_grads = torch.autograd.grad(
        sum(value.sum() for value in actual.values()), (x, *parameters)
    )
    expected_grads = torch.autograd.grad(
        sum(value.sum() for value in expected.values()), (x, *parameters)
    )
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right)


def test_compact_graph_retains_only_required_source_and_constant_output_values() -> None:
    """Source constants survive while executable ordinary outputs are discarded."""

    with torch.no_grad():
        graph, trace = _capture_compact(torch.nn.ReLU(), torch.randn(2, 4))
    template = graph.compute_nodes[0]
    value = torch.tensor([7.0])
    source = replace(
        template,
        canonical_id="constant",
        target=None,
        parents=(),
        op=SimpleNamespace(out=value),
    )
    output = replace(source, canonical_id="constant_output", is_output=True)
    compact = compact_torch_graph(replace(graph, nodes=(source, output)))
    assert compact.nodes[0].op.out is value
    assert compact.nodes[1].op.out is value
    trace.cleanup()
