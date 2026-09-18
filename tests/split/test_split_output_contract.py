"""Torch split outputs come from declared replay data, never stale-value guesses."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from v2_helpers import split_request

import torchlens as tl
from torchlens.split.adapters.torch import GeneratedSuffix
from torchlens.split.errors import SplitUnsupportedError


@pytest.mark.parametrize("retain_trace", [False, True])
def test_missing_output_parent_never_uses_historical_activation(
    retain_trace: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping an executed producer must refuse even when its saved value exists."""

    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Sigmoid())
    x = torch.randn(2, 4)
    with torch.no_grad():
        runtime = tl.split.prepare(model, x, split_request("50%", retain_trace=retain_trace))
        suffix = runtime.segments.suffix
        execute = suffix._execute_nodes
        outputs = [
            runtime.trace_graph.node_by_id[key] for key in runtime.trace_graph.output_node_ids
        ]
        assert outputs and all(node.parents for node in outputs)
        if retain_trace:
            assert all(node.op.out is not None for node in outputs)

        def drop_output_parents(overlay: dict[str, Any]) -> dict[str, Any]:
            """Simulate lost replay evidence while leaving historical capture data intact."""

            execute(overlay)
            for node in outputs:
                overlay.pop(node.canonical_id, None)
                for parent in node.parents:
                    overlay.pop(runtime.trace_graph.node_id_by_alias.get(parent, parent), None)
            return overlay

        monkeypatch.setattr(suffix, "_execute_nodes", drop_output_parents)
        with pytest.raises(SplitUnsupportedError) as error:
            runtime.replay(x)
        assert error.value.context.reason == "missing output replay value"
    if runtime.retains_trace:
        runtime.trace.cleanup()


@pytest.mark.parametrize("has_intermediate", [False, True])
def test_missing_output_records_never_guess_last_intermediate(has_intermediate: bool) -> None:
    """Neither a nonempty overlay nor an empty one defines the final-output contract."""

    with torch.no_grad():
        runtime = tl.split.prepare(torch.nn.ReLU(), torch.ones(2, 4), split_request("50%"))
    graph = replace(runtime.trace_graph, output_node_ids=())
    suffix = GeneratedSuffix(
        graph=graph,
        plan=runtime.plan,
        spec=runtime.request,
        node_ids=runtime.plan.suffix_node_ids,
        use_live_param_sources=False,
    )
    overlay = {graph.compute_nodes[-1].canonical_id: torch.ones(1, 4)} if has_intermediate else {}
    with pytest.raises(SplitUnsupportedError) as error:
        suffix._reconstruct_output(overlay)
    assert error.value.context.reason == "missing output records"


def test_declared_constant_output_keeps_its_explicit_source_value() -> None:
    """A genuine parent-less output is required state, not a historical fallback."""

    with torch.no_grad():
        runtime = tl.split.prepare(torch.nn.ReLU(), torch.ones(2, 4), split_request("50%"))
    node = runtime.trace_graph.node_by_id[runtime.trace_graph.output_node_ids[0]]
    constant = torch.tensor([7.0])
    declared = replace(node, parents=(), op=SimpleNamespace(out=constant))
    assert runtime.segments.suffix._output_leaf(declared, {}) is constant
    missing = replace(declared, op=SimpleNamespace(out=None))
    with pytest.raises(SplitUnsupportedError) as error:
        runtime.segments.suffix._output_leaf(missing, {})
    assert error.value.context.reason == "missing source value"
