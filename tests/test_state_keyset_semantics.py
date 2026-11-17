"""Renamed semantic state survives every supported persistence and copy door."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._state_adapter import state_items


def _semantic_state(trace: tl.Trace) -> dict[str, Any]:
    """Read changed fields as values, including real intervention evidence."""
    ops = []
    for op in trace._layer_list:
        state = dict(state_items(op))
        assert (
            not {
                "intervention_replaced",
                "interventions",
                "label_short",
                "layer_label_short",
                "pass_index",
            }
            & state.keys()
        )
        assert op.source_trace is trace
        ops.append(
            tuple(
                state[key]
                for key in (
                    "label",
                    "short_label",
                    "short_layer_label",
                    "is_edited",
                    "pass_number",
                    "episode_step",
                    "injection_provenance",
                )
            )
            + (len(state["edits"]),)
        )
    layers = []
    for layer in trace.layers:
        state = dict(state_items(layer))
        assert "intervention_replaced" not in state and "layer_label_short" not in state
        assert layer.source_trace is trace
        layers.append((layer.label, state["short_layer_label"], state["is_edited"]))
    calls = []
    for call in trace.module_calls:
        state = dict(state_items(call))
        assert "call_label" not in state and "_source_trace_strong" not in state
        assert call._source_trace is trace
        calls.append((state["label"], call.label, tuple(call.ops)))
    return {"ops": ops, "layers": layers, "calls": calls}


@pytest.mark.parametrize("stage", ["live", "pickle", "fork", "load"])
def test_renamed_state_and_edit_evidence_survive(stage: str, tmp_path: Path) -> None:
    """State renames preserve labels, ownership and nonempty user-edit history."""
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    trace = tl.trace(model, torch.ones(1, 3), intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    expected = _semantic_state(trace)
    assert any(row[3] and row[-1] for row in expected["ops"])
    if stage == "pickle":
        staged = pickle.loads(pickle.dumps(trace))
    elif stage == "fork":
        staged = trace.fork()
    elif stage == "load":
        path = tmp_path / "trace.tlspec"
        trace.save(path)
        staged = tl.load(path)
    else:
        staged = trace
    assert _semantic_state(staged) == expected
    assert torch.equal(staged.output_ops[0].out, trace.output_ops[0].out)
