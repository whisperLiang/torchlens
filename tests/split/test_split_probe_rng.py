"""Batch probes compare stochastic calls in the native B=2 random context."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import pipeline


class StochasticProbeModel(torch.nn.Module):
    """Exercise single and sequential shape-dependent random draws."""

    def __init__(self, kind: str) -> None:
        """Choose the stochastic expression."""

        super().__init__()
        self.kind = kind
        self.dropout = torch.nn.Dropout(p=0.3)
        self.second_dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retain one fixed cut before stochastic suffix operations."""

        x = torch.relu(x)
        if self.kind == "single_dropout":
            return self.dropout(x)
        if self.kind == "multiple_dropout":
            return self.second_dropout(self.dropout(x))
        first = torch.rand_like(x)
        second = torch.rand_like(x)
        return x + first + second


@pytest.mark.parametrize("kind", ["single_dropout", "multiple_dropout", "multiple_rand"])
def test_probe_matches_native_random_draws_without_mutating_capture(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only two captures run, and the retained B=1 RNG records remain untouched."""

    captures: list[tuple[int, int]] = []
    retained_states: dict[str, tuple[Any, torch.Tensor]] = {}
    original = pipeline.capture_model

    def capture(model: Any, inputs: tuple[Any, ...], spec: Any, **kwargs: Any) -> Any:
        """Retain native capture seeds and original operation RNG references."""

        result = original(model, inputs, spec, **kwargs)
        batch = int(inputs[0].shape[0])
        captures.append((batch, result.random_seed))
        if batch == 1:
            for op in result.ops:
                state = op.func_rng_states
                if state:
                    retained_states[op.label] = (state, state["torch"].clone())
        return result

    monkeypatch.setattr(pipeline, "capture_model", capture)
    runtime = tl.split.prepare(
        StochasticProbeModel(kind), torch.ones(8, 16), split_request("after:relu")
    )
    assert runtime.batch_validation["status"] == "passed", runtime.batch_validation["reason"]
    assert [batch for batch, _seed in captures] == [1, 2]
    assert captures[0][1] == captures[1][1]
    assert retained_states
    for node in runtime.trace_graph.nodes:
        if node.canonical_id in retained_states:
            original_state, saved_values = retained_states[node.canonical_id]
            assert node.op.func_rng_states is original_state
            assert torch.equal(node.op.func_rng_states["torch"], saved_values)
    for batch in (1, 2, 8):
        assert runtime.replay(torch.ones(batch, 16)).shape == (batch, 16)
    assert len(captures) == 2


def test_stochastic_probe_preserves_user_rng_states() -> None:
    """Preparation leaves Python, NumPy, Torch and initialized CUDA RNGs intact."""

    model = StochasticProbeModel("multiple_rand")
    x = torch.ones(8, 16)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    runtime = tl.split.prepare(model, x, split_request("after:relu"))
    assert runtime.batch_validation["status"] == "passed", runtime.batch_validation["reason"]
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_states:
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_states, strict=True):
            assert torch.equal(actual, expected)


def test_stochastic_probe_does_not_mask_real_scalar_branch() -> None:
    """Aligning RNG never replaces captured B=1 arguments with B=2 arguments."""

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Keep tensor topology stable while changing a scalar with B."""

            x = torch.relu(x)
            return (x + torch.rand_like(x)) * (3 if x.shape[0] >= 2 else 2)

    runtime = tl.split.prepare(Model(), torch.ones(8, 16), split_request("after:relu"))
    assert runtime.batch_validation["status"] == "failed"
    assert "numeric mismatch" in runtime.batch_validation["reason"]
