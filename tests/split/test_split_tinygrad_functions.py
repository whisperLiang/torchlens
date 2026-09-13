"""Function-boundary UOps retain their nested input occurrence addresses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.backends.tinygrad import TinygradBackend


def _nested_function_model() -> Any:
    """Build nested precompiled functions with multiple returned tensor leaves."""

    pytest.importorskip("tinygrad")
    from tinygrad.function import function

    @function
    def inner(x: Any) -> Any:
        """Apply a nonlinearity inside the nested function boundary."""

        return x.relu() * 2

    @function(precompile=True)
    def outer(x: Any) -> tuple[Any, Any]:
        """Return two outputs from the same precompiled invocation."""

        hidden = inner(x)
        return hidden + 1, hidden - 1

    def model(x: Any) -> Any:
        """Combine both tuple projections outside the function boundary."""

        left, right = outer(x)
        return left + right

    return model


@pytest.mark.parametrize("point", ["25%", "50%", "75%"])
def test_tinygrad_function_replay_uses_changed_inputs(point: str) -> None:
    """Bind new values through FUNCTION/GETTUPLE instead of replaying constants.

    Parameters
    ----------
    point:
        Representative cut before, between, or after function projections.
    """

    Tensor = pytest.importorskip("tinygrad").Tensor
    model = _nested_function_model()
    runtime = tl.split.prepare(
        model,
        Tensor.ones(2, 3, device="CPU", dtype="float32").realize(),
        split_request(point, backend="tinygrad", batch_axes={}),
    )
    values = np.array([[-3, 2, 1], [4, -5, 6]], dtype=np.float32)
    replay_input = Tensor(values, device="CPU").realize()
    np.testing.assert_allclose(model(replay_input).numpy(), 4 * np.maximum(values, 0))
    np.testing.assert_allclose(runtime.replay(replay_input).numpy(), 4 * np.maximum(values, 0))
    input_node = runtime.trace_graph.input_node_ids[0]
    assert any(
        runtime.trace_graph.parent_id_for_alias(node, parent) == input_node
        for node in runtime.trace_graph.nodes
        for parent in node.parents
    )
    assert TinygradBackend().validate_trace(runtime.trace) is True


def test_tinygrad_function_validation_rejects_tampered_nested_path() -> None:
    """Nested occurrence addresses stay checked against capture evidence."""

    Tensor = pytest.importorskip("tinygrad").Tensor
    trace = tl.trace(
        _nested_function_model(),
        Tensor.ones(2, 3, device="CPU", dtype="float32").realize(),
        backend="tinygrad",
    )
    assert TinygradBackend().validate_trace(trace) is True
    projection = next(op for op in trace.layer_list if op.layer_type == "gettuple")
    positions = projection.parent_arg_positions["args"]
    position = next(key for key in positions if isinstance(key, tuple))
    positions[(0, 999)] = positions.pop(position)
    assert TinygradBackend().validate_trace(trace) is False
