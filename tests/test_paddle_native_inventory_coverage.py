"""Behavioral evidence for the explicitly curated Paddle native-wrapper census."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends.paddle import PaddleBackend
from torchlens.validation.invariants import check_metadata_invariants

pytestmark = [pytest.mark.backend_paddle, pytest.mark.optional]


def _native_case(paddle: Any, name: str) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Build one deterministic native-call fixture with all tensors explicit."""

    x = paddle.arange(32, dtype="float32").reshape([1, 2, 4, 4]) / 10
    weight = paddle.ones([2, 1, 2, 2])
    bias = paddle.to_tensor([0.25, 0.5])
    if name == "add":
        return lambda a, b: paddle._C_ops.add(a, b), (x, x)
    if name == "conv2d":
        return (
            lambda a, b: paddle._C_ops.conv2d(a, b, [1, 1], [0, 0], "EXPLICIT", [1, 1], 2, "NCHW"),
            (x, weight),
        )
    if name == "depthwise_conv2d":
        return (
            lambda a, b: paddle._C_ops.depthwise_conv2d(
                a, b, [1, 1], [0, 0], "EXPLICIT", 2, [1, 1], "NCHW"
            ),
            (x, weight),
        )
    if name == "depthwise_conv2d_bias":
        return (
            lambda a, b, c: paddle._C_ops.depthwise_conv2d_bias(
                a, b, c, [1, 1], [0, 0], "EXPLICIT", 2, [1, 1], "NCHW"
            ),
            (x, weight, bias),
        )
    if name == "pool2d":
        return (
            lambda a: paddle._C_ops.pool2d(
                a, [2, 2], [1, 1], [0, 0], False, True, "NCHW", "avg", False, False, "EXPLICIT"
            ),
            (x,),
        )
    if name == "batch_norm":
        return (
            lambda a, mean, var, scale, offset: paddle._C_ops.batch_norm(
                a, mean, var, scale, offset, True, 0.9, 1e-5, "NCHW", True, False
            )[0],
            (x, paddle.zeros([2]), paddle.ones([2]), paddle.ones([2]), bias),
        )
    if name == "functional.relu6":
        return lambda a: paddle.nn.functional.relu6(a), (x,)
    if name == "tensor._use_gpudnn":
        return lambda a: paddle.nn.functional.relu(a._use_gpudnn(False)), (x,)
    assert name in {"relu6", "hardswish"}
    return lambda a: getattr(paddle._C_ops, name)(a), (x,)


@pytest.mark.parametrize(
    "name",
    [
        "add",
        "batch_norm",
        "conv2d",
        "depthwise_conv2d",
        "depthwise_conv2d_bias",
        "hardswish",
        "pool2d",
        "relu6",
        "functional.relu6",
        "tensor._use_gpudnn",
    ],
)
def test_curated_native_calls_preserve_outputs_and_validate(name: str) -> None:
    """Native captures preserve stock outputs, replay, metadata, and kernel refusals."""

    paddle = pytest.importorskip("paddle")
    model, args = _native_case(paddle, name)
    try:
        expected = model(*args)
    except RuntimeError as error:
        # The symbol exists in the CPU wheel but its fused kernel is not built.
        # This is a stock Paddle refusal, not permission for a silent fallback
        # or a skipped capture check. On a wheel providing the CPU kernel the
        # ordinary value/replay assertions below apply instead.
        assert name == "depthwise_conv2d_bias"
        assert "kernel `depthwise_conv2d_bias` is not registered" in str(error)
        with pytest.raises(RuntimeError) as captured:
            tl.trace(model, args, backend="paddle")
        assert str(captured.value) == str(error)
        return

    trace = tl.trace(model, args, backend="paddle")
    try:
        expected_op = name if "." in name else f"c_ops.{name}"
        assert any(op.func_name == expected_op for op in trace.layer_list)
        np.testing.assert_allclose(trace.output_ops[0].out.numpy(), expected.numpy())
        assert check_metadata_invariants(trace) is True
        assert PaddleBackend().validate_trace(trace) is True
    finally:
        trace.cleanup()
