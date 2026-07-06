"""Paddle split replay tests."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

import pytest

import torchlens as tl


def _paddle_runtime_or_skip() -> Any:
    """Import Paddle unless TensorFlow has made this process unsafe for it."""

    tensorflow_loaded = any(
        name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules
    )
    if "paddle" not in sys.modules and tensorflow_loaded:
        pytest.skip("Paddle runtime is unsafe to import after TensorFlow in this process")
    return pytest.importorskip("paddle")


def _paddle() -> Any:
    """Import Paddle lazily inside selected tests."""

    return _paddle_runtime_or_skip()


def _assert_close(left: Any, right: Any) -> None:
    """Assert Paddle tensors are close."""

    paddle = _paddle()
    assert bool(paddle.allclose(left, right, atol=1e-5, rtol=1e-4).item())


def _paddle_block(x: Any) -> Any:
    """Small Paddle function with a live split frontier."""

    paddle = _paddle()
    hidden = paddle.nn.functional.relu(x)
    return hidden * 2.0 + 1.0


def test_paddle_split_replay_equivalence() -> None:
    """Paddle prefix + suffix replay matches the original callable."""

    paddle = _paddle()
    x = paddle.randn([2, 3], dtype="float32")

    runtime = tl.prepare_split(
        _paddle_block,
        x,
        tl.SplitSpec("after:relu", backend="paddle"),
    )

    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    _assert_close(runtime.run_suffix(boundary), _paddle_block(x))
    _assert_close(runtime.replay(x), _paddle_block(x))


def test_paddle_boundary_cache_roundtrip(tmp_path: Path) -> None:
    """Paddle replay boundaries round-trip through trusted local cache."""

    paddle = _paddle()
    x = paddle.randn([2, 3], dtype="float32")
    runtime = tl.prepare_split(
        _paddle_block,
        x,
        tl.SplitSpec("after:relu", backend="paddle"),
    )
    boundary = runtime.run_prefix(x)

    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)

    runtime.validate_boundary(loaded)
    _assert_close(runtime.run_suffix(loaded), _paddle_block(x))


def test_paddle_dynamic_batch_replay_for_reshape() -> None:
    """Paddle replay rewrites conservative leading-batch shape literals."""

    paddle = _paddle()

    def model(x: Any) -> Any:
        batch = x.shape[0]
        flat = paddle.reshape(x, [batch, -1])
        return paddle.nn.functional.relu(flat) * 2.0

    x = paddle.randn([2, 3, 2], dtype="float32")
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec("after:reshape", backend="paddle", dynamic_batch=(1, 4)),
    )

    for batch in (1, 2, 4):
        replay_x = paddle.randn([batch, 3, 2], dtype="float32")
        _assert_close(runtime.replay(replay_x), model(replay_x))


def test_paddle_dynamic_batch_preserves_fixed_dim_matching_trace_batch() -> None:
    """Paddle dynamic replay only rewrites the leading shape dimension."""

    paddle = _paddle()

    def model(x: Any) -> Any:
        flat = paddle.reshape(x, [x.shape[0], 2])
        return paddle.nn.functional.relu(flat)

    x = paddle.randn([2, 1, 2], dtype="float32")
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec("after:reshape", backend="paddle", dynamic_batch=(1, 4)),
    )

    for batch in (1, 2, 4):
        replay_x = paddle.randn([batch, 1, 2], dtype="float32")
        _assert_close(runtime.replay(replay_x), model(replay_x))


def test_paddle_layer_split_training_optimizer_step_matches_full_step() -> None:
    """Paddle split training should resolve unlabeled Linear params to live handles."""

    paddle = _paddle()

    class PaddleTrainMlp(paddle.nn.Layer):
        """Small Paddle layer with trainable prefix and suffix parameters."""

        def __init__(self) -> None:
            """Initialize the Paddle sublayers."""

            super().__init__()
            self.fc1 = paddle.nn.Linear(4, 5)
            self.relu = paddle.nn.ReLU()
            self.fc2 = paddle.nn.Linear(5, 3)

        def forward(self, x: Any) -> Any:
            """Run the Paddle MLP."""

            return self.fc2(self.relu(self.fc1(x)))

    paddle.seed(123)
    model = PaddleTrainMlp()
    split_model = copy.deepcopy(model)
    x = paddle.randn([2, 4], dtype="float32")
    y = paddle.randn([2, 3], dtype="float32")
    runtime = tl.prepare_split(
        split_model,
        x,
        tl.SplitSpec("after:relu", backend="paddle", trainable=True, dynamic_batch=(1, 4)),
    )
    suffix_opt = paddle.optimizer.SGD(
        learning_rate=0.05,
        parameters=split_model.fc2.parameters(),
    )
    prefix_opt = paddle.optimizer.SGD(
        learning_rate=0.05,
        parameters=split_model.fc1.parameters(),
    )
    full_opt = paddle.optimizer.SGD(learning_rate=0.05, parameters=model.parameters())

    full_opt.clear_grad()
    full_loss = paddle.nn.functional.mse_loss(model(x), y)
    full_loss.backward()
    full_opt.step()

    boundary = runtime.run_training_prefix(x)
    split_loss, grads = runtime.train_suffix(boundary, y, optimizer=suffix_opt)
    runtime.backward_prefix(boundary, grads, optimizer=prefix_opt)

    assert grads
    _assert_close(split_loss, full_loss)
    for left, right in zip(split_model.parameters(), model.parameters(), strict=True):
        _assert_close(left, right)
