"""TensorFlow split-training parameter replay tests."""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens.split.errors import SplitUnsupportedError


def _assert_tf_close(tf: Any, left: Any, right: Any) -> None:
    """Assert TensorFlow tensors are close."""

    assert bool(tf.reduce_all(tf.abs(left - right) < 1e-5).numpy())


def test_tf_module_split_training_optimizer_step_matches_full_step() -> None:
    """TensorFlow split training should replay live variable reads for optimizers."""

    tf = pytest.importorskip("tensorflow")

    class TrainModule(tf.Module):
        """Small TensorFlow module with trainable prefix and suffix variables."""

        def __init__(self) -> None:
            """Initialize deterministic variables."""

            super().__init__()
            self.w1 = tf.Variable(tf.reshape(tf.linspace(-0.3, 0.3, 20), (4, 5)), name="w1")
            self.b1 = tf.Variable(tf.linspace(-0.1, 0.1, 5), name="b1")
            self.w2 = tf.Variable(tf.reshape(tf.linspace(-0.2, 0.2, 15), (5, 3)), name="w2")
            self.b2 = tf.Variable(tf.linspace(-0.05, 0.05, 3), name="b2")

        def __call__(self, x: Any) -> Any:
            """Run the TensorFlow MLP."""

            hidden = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
            return tf.matmul(hidden, self.w2) + self.b2

    def clone_from(source: TrainModule) -> TrainModule:
        """Create a module with variables cloned from ``source``."""

        target = TrainModule()
        for left, right in zip(target.trainable_variables, source.trainable_variables, strict=True):
            left.assign(right)
        return target

    with tf.device("/CPU:0"):
        base = TrainModule()
        model = clone_from(base)
        split_model = clone_from(base)
        x = tf.reshape(tf.linspace(-1.0, 1.0, 8), (2, 4))
        y = tf.reshape(tf.linspace(0.25, -0.25, 6), (2, 3))
        runtime = tl.prepare_split(
            split_model,
            x,
            tl.SplitSpec("after:relu", backend="tf", trainable=True, dynamic_batch=(1, 4)),
        )
        full_opt = tf.keras.optimizers.SGD(learning_rate=0.05)
        suffix_opt = tf.keras.optimizers.SGD(learning_rate=0.05)
        prefix_opt = tf.keras.optimizers.SGD(learning_rate=0.05)

        with tf.GradientTape() as tape:
            full_loss = tf.reduce_mean(tf.math.squared_difference(model(x), y))
        full_grads = tape.gradient(full_loss, model.trainable_variables)
        full_opt.apply_gradients(zip(full_grads, model.trainable_variables))

        boundary = runtime.run_training_prefix(x)
        split_loss, grads = runtime.train_suffix(boundary, y, optimizer=suffix_opt)
        runtime.backward_prefix(boundary, grads, optimizer=prefix_opt)

        assert grads
        _assert_tf_close(tf, split_loss, full_loss)
        for left, right in zip(
            split_model.trainable_variables,
            model.trainable_variables,
            strict=True,
        ):
            _assert_tf_close(tf, left, right)


def test_tf_training_boundary_cache_strips_gradient_tape(tmp_path) -> None:
    """TensorFlow training boundaries save as suffix-only cache payloads."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        hidden = tf.nn.relu(x)
        return hidden * 2.0

    x = tf.ones((2, 2), dtype=tf.float32)
    target = tf.zeros_like(x)
    runtime = tl.prepare_split(model, x, tl.SplitSpec("after:relu", backend="tf", trainable=True))
    boundary = runtime.run_training_prefix(x)

    runtime.save_boundary(boundary, tmp_path / "boundary")
    loaded = runtime.load_boundary(tmp_path / "boundary")

    assert loaded.metadata.get("supports_prefix_backward") is False
    assert "tf_tape" not in loaded.metadata
    _loss, grads = runtime.train_suffix(loaded, target)
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(loaded, grads)
