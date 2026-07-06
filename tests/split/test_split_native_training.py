"""Native backend split-training tests."""

from __future__ import annotations

import sys
from typing import Any

import pytest

import torchlens as tl
from torchlens.split.errors import SplitUnsupportedError


def _paddle_runtime_or_skip() -> Any:
    """Import Paddle unless TensorFlow has made this process unsafe for it."""

    tensorflow_loaded = any(
        name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules
    )
    if "paddle" not in sys.modules and tensorflow_loaded:
        pytest.skip("Paddle runtime is unsafe to import after TensorFlow in this process")
    return pytest.importorskip("paddle")


def _flatten_numbers(value: Any) -> list[float]:
    """Flatten a nested numeric list for approximate assertions."""

    if isinstance(value, list):
        return [number for item in value for number in _flatten_numbers(item)]
    return [float(value)]


def test_jax_suffix_and_prefix_gradient_handoff() -> None:
    """JAX split training returns suffix boundary grads and prefix input grads."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        hidden = jnp.maximum(x, 0)
        return hidden * 3.0

    x = jnp.array([[-1.0, 2.0], [3.0, -4.0]])
    targets = jnp.ones_like(x)
    runtime = tl.prepare_split(model, x, tl.SplitSpec("after:max", backend="jax", trainable=True))
    boundary = runtime.run_training_prefix(x)

    loss, grads = runtime.train_suffix(boundary, targets)

    boundary_key = next(iter(boundary.tensors))

    def suffix_loss(hidden: Any) -> Any:
        return jnp.mean((hidden * 3.0 - targets) ** 2)

    expected_boundary_grad = jax.grad(suffix_loss)(jnp.maximum(x, 0))
    assert bool(jnp.allclose(grads[boundary_key], expected_boundary_grad))

    prefix_grads = runtime.backward_prefix(boundary, grads)
    expected_input_grad = jax.grad(lambda item: jnp.mean((model(item) - targets) ** 2))(x)
    assert bool(jnp.allclose(prefix_grads["inputs"][0], expected_input_grad))
    assert bool(jnp.allclose(loss, suffix_loss(jnp.maximum(x, 0))))

    with pytest.raises(SplitUnsupportedError, match="optimizer"):
        runtime.train_suffix(boundary, targets, optimizer=object())


def test_tf_suffix_boundary_gradients() -> None:
    """TensorFlow split training returns suffix boundary gradients."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        hidden = tf.nn.relu(x)
        return hidden * 3.0

    x = tf.constant([[-1.0, 2.0], [3.0, -4.0]], dtype=tf.float32)
    targets = tf.ones_like(x)
    runtime = tl.prepare_split(model, x, tl.SplitSpec("after:relu", backend="tf", trainable=True))
    boundary = runtime.run_training_prefix(x)

    loss, grads = runtime.train_suffix(boundary, targets)

    boundary_key = next(iter(boundary.tensors))
    with tf.GradientTape() as tape:
        hidden = tf.nn.relu(x)
        tape.watch(hidden)
        expected_loss = tf.reduce_mean(tf.math.squared_difference(hidden * 3.0, targets))
    expected_grad = tape.gradient(expected_loss, hidden)
    assert bool(tf.reduce_all(tf.abs(grads[boundary_key] - expected_grad) < 1e-5).numpy())
    assert bool(tf.abs(loss - expected_loss).numpy() < 1e-5)


def test_paddle_suffix_boundary_gradients() -> None:
    """Paddle split training returns suffix boundary gradients."""

    paddle = _paddle_runtime_or_skip()

    def model(x: Any) -> Any:
        hidden = paddle.nn.functional.relu(x)
        return hidden * 3.0

    x = paddle.to_tensor([[-1.0, 2.0], [3.0, -4.0]], dtype="float32")
    targets = paddle.ones_like(x)
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec("after:relu", backend="paddle", trainable=True),
    )
    boundary = runtime.run_training_prefix(x)

    loss, grads = runtime.train_suffix(boundary, targets)

    boundary_key = next(iter(boundary.tensors))
    hidden = paddle.nn.functional.relu(x)
    hidden.stop_gradient = False
    expected_loss = paddle.nn.functional.mse_loss(hidden * 3.0, targets)
    (expected_grad,) = paddle.grad(expected_loss, [hidden])
    assert bool(paddle.allclose(grads[boundary_key], expected_grad, atol=1e-5, rtol=1e-4).item())
    assert bool(paddle.allclose(loss, expected_loss, atol=1e-5, rtol=1e-4).item())


def test_tinygrad_suffix_boundary_gradients() -> None:
    """tinygrad split training returns suffix boundary gradients."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        hidden = x.relu()
        return hidden * 3.0

    x = tinygrad.Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
    targets = tinygrad.Tensor.ones(2, 2).realize()
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec("after:where", backend="tinygrad", trainable=True),
    )
    boundary = runtime.run_training_prefix(x)

    loss, grads = runtime.train_suffix(boundary, targets)

    boundary_key = next(iter(boundary.tensors))
    hidden = x.relu().detach()
    hidden.requires_grad = True
    expected_loss = ((hidden * 3.0 - targets) ** 2).mean()
    expected_loss.backward()
    assert _flatten_numbers(grads[boundary_key].tolist()) == pytest.approx(
        _flatten_numbers(hidden.grad.tolist())
    )
    assert loss.realize().tolist() == pytest.approx(expected_loss.realize().tolist())


def test_tinygrad_suffix_and_prefix_gradient_handoff() -> None:
    """tinygrad split training hands suffix gradients back through the prefix."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        hidden = x.relu()
        return hidden * 3.0

    x = tinygrad.Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
    x.requires_grad = True
    targets = tinygrad.Tensor.ones(2, 2).realize()
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec("after:where", backend="tinygrad", trainable=True),
    )
    boundary = runtime.run_training_prefix(x)
    _loss, grads = runtime.train_suffix(boundary, targets)

    runtime.backward_prefix(boundary, grads)

    baseline_x = tinygrad.Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
    baseline_x.requires_grad = True
    expected_loss = ((model(baseline_x) - targets) ** 2).mean()
    expected_loss.backward()
    assert _flatten_numbers(x.grad.tolist()) == pytest.approx(
        _flatten_numbers(baseline_x.grad.tolist())
    )


def test_tinygrad_suffix_optimizer_step_matches_full_step() -> None:
    """tinygrad suffix optimizer mutates suffix-owned live params like a full step."""

    tinygrad = pytest.importorskip("tinygrad")
    optim = pytest.importorskip("tinygrad.nn.optim")

    class ScaleHead:
        """Tiny suffix parameter model for split optimizer parity."""

        def __init__(self) -> None:
            """Initialize the tinygrad parameter."""

            self.weight = tinygrad.Tensor([2.0]).realize()
            self.weight.requires_grad = True

        def __call__(self, x: Any) -> Any:
            """Run a ReLU prefix and trainable scale suffix."""

            return x.relu() * self.weight

    split_model = ScaleHead()
    x = tinygrad.Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
    targets = tinygrad.Tensor.ones(2, 2).realize()
    runtime = tl.prepare_split(
        split_model,
        x,
        tl.SplitSpec("after:where", backend="tinygrad", trainable=True),
    )
    optimizer = optim.SGD([split_model.weight], lr=0.1)
    boundary = runtime.run_training_prefix(x)
    runtime.train_suffix(boundary, targets, optimizer=optimizer)

    full_model = ScaleHead()
    full_model.weight.assign(tinygrad.Tensor([2.0]))
    full_optimizer = optim.SGD([full_model.weight], lr=0.1)
    old_training = tinygrad.Tensor.training
    tinygrad.Tensor.training = True
    try:
        full_optimizer.zero_grad()
        expected_loss = ((full_model(x) - targets) ** 2).mean()
        expected_loss.backward()
        full_optimizer.step()
    finally:
        tinygrad.Tensor.training = old_training

    assert split_model.weight.realize().tolist() == pytest.approx(
        full_model.weight.realize().tolist()
    )


def test_tf_detached_boundary_rejects_prefix_backward() -> None:
    """A detached TensorFlow replay boundary cannot drive prefix backprop."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        return tf.nn.relu(x) * 2.0

    x = tf.ones((2, 2), dtype=tf.float32)
    runtime = tl.prepare_split(model, x, tl.SplitSpec("after:relu", backend="tf", trainable=True))
    boundary = runtime.run_prefix(x)
    _loss, grads = runtime.train_suffix(boundary, tf.ones_like(x))

    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(boundary, grads)


def test_tinygrad_detached_boundary_rejects_prefix_backward() -> None:
    """A detached tinygrad replay boundary cannot drive prefix backprop."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        return x.relu() * 2.0

    x = tinygrad.Tensor.ones(2, 2).realize()
    runtime = tl.prepare_split(model, x, tl.SplitSpec("after:where", backend="tinygrad"))
    boundary = runtime.run_prefix(x)
    _loss, grads = runtime.train_suffix(boundary, tinygrad.Tensor.ones(2, 2).realize())

    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(boundary, grads)
