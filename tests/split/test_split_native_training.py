"""Native backend split-training tests."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

import torchlens as tl
from torchlens.split.errors import SplitUnsupportedError

from _paddle_subprocess import run_paddle_subprocess


def _run_tinygrad_subprocess(code: str) -> None:
    """Run a tinygrad training test in an isolated Python-device subprocess."""

    if find_spec("tinygrad") is None:
        pytest.skip("'tinygrad' is not installed.")
    env = os.environ.copy()
    env["DEV"] = "PYTHON"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "tinygrad split-training subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


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

    result = runtime.train_suffix_result(
        boundary,
        targets,
        optimizer=tf.keras.optimizers.SGD(0.1),
    )
    assert result.optimizer_applied is False


def test_paddle_suffix_boundary_gradients() -> None:
    """Paddle split training returns suffix boundary gradients."""

    run_paddle_subprocess(
        """
        import paddle
        import torchlens as tl

        def model(x):
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
        assert bool(
            paddle.allclose(grads[boundary_key], expected_grad, atol=1e-5, rtol=1e-4).item()
        )
        assert bool(paddle.allclose(loss, expected_loss, atol=1e-5, rtol=1e-4).item())
        """
    )


def test_tinygrad_suffix_boundary_gradients() -> None:
    """tinygrad split training returns suffix boundary gradients."""

    _run_tinygrad_subprocess(
        """
        import pytest
        from tinygrad import Tensor

        import torchlens as tl

        def flatten_numbers(value):
            if isinstance(value, list):
                return [number for item in value for number in flatten_numbers(item)]
            return [float(value)]

        def model(x):
            hidden = x.relu()
            return hidden * 3.0

        x = Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
        targets = Tensor.ones(2, 2).realize()
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
        assert flatten_numbers(grads[boundary_key].tolist()) == pytest.approx(
            flatten_numbers(hidden.grad.tolist())
        )
        assert loss.realize().tolist() == pytest.approx(expected_loss.realize().tolist())
        """
    )


def test_tinygrad_suffix_and_prefix_gradient_handoff() -> None:
    """tinygrad split training hands suffix gradients back through the prefix."""

    _run_tinygrad_subprocess(
        """
        import pytest
        from tinygrad import Tensor

        import torchlens as tl

        def flatten_numbers(value):
            if isinstance(value, list):
                return [number for item in value for number in flatten_numbers(item)]
            return [float(value)]

        def model(x):
            hidden = x.relu()
            return hidden * 3.0

        x = Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
        x.requires_grad = True
        targets = Tensor.ones(2, 2).realize()
        runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("after:where", backend="tinygrad", trainable=True),
        )
        boundary = runtime.run_training_prefix(x)
        _loss, grads = runtime.train_suffix(boundary, targets)

        runtime.backward_prefix(boundary, grads)

        baseline_x = Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
        baseline_x.requires_grad = True
        expected_loss = ((model(baseline_x) - targets) ** 2).mean()
        expected_loss.backward()
        assert flatten_numbers(x.grad.tolist()) == pytest.approx(
            flatten_numbers(baseline_x.grad.tolist())
        )
        """
    )


def test_tinygrad_suffix_optimizer_step_matches_full_step() -> None:
    """tinygrad suffix optimizer mutates suffix-owned live params like a full step."""

    _run_tinygrad_subprocess(
        """
        import pytest
        from tinygrad import Tensor
        from tinygrad.nn import optim

        import torchlens as tl

        weight = Tensor([2.0]).realize()
        weight.requires_grad = True

        def split_model(x):
            return x.relu() * weight

        x = Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
        targets = Tensor.ones(2, 2).realize()
        runtime = tl.prepare_split(
            split_model,
            x,
            tl.SplitSpec("after:where", backend="tinygrad", trainable=True),
        )
        optimizer = optim.SGD([weight], lr=0.1)
        boundary = runtime.run_training_prefix(x)
        runtime.train_suffix(boundary, targets, optimizer=optimizer)

        full_weight = Tensor([2.0]).realize()
        full_weight.requires_grad = True

        def full_model(x):
            return x.relu() * full_weight

        full_optimizer = optim.SGD([full_weight], lr=0.1)
        old_training = Tensor.training
        Tensor.training = True
        try:
            full_optimizer.zero_grad()
            expected_loss = ((full_model(x) - targets) ** 2).mean()
            expected_loss.backward()
            full_optimizer.step()
        finally:
            Tensor.training = old_training

        assert weight.realize().tolist() == pytest.approx(full_weight.realize().tolist())
        """
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

    _run_tinygrad_subprocess(
        """
        import pytest
        from tinygrad import Tensor

        import torchlens as tl
        from torchlens.split.errors import SplitUnsupportedError

        def model(x):
            return x.relu() * 2.0

        x = Tensor.ones(2, 2).realize()
        runtime = tl.prepare_split(model, x, tl.SplitSpec("after:where", backend="tinygrad"))
        boundary = runtime.run_prefix(x)
        _loss, grads = runtime.train_suffix(boundary, Tensor.ones(2, 2).realize())

        with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
            runtime.backward_prefix(boundary, grads)
        """
    )
