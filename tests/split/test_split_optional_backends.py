"""Optional backend split capability tests."""

from __future__ import annotations

from typing import Any

import pytest
from _paddle_subprocess import run_paddle_subprocess
from v2_helpers import split_request

from torchlens.split.adapters import resolve_split_adapter


def _flatten_numbers(value: Any) -> list[float]:
    """Flatten a nested numeric list for approximate assertions."""

    if isinstance(value, list):
        return [number for item in value for number in _flatten_numbers(item)]
    return [float(value)]


def test_jax_optional_adapter_gate() -> None:
    """Installed JAX supports native-IR split replay."""

    jnp = pytest.importorskip("jax.numpy")
    import torchlens as tl

    def model(x: object) -> object:
        return jnp.maximum(x, 0) * 2.0 + 1.0

    x = jnp.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    adapter = resolve_split_adapter("jax")

    assert adapter.supports_replay is True
    runtime = tl.split.prepare(model, x, split_request("after:max", backend="jax"))
    replayed = runtime.replay(x)
    assert bool(jnp.allclose(replayed, model(x)))
    trainable = tl.split.prepare(
        model,
        x,
        split_request("after:max", backend="jax", trainable=True),
    )
    assert trainable.segments.training_prefix is not None


def test_tinygrad_optional_adapter_gate() -> None:
    """Installed tinygrad supports UOp split replay and split training construction."""

    tinygrad = pytest.importorskip("tinygrad")
    import torchlens as tl

    def model(x: object) -> object:
        return x.relu() * 2.0 + 1.0

    x = tinygrad.Tensor([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]).realize()
    adapter = resolve_split_adapter("tinygrad")

    assert adapter.supports_replay is True
    runtime = tl.split.prepare(model, x, split_request("after:where", backend="tinygrad"))
    replayed = runtime.replay(x)
    assert _flatten_numbers(replayed.tolist()) == pytest.approx(
        _flatten_numbers(model(x).realize().tolist())
    )
    trainable = tl.split.prepare(
        model,
        x,
        split_request("after:where", backend="tinygrad", trainable=True),
    )
    assert trainable.segments.training_prefix is not None


def test_mlx_optional_adapter_gate() -> None:
    """Installed MLX supports generated eager replay and boundary caching."""

    mx = pytest.importorskip("mlx.core")
    import torchlens as tl

    def model(x: object) -> object:
        hidden = mx.maximum(x, 0)
        left, right = mx.split(hidden, 2, axis=-1)
        return mx.add(mx.multiply(left, 2), right)

    x = mx.arange(8, dtype=mx.float32).reshape((2, 4)) - 3
    adapter = resolve_split_adapter("mlx")

    assert adapter.supports_replay is True
    assert adapter.supports_boundary_cache is True
    runtime = tl.split.prepare(model, x, split_request("after:maximum", backend="mlx"))
    replayed = runtime.replay(x)
    mx.eval(replayed)
    expected = model(x)
    mx.eval(expected)
    assert bool(mx.allclose(replayed, expected))
    assert adapter.supports_training is True
    assert adapter.supports_state_placement is True
    boundary = runtime.run_prefix(x).to("cpu")
    assert bool(mx.allclose(runtime.run_suffix(boundary), expected))
    training = tl.split.prepare(
        model, x, split_request("after:maximum", backend="mlx", trainable=True)
    )
    assert training.run_training_prefix(x).metadata["supports_prefix_backward"]


def test_mlx_split_reconstructs_direct_multi_output() -> None:
    """MLX replay preserves a list returned directly by a split call."""

    mx = pytest.importorskip("mlx.core")
    import torchlens as tl

    def model(x: object) -> object:
        return list(mx.split(mx.maximum(x, 0), 2, axis=-1))

    x = mx.arange(8, dtype=mx.float32).reshape((2, 4)) - 3
    runtime = tl.split.prepare(model, x, split_request("after:maximum", backend="mlx"))
    replayed = runtime.replay(x)
    expected = model(x)
    mx.eval(*replayed, *expected)
    assert isinstance(replayed, list)
    assert len(replayed) == len(expected) == 2
    assert all(
        bool(mx.allclose(left, right)) for left, right in zip(replayed, expected, strict=True)
    )


def test_mlx_split_reconstructs_nested_output_and_keeps_scalar_literals() -> None:
    """MLX replay keeps final output paths distinct from call output paths."""

    mx = pytest.importorskip("mlx.core")
    import torchlens as tl

    def model(x: object) -> object:
        hidden = mx.maximum(x, 0)
        left, right = mx.split(hidden, 2, axis=-1)
        return {"left": left, "right": [right, mx.add(mx.multiply(left, 2), right)]}

    x = mx.arange(8, dtype=mx.float32).reshape((2, 4)) - 3
    runtime = tl.split.prepare(model, x, split_request("after:maximum", backend="mlx"))
    replayed = runtime.replay(x)
    expected = model(x)
    assert isinstance(replayed, dict)
    assert tuple(replayed) == tuple(expected)
    assert isinstance(replayed["right"], list)
    assert len(replayed["right"]) == len(expected["right"]) == 2
    mx.eval(replayed["left"], *replayed["right"], expected["left"], *expected["right"])
    assert bool(mx.allclose(replayed["left"], expected["left"]))
    assert all(
        bool(mx.allclose(left, right))
        for left, right in zip(replayed["right"], expected["right"], strict=True)
    )


@pytest.mark.heavy
def test_paddle_optional_adapter_gate() -> None:
    """Installed Paddle supports generated-eager split replay."""

    run_paddle_subprocess(
        """
        import paddle
        import torchlens as tl
        from torchlens.split.adapters import resolve_split_adapter

        def model(x):
            return paddle.nn.functional.relu(x) * 2

        x = paddle.randn([2, 3], dtype="float32")
        adapter = resolve_split_adapter("paddle")

        assert adapter.supports_replay is True
        runtime = tl.split.prepare(model, x, split_request("after:relu", backend="paddle"))
        replayed = runtime.replay(x)
        assert bool(paddle.allclose(replayed, model(x)).item())
        trainable = tl.split.prepare(
            model,
            x,
            split_request("after:relu", backend="paddle", trainable=True),
        )
        assert trainable.segments.training_prefix is not None
        """
    )


def test_paddle_training_capability_is_advertised() -> None:
    """Paddle split training is exposed by the adapter."""

    run_paddle_subprocess(
        """
        from torchlens.split.adapters import resolve_split_adapter

        adapter = resolve_split_adapter("paddle")

        assert adapter.supports_training is True
        """
    )


def test_tf_optional_adapter_gate() -> None:
    """Installed TensorFlow supports raw-op split replay."""

    tf = pytest.importorskip("tensorflow")
    import torchlens as tl

    def model(x: object) -> object:
        return tf.nn.relu(x) * 2.0 + 1.0

    x = tf.constant([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=tf.float32)
    adapter = resolve_split_adapter("tf")

    assert adapter.supports_replay is True
    runtime = tl.split.prepare(model, x, split_request("after:relu", backend="tf"))
    replayed = runtime.replay(x)
    assert bool(tf.reduce_all(tf.abs(replayed - model(x)) < 1e-5).numpy())
    trainable = tl.split.prepare(
        model,
        x,
        split_request("after:relu", backend="tf", trainable=True),
    )
    assert trainable.segments.training_prefix is not None
