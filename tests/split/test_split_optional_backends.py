"""Optional backend split capability tests."""

from __future__ import annotations

from v2_helpers import split_request

from typing import Any

import pytest

from torchlens.split.adapters import resolve_split_adapter
from torchlens.split.errors import SplitUnsupportedError

from _paddle_subprocess import run_paddle_subprocess


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
    """MLX reports deferred replay explicitly, even when MLX is unavailable."""

    adapter = resolve_split_adapter("mlx")
    assert adapter.supports_replay is False
    assert adapter.supports_training is False
    assert adapter.supports_dynamic_batch is False
    with pytest.raises(SplitUnsupportedError):
        adapter.build_segments(None, None, None)  # type: ignore[arg-type]


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
