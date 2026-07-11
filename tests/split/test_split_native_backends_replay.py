"""Native non-Torch split replay tests."""

from __future__ import annotations

from v2_helpers import split_request

from pathlib import Path
from typing import Any

import pytest

import torchlens as tl
from torchlens.split.errors import SplitUnsupportedError


def _flatten_numbers(value: Any) -> list[float]:
    """Flatten a nested numeric list for approximate assertions."""

    if isinstance(value, list):
        return [number for item in value for number in _flatten_numbers(item)]
    return [float(value)]


def test_jax_split_replay_and_cache_roundtrip(tmp_path: Path) -> None:
    """JAX native-IR prefix + suffix replay matches the original callable."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        hidden = jnp.maximum(x, 0)
        return hidden * 2.0 + 1.0

    x = jnp.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    runtime = tl.split.prepare(model, x, split_request("after:max", backend="jax"))

    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    assert bool(jnp.allclose(runtime.run_suffix(boundary), model(x)))

    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    runtime.validate_boundary(loaded)
    assert bool(jnp.allclose(runtime.run_suffix(loaded), model(x)))


def test_tinygrad_split_replay_and_cache_roundtrip(tmp_path: Path) -> None:
    """tinygrad UOp prefix + suffix replay matches the original callable."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        hidden = x.relu()
        return hidden * 2.0 + 1.0

    x = tinygrad.Tensor([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]).realize()
    runtime = tl.split.prepare(model, x, split_request("after:where", backend="tinygrad"))

    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    assert _flatten_numbers(runtime.run_suffix(boundary).tolist()) == pytest.approx(
        _flatten_numbers(model(x).realize().tolist())
    )

    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    runtime.validate_boundary(loaded)
    assert _flatten_numbers(runtime.run_suffix(loaded).tolist()) == pytest.approx(
        _flatten_numbers(model(x).realize().tolist())
    )


def test_tf_split_replay_and_cache_roundtrip(tmp_path: Path) -> None:
    """TensorFlow raw-op prefix + suffix replay matches the original callable."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        hidden = tf.nn.relu(x)
        return hidden * 2.0 + 1.0

    x = tf.constant([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=tf.float32)
    runtime = tl.split.prepare(model, x, split_request("after:relu", backend="tf"))

    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    assert bool(tf.reduce_all(tf.abs(runtime.run_suffix(boundary) - model(x)) < 1e-5).numpy())

    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    runtime.validate_boundary(loaded)
    assert bool(tf.reduce_all(tf.abs(runtime.run_suffix(loaded) - model(x)) < 1e-5).numpy())


def test_jax_dynamic_batch_replay_for_reshape() -> None:
    """JAX replay rewrites conservative leading-batch shape literals."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        batch = x.shape[0]
        flat = jnp.reshape(x, (batch, -1))
        return jnp.maximum(flat, 0) * 2.0

    x = jnp.ones((2, 3, 2))
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape", backend="jax", dynamic_batch=(1, 4)),
    )

    for batch in (1, 2, 4):
        replay_x = jnp.ones((batch, 3, 2))
        assert bool(jnp.allclose(runtime.replay(replay_x), model(replay_x)))


def test_jax_dynamic_batch_preserves_fixed_dim_matching_trace_batch() -> None:
    """JAX dynamic replay only rewrites the leading shape dimension."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        flat = jnp.reshape(x, (x.shape[0], 2))
        return jnp.maximum(flat, 0)

    x = jnp.ones((2, 1, 2))
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape", backend="jax", dynamic_batch=(1, 4)),
    )

    for batch in (1, 2, 4):
        replay_x = jnp.ones((batch, 1, 2))
        assert bool(jnp.allclose(runtime.replay(replay_x), model(replay_x)))


def test_tf_dynamic_batch_replay_for_reshape() -> None:
    """TensorFlow replay rewrites conservative leading-batch shape literals."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        flat = tf.reshape(x, (tf.shape(x)[0], -1))
        return tf.nn.relu(flat) * 2.0

    x = tf.ones((2, 3, 2), dtype=tf.float32)
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape", backend="tf", dynamic_batch=(1, 4)),
    )

    for batch in (1, 2, 4):
        replay_x = tf.ones((batch, 3, 2), dtype=tf.float32)
        diff = tf.abs(runtime.replay(replay_x) - model(replay_x))
        assert bool(tf.reduce_all(diff < 1e-5).numpy())


def test_tf_dynamic_batch_preserves_fixed_dim_matching_trace_batch() -> None:
    """TensorFlow dynamic replay only rewrites the leading shape dimension."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        flat = tf.reshape(x, (tf.shape(x)[0], 2))
        return tf.nn.relu(flat)

    x = tf.ones((2, 1, 2), dtype=tf.float32)
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape", backend="tf", dynamic_batch=(1, 4)),
    )

    for batch in (1, 2, 4):
        replay_x = tf.ones((batch, 1, 2), dtype=tf.float32)
        diff = tf.abs(runtime.replay(replay_x) - model(replay_x))
        assert bool(tf.reduce_all(diff < 1e-5).numpy())


def test_tinygrad_dynamic_batch_replay_for_reshape() -> None:
    """tinygrad replay rewrites conservative leading-batch shape literals."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        flat = x.reshape(x.shape[0], -1)
        return flat.relu() * 2.0

    x = tinygrad.Tensor.ones(2, 3, 2).realize()
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape_3", backend="tinygrad", dynamic_batch=(1, 4)),
    )

    for batch in (1, 2, 4):
        replay_x = tinygrad.Tensor.ones(batch, 3, 2).realize()
        assert _flatten_numbers(runtime.replay(replay_x).tolist()) == pytest.approx(
            _flatten_numbers(model(replay_x).realize().tolist())
        )


def test_tinygrad_dynamic_batch_rejects_out_of_range_batch() -> None:
    """tinygrad dynamic batch replay validates the allowed leading-dim range."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        flat = x.reshape(x.shape[0], -1)
        return flat.relu() * 2.0

    x = tinygrad.Tensor.ones(2, 3, 2).realize()
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape_3", backend="tinygrad", dynamic_batch=(1, 4)),
    )

    with pytest.raises(SplitUnsupportedError, match="outside"):
        runtime.replay(tinygrad.Tensor.ones(5, 3, 2).realize())


def test_tinygrad_dynamic_batch_rejects_non_batch_dim_change() -> None:
    """tinygrad dynamic-batch replay fails closed for changed non-batch dimensions."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        flat = x.reshape(x.shape[0], -1)
        return flat.relu() * 2.0

    x = tinygrad.Tensor.ones(2, 3, 2).realize()
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape_3", backend="tinygrad", dynamic_batch=(1, 4)),
    )

    with pytest.raises((SplitUnsupportedError, ValueError, RuntimeError)):
        runtime.replay(tinygrad.Tensor.ones(2, 4, 2).realize())
