"""Native non-Torch split replay tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError


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
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:max", backend="jax"),
    )

    replay_x = jnp.ones((3, 3))
    boundary = runtime.run_prefix(replay_x)
    runtime.validate_boundary(boundary)
    assert bool(jnp.allclose(runtime.run_suffix(boundary), model(replay_x)))

    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    runtime.validate_boundary(loaded)
    assert bool(jnp.allclose(runtime.run_suffix(loaded), model(replay_x)))


def test_tinygrad_split_replay_and_cache_roundtrip(tmp_path: Path) -> None:
    """tinygrad UOp prefix + suffix replay matches the original callable."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        hidden = x.relu()
        return hidden * 2.0 + 1.0

    x = tinygrad.Tensor([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]).realize()
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:where", backend="tinygrad"),
    )

    replay_x = tinygrad.Tensor([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [7.0, -8.0, 9.0]]).realize()
    boundary = runtime.run_prefix(replay_x)
    runtime.validate_boundary(boundary)
    assert _flatten_numbers(runtime.run_suffix(boundary).tolist()) == pytest.approx(
        _flatten_numbers(model(replay_x).realize().tolist())
    )

    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    runtime.validate_boundary(loaded)
    assert _flatten_numbers(runtime.run_suffix(loaded).tolist()) == pytest.approx(
        _flatten_numbers(model(replay_x).realize().tolist())
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


def test_tf_gpu_kernel_after_torch_optimizer() -> None:
    """Torch training must not corrupt TensorFlow's subsequent LLVM GPU compilation."""

    tf = pytest.importorskip("tensorflow")
    import torch

    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("TensorFlow GPU is required for the native compiler regression.")

    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()

    with tf.device("/GPU:0"):
        x = tf.constant([[-1.0, 2.0, 3.0]], dtype=tf.float32)
        output = tf.nn.relu(x)
    assert "GPU:0" in output.device
    assert output.numpy().tolist() == [[0.0, 2.0, 3.0]]


def test_jax_split_replay_preserves_integer_dict_and_list_containers() -> None:
    """JAX replay preserves container kinds when integer keys are present."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        hidden = jnp.maximum(x, 0)
        result = hidden * 2.0
        return {0: result, 1: [result + 1.0]}

    x = jnp.ones((2, 3))
    runtime = tl.split.prepare(model, x, split_request("after:max", backend="jax"))
    output = runtime.replay(x)
    assert isinstance(output, dict)
    assert isinstance(output[1], list)
    expected = jnp.maximum(x, 0) * 2.0
    assert bool(jnp.allclose(output[0], expected))
    assert bool(jnp.allclose(output[1][0], expected + 1.0))


def test_tf_split_replay_preserves_integer_dict_and_list_containers() -> None:
    """TensorFlow replay preserves container kinds when integer keys are present."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        hidden = tf.nn.relu(x)
        result = hidden * 2.0
        return {0: result, 1: [result + 1.0]}

    x = tf.ones((2, 3), dtype=tf.float32)
    runtime = tl.split.prepare(model, x, split_request("after:relu", backend="tf"))
    output = runtime.replay(x)
    assert isinstance(output, dict)
    assert isinstance(output[1], list)
    expected = tf.nn.relu(x) * 2.0
    assert bool(tf.reduce_all(tf.equal(output[0], expected)).numpy())
    assert bool(tf.reduce_all(tf.equal(output[1][0], expected + 1.0)).numpy())


def test_jax_batch_symbolic_replay_for_reshape() -> None:
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
        split_request("after:reshape", backend="jax"),
    )

    for batch in (1, 2, 4):
        replay_x = jnp.ones((batch, 3, 2))
        assert bool(jnp.allclose(runtime.replay(replay_x), model(replay_x)))


def test_jax_batch_symbolic_replay_for_broadcast_bias() -> None:
    """JAX rewrites data-batch broadcast shapes without rewriting parameters."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        hidden = x + jnp.ones((4,), dtype=x.dtype)
        return hidden * 2.0

    x = jnp.ones((2, 4))
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:broadcast_in_dim_2_3_raw", backend="jax"),
    )

    for batch in (1, 2, 4):
        replay_x = jnp.ones((batch, 4))
        assert bool(jnp.allclose(runtime.replay(replay_x), model(replay_x)))


def test_jax_batch_symbolic_preserves_fixed_dim_matching_trace_batch() -> None:
    """JAX dynamic replay only rewrites the leading shape dimension."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        flat = jnp.reshape(x, (x.shape[0], 2))
        return jnp.maximum(flat, 0)

    x = jnp.ones((2, 1, 2))
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape", backend="jax"),
    )

    for batch in (1, 2, 4):
        replay_x = jnp.ones((batch, 1, 2))
        assert bool(jnp.allclose(runtime.replay(replay_x), model(replay_x)))


def test_tf_batch_symbolic_replay_for_reshape() -> None:
    """TensorFlow replay rewrites conservative leading-batch shape literals."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        flat = tf.reshape(x, (tf.shape(x)[0], -1))
        return tf.nn.relu(flat) * 2.0

    x = tf.ones((2, 3, 2), dtype=tf.float32)
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape", backend="tf"),
    )

    for batch in (1, 2, 4):
        replay_x = tf.ones((batch, 3, 2), dtype=tf.float32)
        diff = tf.abs(runtime.replay(replay_x) - model(replay_x))
        assert bool(tf.reduce_all(diff < 1e-5).numpy())


def test_tf_batch_symbolic_preserves_fixed_dim_matching_trace_batch() -> None:
    """TensorFlow dynamic replay only rewrites the leading shape dimension."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        flat = tf.reshape(x, (tf.shape(x)[0], 2))
        return tf.nn.relu(flat)

    x = tf.ones((2, 1, 2), dtype=tf.float32)
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape", backend="tf"),
    )

    for batch in (1, 2, 4):
        replay_x = tf.ones((batch, 1, 2), dtype=tf.float32)
        diff = tf.abs(runtime.replay(replay_x) - model(replay_x))
        assert bool(tf.reduce_all(diff < 1e-5).numpy())


def test_tinygrad_batch_symbolic_replay_for_reshape() -> None:
    """tinygrad replay rewrites conservative leading-batch shape literals."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        flat = x.reshape(x.shape[0], -1)
        return flat.relu() * 2.0

    x = tinygrad.Tensor.ones(2, 3, 2).realize()
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape_3", backend="tinygrad"),
    )

    # B=1 may retain a singleton UOp that B=2 optimizes away. The native
    # output comparison must still run without inventing aligned shape rows.
    assert runtime.batch_validation["status"] == "passed"
    for batch in (1, 2, 4):
        replay_x = tinygrad.Tensor.ones(batch, 3, 2).realize()
        assert _flatten_numbers(runtime.replay(replay_x).tolist()) == pytest.approx(
            _flatten_numbers(model(replay_x).realize().tolist())
        )


def test_tinygrad_unaligned_shape_probe_still_requires_numeric_equivalence() -> None:
    """Singleton UOp differences cannot bypass the native/replay comparison."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        """Reshape with a batch-dependent scalar on the same tensor path."""

        flat = x.reshape(x.shape[0], -1)
        return flat.relu() * (3.0 if x.shape[0] >= 2 else 2.0)

    runtime = tl.split.prepare(
        model,
        tinygrad.Tensor.ones(2, 3, 2).realize(),
        split_request("after:reshape_3", backend="tinygrad"),
    )
    assert runtime.batch_validation["status"] == "failed"
    assert "numeric mismatch" in runtime.batch_validation["reason"]
    singleton = tinygrad.Tensor.ones(1, 3, 2).realize()
    assert runtime.replay(singleton).tolist() == model(singleton).tolist()
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        runtime.replay(tinygrad.Tensor.ones(2, 3, 2).realize())


def test_tinygrad_batch_symbolic_accepts_untested_batch() -> None:
    """tinygrad replay accepts a compatible batch that was never part of a range."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        flat = x.reshape(x.shape[0], -1)
        return flat.relu() * 2.0

    x = tinygrad.Tensor.ones(2, 3, 2).realize()
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape_3", backend="tinygrad"),
    )

    replay_x = tinygrad.Tensor.ones(5, 3, 2).realize()
    actual = runtime.replay(replay_x)
    expected = model(replay_x)
    assert _flatten_numbers(actual.tolist()) == pytest.approx(
        _flatten_numbers(expected.realize().tolist())
    )


def test_tinygrad_batch_symbolic_rejects_non_batch_dim_change() -> None:
    """tinygrad dynamic-batch replay fails closed for changed non-batch dimensions."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: Any) -> Any:
        flat = x.reshape(x.shape[0], -1)
        return flat.relu() * 2.0

    x = tinygrad.Tensor.ones(2, 3, 2).realize()
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:reshape_3", backend="tinygrad"),
    )

    with pytest.raises((SplitUnsupportedError, ValueError, RuntimeError)):
        runtime.replay(tinygrad.Tensor.ones(2, 4, 2).realize())
