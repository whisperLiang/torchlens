"""Optional backend split capability tests."""

from __future__ import annotations

from typing import Any

import pytest
from _paddle_subprocess import run_paddle_subprocess
from dag_helpers import call_ids, close_call_ids

from torchlens.split import SplitFeatures, prepare_graph
from torchlens.split.adapters import resolve_split_adapter


def _flatten_numbers(value: Any) -> list[float]:
    """Flatten a nested numeric list for approximate assertions."""

    if isinstance(value, list):
        return [number for item in value for number in _flatten_numbers(item)]
    return [float(value)]


def test_jax_optional_adapter_gate() -> None:
    """Installed JAX supports native-IR split replay."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: object) -> object:
        return jnp.maximum(x, 0) * 2.0 + 1.0

    x = jnp.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    adapter = resolve_split_adapter("jax")

    assert adapter.supports_replay is True
    graph = prepare_graph(model, x, backend="jax")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"max"}), ops=frozenset({"max"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    replayed = runtime.replay(x)
    assert bool(jnp.allclose(replayed, model(x)))
    trainable_graph = prepare_graph(model, x, backend="jax", features=SplitFeatures(training=True))
    trainable_dag = trainable_graph.compute_dag()
    trainable_partition = trainable_dag.to_partition(
        close_call_ids(
            trainable_dag,
            call_ids(trainable_graph, modules=frozenset({"max"}), ops=frozenset({"max"})),
        )
    )
    trainable = trainable_graph.materialize(trainable_graph.analyze(trainable_partition))
    assert trainable.segments.training_prefix is not None


def test_tinygrad_optional_adapter_gate() -> None:
    """Installed tinygrad supports UOp split replay and split training construction."""

    tinygrad = pytest.importorskip("tinygrad")

    def model(x: object) -> object:
        return x.relu() * 2.0 + 1.0

    x = tinygrad.Tensor([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]).realize()
    adapter = resolve_split_adapter("tinygrad")

    assert adapter.supports_replay is True
    graph = prepare_graph(model, x, backend="tinygrad")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"where"}), ops=frozenset({"where"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    replayed = runtime.replay(x)
    assert _flatten_numbers(replayed.tolist()) == pytest.approx(
        _flatten_numbers(model(x).realize().tolist())
    )
    trainable_graph = prepare_graph(
        model, x, backend="tinygrad", features=SplitFeatures(training=True)
    )
    trainable_dag = trainable_graph.compute_dag()
    trainable_partition = trainable_dag.to_partition(
        close_call_ids(
            trainable_dag,
            call_ids(trainable_graph, modules=frozenset({"where"}), ops=frozenset({"where"})),
        )
    )
    trainable = trainable_graph.materialize(trainable_graph.analyze(trainable_partition))
    assert trainable.segments.training_prefix is not None


def test_mlx_optional_adapter_gate() -> None:
    """Installed MLX supports generated eager replay and boundary caching."""

    mx = pytest.importorskip("mlx.core")

    def model(x: object) -> object:
        hidden = mx.maximum(x, 0)
        left, right = mx.split(hidden, 2, axis=-1)
        return mx.add(mx.multiply(left, 2), right)

    x = mx.arange(8, dtype=mx.float32).reshape((2, 4)) - 3
    adapter = resolve_split_adapter("mlx")

    assert adapter.supports_replay is True
    assert adapter.supports_boundary_cache is True
    graph = prepare_graph(model, x, backend="mlx")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"maximum"}), ops=frozenset({"maximum"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    replayed = runtime.replay(x)
    mx.eval(replayed)
    expected = model(x)
    mx.eval(expected)
    assert bool(mx.allclose(replayed, expected))
    assert adapter.supports_training is True
    assert adapter.supports_state_placement is True
    boundary = runtime.run_prefix(x).to("cpu")
    assert bool(mx.allclose(runtime.run_suffix(boundary), expected))
    training_graph = prepare_graph(model, x, backend="mlx", features=SplitFeatures(training=True))
    training_dag = training_graph.compute_dag()
    training_partition = training_dag.to_partition(
        close_call_ids(
            training_dag,
            call_ids(training_graph, modules=frozenset({"maximum"}), ops=frozenset({"maximum"})),
        )
    )
    training = training_graph.materialize(training_graph.analyze(training_partition))
    assert training.run_training_prefix(x).metadata["supports_prefix_backward"]


def test_mlx_split_reconstructs_direct_multi_output() -> None:
    """MLX replay preserves a list returned directly by a split call."""

    mx = pytest.importorskip("mlx.core")

    def model(x: object) -> object:
        return list(mx.split(mx.maximum(x, 0), 2, axis=-1))

    x = mx.arange(8, dtype=mx.float32).reshape((2, 4)) - 3
    graph = prepare_graph(model, x, backend="mlx")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"maximum"}), ops=frozenset({"maximum"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
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

    def model(x: object) -> object:
        hidden = mx.maximum(x, 0)
        left, right = mx.split(hidden, 2, axis=-1)
        return {"left": left, "right": [right, mx.add(mx.multiply(left, 2), right)]}

    x = mx.arange(8, dtype=mx.float32).reshape((2, 4)) - 3
    graph = prepare_graph(model, x, backend="mlx")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"maximum"}), ops=frozenset({"maximum"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
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
        from torchlens.split import SplitFeatures, prepare_graph
        from dag_helpers import call_ids, close_call_ids


        import paddle
        from torchlens.split.adapters import resolve_split_adapter

        def model(x):
            return paddle.nn.functional.relu(x) * 2

        x = paddle.randn([2, 3], dtype="float32")
        adapter = resolve_split_adapter("paddle")

        assert adapter.supports_replay is True
        graph = prepare_graph(model, x, backend='paddle')
        dag = graph.compute_dag()
        partition = dag.to_partition(close_call_ids(dag, call_ids(graph, modules=frozenset({'relu'}), ops=frozenset({'functional.relu'}))))
        runtime = graph.materialize(graph.analyze(partition))
        replayed = runtime.replay(x)
        assert bool(paddle.allclose(replayed, model(x)).item())
        trainable_graph = prepare_graph(model, x, backend='paddle', features=SplitFeatures(training=True))
        trainable_dag = trainable_graph.compute_dag()
        trainable_partition = trainable_dag.to_partition(close_call_ids(trainable_dag, call_ids(trainable_graph, modules=frozenset({'relu'}), ops=frozenset({'functional.relu'}))))
        trainable = trainable_graph.materialize(trainable_graph.analyze(trainable_partition))
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

    def model(x: object) -> object:
        return tf.nn.relu(x) * 2.0 + 1.0

    x = tf.constant([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=tf.float32)
    adapter = resolve_split_adapter("tf")

    assert adapter.supports_replay is True
    graph = prepare_graph(model, x, backend="tf")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    replayed = runtime.replay(x)
    assert bool(tf.reduce_all(tf.abs(replayed - model(x)) < 1e-5).numpy())
    trainable_graph = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
    trainable_dag = trainable_graph.compute_dag()
    trainable_partition = trainable_dag.to_partition(
        close_call_ids(
            trainable_dag,
            call_ids(trainable_graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})),
        )
    )
    trainable = trainable_graph.materialize(trainable_graph.analyze(trainable_partition))
    assert trainable.segments.training_prefix is not None
