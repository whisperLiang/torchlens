"""Native non-Torch split replay tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dag_helpers import call_ids, close_call_ids

from torchlens.split import prepare_graph
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
    graph = prepare_graph(model, x, backend="jax")
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition([dag.nodes[0].call_id])))

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
    graph = prepare_graph(model, x, backend="tinygrad")
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition([dag.nodes[0].call_id])))

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


def test_tinygrad_split_does_not_bind_derived_weight_to_its_buffers() -> None:
    """A derived model tensor must not replace each of its BUFFER operands."""

    Tensor = pytest.importorskip("tinygrad").Tensor

    class Model:
        """Hold one lazy weight formed from two independent buffers."""

        def __init__(self) -> None:
            """Leave the weight computation in the captured UOp graph."""

            self.weight = (
                Tensor([1.0, 2.0], device="CPU").realize()
                + Tensor([10.0, 20.0], device="CPU").realize()
            )

        def __call__(self, x: Any) -> Any:
            """Add the derived weight to the input."""

            return x + self.weight

    model = Model()
    x = Tensor([3.0, 4.0], device="CPU").realize()
    graph = prepare_graph(model, x, backend="tinygrad")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    assert runtime.replay(x).tolist() == model(x).tolist() == [14.0, 26.0]


def test_tf_split_replay_and_cache_roundtrip(tmp_path: Path) -> None:
    """TensorFlow raw-op prefix + suffix replay matches the original callable."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        hidden = tf.nn.relu(x)
        return hidden * 2.0 + 1.0

    x = tf.constant([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=tf.float32)
    graph = prepare_graph(model, x, backend="tf")
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition([dag.nodes[0].call_id])))

    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    assert bool(tf.reduce_all(tf.abs(runtime.run_suffix(boundary) - model(x)) < 1e-5).numpy())

    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    runtime.validate_boundary(loaded)
    assert bool(tf.reduce_all(tf.abs(runtime.run_suffix(loaded) - model(x)) < 1e-5).numpy())


@pytest.mark.parametrize(("axis", "parts"), [(-1, 2), (1, 3)])
def test_tf_native_split_replays_each_output_in_order(axis: int, parts: int) -> None:
    """Compare every division output and a fan-out consumer at four batch sizes."""

    tf = pytest.importorskip("tensorflow")

    def model(value: Any) -> tuple[Any, ...]:
        """Use all outputs and consume the first one again."""

        divided = tf.split(value, parts, axis=axis)
        return (*divided, divided[0] + divided[-1])

    with tf.device("/CPU:0"):
        values = tf.reshape(tf.range(48, dtype=tf.float64), (4, 12))
        graph = prepare_graph(model, values[:1], backend="tf")
        runtime = graph.materialize(graph.analyze(graph.compute_dag().to_partition(())))
        compiled_replay = tf.function(runtime.replay, autograph=False)
        for batch in (1, 2, 3, 4):
            inputs = values[:batch]
            expected = model(inputs)
            for replay in (runtime.replay, compiled_replay):
                for result, reference in zip(replay(inputs), expected, strict=True):
                    assert result.dtype == reference.dtype
                    tf.debugging.assert_equal(result, reference)


def test_tf_split_analysis_refuses_an_unregistered_raw_op() -> None:
    """Report an unsupported operation during Analyze, not during replay."""

    tf = pytest.importorskip("tensorflow")

    def model(value: Any) -> Any:
        """TensorFlow lowers this unequal split to SplitV, not Split."""

        return tf.split(value, [1, 3], axis=-1)

    with tf.device("/CPU:0"):
        inputs = tf.ones((1, 4), dtype=tf.float64)
        graph = prepare_graph(model, inputs, backend="tf")
        with pytest.raises(SplitUnsupportedError, match="SplitV") as caught:
            graph.analyze(graph.compute_dag().to_partition(()))
        assert caught.value.context.reason == "split preflight rejected"


@pytest.mark.parametrize("layout", ["omitted", None, "NHWC", b"NHWC"])
def test_tf_avg_pool_preserves_native_default_layout(layout: str | bytes | None) -> None:
    """Replay the real callback defaults and bytes through validation and Split."""

    tf = pytest.importorskip("tensorflow")
    from torchlens import trace
    from torchlens.backends.tf.validation import validate_tf_trace_detailed

    def model(value: Any) -> Any:
        if isinstance(layout, bytes):
            return tf.raw_ops.AvgPool(
                value=value,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding=b"VALID",
                data_format=layout,
            )
        kwargs = {} if layout == "omitted" else {"data_format": layout}
        return tf.nn.avg_pool(value, ksize=2, strides=2, padding="VALID", **kwargs)

    with tf.device("/CPU:0"):
        inputs = tf.reshape(tf.range(40, dtype=tf.float32), (1, 4, 5, 2))
        expected = model(inputs)
        result = validate_tf_trace_detailed(trace(model, inputs, backend="tf"))
        assert result.failures == ()
        assert result.replayed_histogram["AvgPool"] == 1
        graph = prepare_graph(model, inputs, backend="tf")
        runtime = graph.materialize(graph.analyze(graph.compute_dag().to_partition(())))
        tf.debugging.assert_equal(runtime.replay(inputs), expected)


def test_tf_avg_pool_refuses_unknown_native_layout() -> None:
    """Do not treat a nonempty invalid layout as the omitted default."""

    tf = pytest.importorskip("tensorflow")
    from torchlens import trace

    def model(value: Any) -> Any:
        return tf.nn.avg_pool(value, ksize=2, strides=2, padding="VALID", data_format="WRONG")

    with tf.device("/CPU:0"):
        inputs = tf.ones((1, 4, 5, 2), dtype=tf.float32)
        for run in (
            model,
            lambda x: trace(model, x, backend="tf"),
            lambda x: prepare_graph(model, x, backend="tf"),
        ):
            with pytest.raises(tf.errors.InvalidArgumentError, match="WRONG"):
                run(inputs)


@pytest.mark.parametrize(
    ("pool_type", "padding"),
    [
        ("avg_pool", "SAME"),
        ("avg_pool", "VALID"),
        ("max_pool", "SAME"),
        ("max_pool", "VALID"),
        ("max_pool", "EXPLICIT"),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("ksize", "strides"), [(2, 2), ((2, 3), (1, 2))])
def test_tf_pooling_replays_native_arguments(
    pool_type: str, padding: str, dtype: str, ksize: Any, strides: Any
) -> None:
    """Keep the native pooling signature and padding on both sides of a cut."""

    tf = pytest.importorskip("tensorflow")
    op = getattr(tf.nn, pool_type)

    def model(value: Any) -> Any:
        """Pool ordered values, then consume the result in a separate call."""

        mode: Any = padding
        if padding == "EXPLICIT":
            mode = [[0, 0], [0, 1], [1, 0], [0, 0]]
        return op(value, ksize, strides, mode, data_format="NHWC") + 0.5

    with tf.device("/CPU:0"):
        values = tf.reshape(tf.range(160, dtype=tf.as_dtype(dtype)), (4, 4, 5, 2))
        graph = prepare_graph(model, values[:1], backend="tf")
        dag = graph.compute_dag()
        raw_name = "AvgPool" if pool_type == "avg_pool" else "MaxPool"
        prefix_ids = close_call_ids(dag, call_ids(graph, ops=frozenset({raw_name.lower()})))
        assert prefix_ids
        runtime = graph.materialize(graph.analyze(dag.to_partition(())))
        for partition in (dag.to_partition(()), dag.to_partition(prefix_ids)):
            runtime = runtime.repartition(partition)
            for batch in (1, 2, 3, 4):
                inputs = values[:batch]
                actual = runtime.replay(inputs)
                assert actual.dtype == inputs.dtype
                tf.debugging.assert_equal(actual, model(inputs))


@pytest.mark.parametrize(("raw_name", "input_name"), [("AvgPool", "value"), ("MaxPool", "input")])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_tf_pooling_replays_nchw(
    raw_name: str, input_name: str, dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the NCHW kernel on GPU and the exact ABI on CPU."""

    tf = pytest.importorskip("tensorflow")
    from torchlens.backends.tf._raw_op_replay import _pause_tf_replay, _prepare_native_call
    from torchlens.backends.tf.op_callback_capture import TFInputCapture, TFOpCapture

    native = getattr(tf.raw_ops, raw_name)
    attrs = {
        "ksize": [1, 1, 2, 2],
        "strides": [1, 1, 2, 2],
        "padding": "VALID",
        "data_format": "NCHW",
    }
    gpu = bool(tf.config.list_logical_devices("GPU"))
    nhwc = {"data_format": "NHWC", "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]}

    def replay_on_cpu(**kwargs: Any) -> Any:
        """Check NCHW arguments before substituting the equivalent NHWC kernel."""

        for name in ("ksize", "strides"):
            assert tuple(kwargs[name]) == tuple(attrs[name])
        assert kwargs["data_format"] == attrs["data_format"]
        assert kwargs["padding"] == attrs["padding"]
        kwargs[input_name] = tf.transpose(kwargs[input_name], (0, 2, 3, 1))
        return tf.transpose(native(**(kwargs | nhwc)), (0, 3, 1, 2))

    if not gpu:
        monkeypatch.setattr(tf.raw_ops, raw_name, replay_on_cpu)
    device = "/GPU:0" if gpu else "/CPU:0"
    with tf.device(device):
        values = tf.reshape(tf.range(160, dtype=tf.as_dtype(dtype)), (4, 2, 4, 5))
        if gpu:

            def model(x: Any) -> Any:
                return native(**{input_name: x, **attrs})

            graph = prepare_graph(model, values[:1], backend="tf")
            dag = graph.compute_dag()
            runtime = graph.materialize(graph.analyze(dag.to_partition(())))
        else:
            record = TFInputCapture(0, None, "constant", None, values, None)
            capture = TFOpCapture("", raw_name, attrs, 0, (record,), None)
            call, _ = _prepare_native_call(tf, capture)
        for batch in (1, 2, 3, 4):
            inputs = values[:batch]
            reference = native(
                **{input_name: tf.transpose(inputs, (0, 2, 3, 1)), "padding": "VALID", **nhwc}
            )
            if gpu:
                actual = runtime.replay(inputs)
            else:
                with _pause_tf_replay():
                    actual = call._execute([inputs])
            assert device[1:] in actual.device
            tf.debugging.assert_equal(actual, tf.transpose(reference, (0, 3, 1, 2)))


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
        return {0: result, 1: [result + 1.0, (result + 2.0, {3: result + 3.0})]}

    x = jnp.ones((2, 3))
    graph = prepare_graph(model, x, backend="jax")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"max"}), ops=frozenset({"max"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    assert runtime.batch_validation["status"] == "passed"
    for batch in (1, 2, 4):
        replay_x = jnp.arange(batch * 3, dtype=jnp.float32).reshape(batch, 3) - 2.0
        output = runtime.replay(replay_x)
        assert isinstance(output, dict)
        assert isinstance(output[1], list)
        assert isinstance(output[1][1], tuple)
        assert isinstance(output[1][1][1], dict)
        expected = jnp.maximum(replay_x, 0) * 2.0
        for actual, value in (
            (output[0], expected),
            (output[1][0], expected + 1.0),
            (output[1][1][0], expected + 2.0),
            (output[1][1][1][3], expected + 3.0),
        ):
            assert actual.shape == value.shape
            assert actual.dtype == value.dtype
            assert bool(jnp.allclose(actual, value))


def test_tf_split_replay_preserves_integer_dict_and_list_containers() -> None:
    """TensorFlow replay preserves container kinds when integer keys are present."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        hidden = tf.nn.relu(x)
        result = hidden * 2.0
        return {0: result, 1: [result + 1.0]}

    x = tf.ones((2, 3), dtype=tf.float32)
    graph = prepare_graph(model, x, backend="tf")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
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
    graph = prepare_graph(model, x, backend="jax")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

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
    graph = prepare_graph(model, x, backend="jax")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            call_ids(graph, ops=frozenset({"broadcast_in_dim"})),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

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
    graph = prepare_graph(model, x, backend="jax")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

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
    graph = prepare_graph(model, x, backend="tf")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

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
    graph = prepare_graph(model, x, backend="tf")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

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
    graph = prepare_graph(model, x, backend="tinygrad")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape_3"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

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

    graph = prepare_graph(model, tinygrad.Tensor.ones(2, 3, 2).realize(), backend="tinygrad")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape_3"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
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
    graph = prepare_graph(model, x, backend="tinygrad")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape_3"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

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
    graph = prepare_graph(model, x, backend="tinygrad")
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag, call_ids(graph, modules=frozenset({"reshape_3"}), ops=frozenset({"reshape"}))
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

    with pytest.raises((SplitUnsupportedError, ValueError, RuntimeError)):
        runtime.replay(tinygrad.Tensor.ones(2, 4, 2).realize())
