"""TensorFlow split-training parameter replay tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from dag_helpers import call_ids, close_call_ids

from torchlens.split import SplitFeatures, prepare_graph
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
        graph = prepare_graph(split_model, x, backend="tf", features=SplitFeatures(training=True))
        dag = graph.compute_dag()
        partition = dag.to_partition(
            close_call_ids(
                dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"}))
            )
        )
        runtime = graph.materialize(graph.analyze(partition))
        full_opt = tf.keras.optimizers.SGD(learning_rate=0.05)
        suffix_opt = tf.keras.optimizers.SGD(learning_rate=0.05)
        prefix_opt = tf.keras.optimizers.SGD(learning_rate=0.05)

        with tf.GradientTape() as tape:
            full_loss = tf.reduce_mean(tf.math.squared_difference(model(x), y))
        full_grads = tape.gradient(full_loss, model.trainable_variables)
        full_opt.apply_gradients(zip(full_grads, model.trainable_variables, strict=True))

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


@pytest.mark.parametrize("update", [False, True])
def test_tf_suffix_joint_gradients_keep_sparse_and_disconnected_sources(
    update: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Traverse the loss once and apply only connected suffix variable gradients."""

    tf = pytest.importorskip("tensorflow")

    class Branches(tf.Module):
        """Use two differentiable roots and a sparse embedding read."""

        def __init__(self) -> None:
            """Initialize the used embedding and an independent weight."""
            super().__init__()
            self.embedding = tf.Variable(tf.ones((4, 2), tf.float64), name="embedding")
            self.unused = tf.Variable(tf.ones((2,), tf.float64), name="unused")

        def __call__(self, x: Any, indices: Any) -> tuple[Any, Any]:
            """Return both branches while the loss consumes only the first."""
            return (
                tf.nn.relu(x) + tf.gather(self.embedding.read_value(), indices),
                tf.math.tanh(x) * self.unused.read_value(),
            )

    model = Branches()
    x = tf.constant([[-1.0, 2.0], [3.0, -4.0]], dtype=tf.float64)
    indices = tf.constant([0, 2])
    target = tf.zeros_like(x)
    graph = prepare_graph(
        model,
        (x, indices),
        backend="tf",
        features=SplitFeatures(training=True, batch_axes={"/args/0": 0, "/args/1": 0}),
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, ops=frozenset({"relu", "tanh"})))
    )
    runtime = graph.materialize(graph.analyze(partition))

    def objective(output: tuple[Any, Any], y: Any) -> Any:
        """Use only the first output so the other root and weight disconnect."""
        return tf.reduce_mean(tf.math.squared_difference(output[0], y))

    hidden = tf.nn.relu(x)
    with tf.GradientTape() as tape:
        tape.watch(hidden)
        expected_loss = objective((hidden + tf.gather(model.embedding, indices), None), target)
    expected_root_grad, expected_var_grad = tape.gradient(expected_loss, [hidden, model.embedding])
    loss_variable = tf.Variable(1.0, dtype=tf.float64)

    def loss_with_external_variable(output: tuple[Any, Any], y: Any) -> Any:
        """Read a loss-owned variable outside the segment's optimizer scope."""
        return objective(output, y) + tf.square(loss_variable - 1.0)

    queries: list[tuple[int, set[object]]] = []
    native_gradient = tf.GradientTape.gradient

    def observe_gradient(tape: Any, loss: Any, sources: list[Any]) -> Any:
        """Record the eager request while leaving its differentiation to TensorFlow."""
        grads = native_gradient(tape, loss, sources)
        queries.append((len(sources), {variable.ref() for variable in tape.watched_variables()}))
        return grads

    class RecordingSGD(tf.keras.optimizers.SGD):
        """Assert the native optimizer receives the original sparse gradient."""

        def apply_gradients(self, grads_and_vars: Any) -> Any:
            """Apply the one connected gradient and preserve its sparse type."""
            [(grad, variable)] = grads_and_vars
            assert variable is model.embedding
            assert isinstance(grad, tf.IndexedSlices)
            return super().apply_gradients([(grad, variable)])

    optimizer = RecordingSGD(0.05) if update else None
    expected_weights = tf.identity(model.embedding)
    if update:
        expected_weights -= tf.cast(0.05, tf.float64) * tf.convert_to_tensor(expected_var_grad)
    boundary = runtime.run_training_prefix(x, indices)
    assert sum(value.dtype.is_floating for value in boundary.tensors.values()) == 2
    monkeypatch.setattr(tf.GradientTape, "gradient", observe_gradient)
    result = runtime.train_suffix_result(
        boundary,
        target,
        loss_fn=loss_with_external_variable,
        optimizer=optimizer,
    )
    expected_watched = {model.embedding.ref(), model.unused.ref()} if update else set()
    assert queries == [(2 + len(expected_watched), expected_watched)]
    assert result.optimizer_applied is update
    if optimizer is not None:
        assert int(optimizer.iterations) == 1
    assert len(result.boundary_grads) == 1
    _assert_tf_close(tf, next(iter(result.boundary_grads.values())), expected_root_grad)
    _assert_tf_close(tf, result.loss, expected_loss)
    _assert_tf_close(tf, model.embedding, expected_weights)
    _assert_tf_close(tf, model.unused, tf.ones_like(model.unused))


@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam"])
@pytest.mark.parametrize("compiled", [False, True])
def test_tf_keras_split_training_matches_native_trajectory(
    optimizer_name: str, compiled: bool
) -> None:
    """Match 50 FP64 Keras updates, including graph gradients applied in eager mode."""

    tf = pytest.importorskip("tensorflow")
    with tf.device("/CPU:0"):
        model = tf.keras.Sequential(
            [
                tf.keras.Input((3,), dtype=tf.float64),
                tf.keras.layers.Dense(4, activation="relu", dtype=tf.float64),
                tf.keras.layers.Dense(2, dtype=tf.float64),
            ]
        )
        initial = [np.linspace(-0.2, 0.3, w.size).reshape(w.shape) for w in model.get_weights()]
        model.set_weights(initial)
        reference = tf.keras.models.clone_model(model)
        reference.set_weights(initial)
        inputs = tf.reshape(tf.linspace(tf.constant(-0.5, tf.float64), 0.5, 24), (8, 3))
        targets = tf.ones((8, 2), dtype=tf.float64)
        graph = prepare_graph(
            model, inputs[:1], backend="tf", features=SplitFeatures(training=True)
        )
        dag = graph.compute_dag()
        partition = dag.to_partition(close_call_ids(dag, call_ids(graph, ops=frozenset({"relu"}))))
        runtime = graph.materialize(graph.analyze(partition))

        optimizer_cls = getattr(tf.keras.optimizers, optimizer_name)
        prefix_opt, suffix_opt, reference_opt = [optimizer_cls(0.01) for _ in range(3)]
        suffix_variables: list[Any] = []

        class CaptureUpdates:
            """Return the symbolic suffix gradients to the eager caller."""

            def apply_gradients(self, pairs: list[tuple[Any, Any]]) -> None:
                """Record one request without creating optimizer state inside the graph."""
                grads, variables = zip(*pairs, strict=True)
                self.grads = grads
                suffix_variables[:] = variables

        @tf.function(autograph=False)
        def graph_suffix(values: dict[str, Any], y: Any) -> tuple[Any, ...]:
            """Trace the same public training call using a batched boundary."""
            batch_size = int(y.shape[0])
            boundary = replace(boundaries[batch_size], tensors=values)
            updates = CaptureUpdates()
            result = runtime.train_suffix_result(boundary, y, optimizer=updates)
            assert result.optimizer_applied
            return result.loss, result.boundary_grads, updates.grads

        boundaries = {b: runtime.run_prefix(inputs[:b]) for b in (1, 2, 4, 8)}
        for step in range(50):
            batch = (1, 2, 4, 8)[step % 4]
            x, y = inputs[:batch], targets[:batch]
            with tf.GradientTape() as tape:
                reference_loss = tf.reduce_mean(tf.math.squared_difference(reference(x), y))
            variables = reference.trainable_variables
            reference_opt.apply_gradients(zip(tape.gradient(reference_loss, variables), variables))

            boundary = runtime.run_training_prefix(x)
            if compiled:
                loss, root_grads, var_grads = graph_suffix(boundary.tensors, y)
                suffix_opt.apply_gradients(zip(var_grads, suffix_variables, strict=True))
            else:
                result = runtime.train_suffix_result(boundary, y, optimizer=suffix_opt)
                assert result.optimizer_applied
                loss, root_grads = result.loss, result.boundary_grads
            assert runtime.backward_prefix(boundary, root_grads, optimizer=prefix_opt)
            np.testing.assert_allclose(loss.numpy(), reference_loss.numpy(), atol=1e-8, rtol=1e-7)
            for actual, expected in zip(model.weights, reference.weights, strict=True):
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-8, rtol=1e-7)

        assert int(prefix_opt.iterations) == int(suffix_opt.iterations) == 50


def test_tf_integer_boundary_supports_suffix_updates_without_root_gradients() -> None:
    """Train from a nondifferentiable input, including an empty gradient request."""

    tf = pytest.importorskip("tensorflow")

    class Scale(tf.Module):
        """Multiply an integer input by a trainable floating point variable."""

        def __init__(self) -> None:
            """Create a scalar parameter."""
            super().__init__()
            self.scale = tf.Variable(2.0, dtype=tf.float64)

        def __call__(self, x: Any) -> Any:
            """Cast the input only in the suffix."""
            return tf.cast(x, tf.float64) * self.scale.read_value()

    model = Scale()
    x = tf.constant([[1], [2]])
    target = tf.zeros((2, 1), dtype=tf.float64)
    graph = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
    runtime = graph.materialize(graph.analyze(graph.compute_dag().to_partition(())))
    boundary = runtime.run_training_prefix(x)
    assert all(value.dtype.is_integer for value in boundary.tensors.values())

    no_update = runtime.train_suffix_result(boundary, target)
    assert no_update.boundary_grads == {}
    assert no_update.optimizer_applied is False
    optimizer = tf.keras.optimizers.SGD(0.1)
    disconnected = runtime.train_suffix_result(
        boundary,
        target,
        loss_fn=lambda output, _: tf.reduce_sum(tf.stop_gradient(output)),
        optimizer=optimizer,
    )
    assert disconnected.optimizer_applied is False
    assert int(optimizer.iterations) == 0

    updated = runtime.train_suffix_result(boundary, target, optimizer=optimizer)
    assert updated.boundary_grads == {}
    assert updated.optimizer_applied is True
    assert int(optimizer.iterations) == 1
    assert float(model.scale) == pytest.approx(1.0)

    old_scale = model.scale
    model.scale = tf.Variable(3.0, dtype=tf.float64)
    with pytest.raises(SplitUnsupportedError, match="prepare a new split graph"):
        runtime.train_suffix_result(boundary, target, optimizer=optimizer)
    assert float(old_scale) == pytest.approx(1.0)

    refreshed = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
    recaptured = refreshed.materialize(refreshed.analyze(refreshed.compute_dag().to_partition(())))
    assert recaptured.train_suffix_result(
        recaptured.run_training_prefix(x), target, optimizer=tf.keras.optimizers.SGD(0.1)
    ).optimizer_applied
    assert float(model.scale) == pytest.approx(1.5)


def test_tf_training_binds_shared_variables_once_and_respects_live_freeze() -> None:
    """Repartition preserves live weights and selects only currently trainable variables."""

    tf = pytest.importorskip("tensorflow")
    inputs = tf.keras.Input((2,), dtype=tf.float64)
    frozen = tf.keras.layers.Dense(2, dtype=tf.float64, kernel_initializer="ones")
    shared = tf.keras.layers.Dense(2, dtype=tf.float64, kernel_initializer="ones")
    model = tf.keras.Model(inputs, shared(shared(frozen(inputs))))
    x = tf.constant([[1.0, -0.5]], dtype=tf.float64)
    graph = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition(c.call_id for c in dag.nodes)))
    runtime = runtime.repartition(dag.to_partition(()))
    frozen.trainable = False
    expected_frozen = frozen.get_weights()
    optimizer = tf.keras.optimizers.SGD(0.01)

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(x)))
    expected_shared = [
        variable.numpy() - 0.01 * grad.numpy()
        for variable, grad in zip(
            shared.trainable_variables, tape.gradient(loss, shared.trainable_variables), strict=True
        )
    ]
    result = runtime.train_suffix_result(
        runtime.run_training_prefix(x), tf.zeros_like(model(x)), optimizer=optimizer
    )

    assert result.optimizer_applied
    assert int(optimizer.iterations) == 1
    for actual, expected in zip(frozen.get_weights(), expected_frozen, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(shared.get_weights(), expected_shared, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-7)


def test_tf_training_boundary_cache_strips_gradient_tape(tmp_path) -> None:
    """TensorFlow training boundaries save as suffix-only cache payloads."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        hidden = tf.nn.relu(x)
        return hidden * 2.0

    x = tf.ones((2, 2), dtype=tf.float32)
    target = tf.zeros_like(x)
    graph = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    boundary = runtime.run_training_prefix(x)

    runtime.save_boundary(boundary, tmp_path / "boundary")
    loaded = runtime.load_boundary(tmp_path / "boundary")

    assert loaded.metadata.get("supports_prefix_backward") is False
    assert "tf_tape" not in loaded.metadata
    _loss, grads = runtime.train_suffix(loaded, target)
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(loaded, grads)
