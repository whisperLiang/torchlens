"""TensorFlow state ownership follows executed resource reads."""

from __future__ import annotations

import pytest
from dag_helpers import call_ids, close_call_ids

from torchlens.split import SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError

pytestmark = pytest.mark.tf_backend


def test_keras_cut_inside_dense_uses_each_variables_actual_read() -> None:
    """A module-wide parameter annotation cannot assign its bias to a kernel read."""
    tf = pytest.importorskip("tensorflow")
    layer = tf.keras.layers.Dense(3, dtype="float64", name="dense_state")
    model = tf.keras.Sequential([tf.keras.Input((4,), dtype="float64"), layer])
    x = tf.ones((2, 4), dtype=tf.float64)
    graph = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
    dag = graph.compute_dag()
    cut = dag.to_partition(close_call_ids(dag, call_ids(graph, ops=frozenset({"matmul"}))))
    runtime = graph.materialize(graph.analyze(cut))

    bindings = runtime.parameter_bindings()
    kernel, bias = "parameter:dense_state/kernel", "parameter:dense_state/bias"
    assert dict(bindings[kernel]) == {"prefix": layer.kernel}
    assert dict(bindings[bias]) == {"suffix": layer.bias}
    for owner, segment in (
        ("prefix", runtime.segments.training_prefix),
        ("suffix", runtime.segments.suffix),
    ):
        assert {id(values[owner]) for values in bindings.values() if owner in values} == {
            id(value) for value in segment._trainable_variables(model)
        }

    target = dag.to_partition(())
    plan = runtime.plan_state_migration(target)
    assert plan.supported
    assert {row.state_id: row.to_owners for row in plan.entries} == {
        kernel: ("suffix",),
        bias: ("suffix",),
    }
    recut = runtime.repartition(target)
    assert set(recut.parameter_bindings()) == set(bindings)
    assert recut.parameter_bindings()[kernel]["suffix"] is layer.kernel

    layer.trainable = False
    frozen = {row.state_id: row for row in recut.parameter_ownership()}
    assert set(frozen) == set(bindings)
    assert not frozen[kernel].trainable and not frozen[bias].trainable


@pytest.mark.parametrize("explicit_reads", [True, False])
def test_tf_module_aliases_and_non_trainable_state(explicit_reads: bool) -> None:
    """Two attribute aliases resolve to one resource on both sides of the cut."""
    tf = pytest.importorskip("tensorflow")

    class Model(tf.Module):
        """Read the same parameter twice and use a non-trainable offset."""

        def __init__(self) -> None:
            """Establish an aliased variable and one buffer."""
            super().__init__()
            self.scale = self.tied = tf.Variable(2.0, name="scale")
            self.offset = tf.Variable(0.25, trainable=False, name="offset")

        def __call__(self, x):
            """Read the alias separately so both segments use the parameter."""
            scale = self.scale.read_value() if explicit_reads else self.scale
            tied = self.tied.read_value() if explicit_reads else self.tied
            return tf.math.tanh(x * scale) * tied + self.offset

    model = Model()
    graph = prepare_graph(
        model, tf.ones((2, 3)), backend="tf", features=SplitFeatures(training=True)
    )
    dag = graph.compute_dag()
    cut = dag.to_partition(close_call_ids(dag, call_ids(graph, ops=frozenset({"tanh"}))))
    runtime = graph.materialize(graph.analyze(cut))
    if not explicit_reads:
        with pytest.raises(SplitUnsupportedError, match="resource read"):
            runtime.parameter_ownership()
        return
    rows = {row.state_id: row for row in runtime.parameter_ownership()}
    assert rows["parameter:scale"].aliases == ("scale", "tied")
    assert rows["parameter:scale"].owners == ("prefix", "suffix")
    owners = runtime.parameter_bindings()["parameter:scale"]
    assert owners["prefix"] is owners["suffix"] is model.scale
    assert runtime.state_bindings()["buffer:offset"]["suffix"] is model.offset

    model.scale = model.tied = tf.Variable(3.0, name="scale")
    with pytest.raises(SplitUnsupportedError, match="resource"):
        runtime.parameter_ownership()


def test_tf_module_refuses_unresolved_variable_name_collision() -> None:
    """Two distinct unnamed paths cannot be silently merged into one optimizer ID."""

    tf = pytest.importorskip("tensorflow")

    class Model(tf.Module):
        """Store two distinct resources under the same native variable name."""

        def __init__(self) -> None:
            super().__init__()
            self.values = [tf.Variable(2.0, name="duplicate"), tf.Variable(3.0, name="duplicate")]

        def __call__(self, x):
            return x * self.values[0].read_value() + self.values[1].read_value()

    graph = prepare_graph(
        Model(), tf.ones((2, 3)), backend="tf", features=SplitFeatures(training=True)
    )
    cut = graph.compute_dag().to_partition(())
    with pytest.raises(SplitUnsupportedError, match="state_identity_conflict"):
        graph.materialize(graph.analyze(cut)).parameter_ownership()
