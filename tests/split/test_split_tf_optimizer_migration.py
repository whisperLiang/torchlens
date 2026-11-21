"""End-to-end TensorFlow optimizer migration through SplitRuntime."""

from __future__ import annotations

import numpy as np
import pytest
from dag_helpers import call_ids, close_call_ids

from torchlens.split import SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError

pytestmark = pytest.mark.tf_backend


def test_runtime_exports_and_restores_the_keras_plan() -> None:
    """Split one trained suffix optimizer into two fresh optimizer owners."""
    tf = pytest.importorskip("tensorflow")
    inputs = tf.ones((2, 4), tf.float64)
    model = tf.keras.Sequential(
        [
            tf.keras.Input((4,), dtype="float64"),
            tf.keras.layers.Dense(5, dtype="float64", name="stem", activation="relu"),
            tf.keras.layers.Dense(3, dtype="float64", name="head"),
        ]
    )
    graph = prepare_graph(model, inputs, backend="tf", features=SplitFeatures(training=True))
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition(())))
    cut = dag.to_partition(close_call_ids(dag, call_ids(graph, ops=frozenset({"relu"}))))
    optimizer = tf.keras.optimizers.Adam(0.01)
    optimizers = {"suffix": optimizer}
    target = tf.ones((2, 3), tf.float64)
    runtime.train_suffix(runtime.run_training_prefix(inputs), target, optimizer=optimizer)
    plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
    assert plan.supported, plan.unsupported_reasons
    assert {row.action for row in plan.entries} == {"keep", "transfer"}
    snapshot = runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
    recut = runtime.repartition(cut)
    result = recut.restore_optimizer_state(snapshot)
    assert result.restored_slots == 8
    assert result.initialized_slots == 0
    assert all(
        value is not optimizer and int(value.iterations) == 1
        for value in result.optimizers.values()
    )

    runtime.train_suffix(runtime.run_training_prefix(inputs), target, optimizer=optimizer)
    with pytest.raises(SplitUnsupportedError, match="changed"):
        runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
    assert int(optimizer.iterations) == 2
    assert all(int(value.iterations) == 1 for value in result.optimizers.values())
    for state in snapshot.optimizers.values():
        np.testing.assert_array_equal(
            next(slot.payload for slot in state.global_state if slot.slot_name == "iterations"), 1
        )
