"""Compare 50-step FP64 Keras trajectories across native DAG recuts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from dag_helpers import call_ids, close_call_ids

from torchlens.split import SplitFeatures, prepare_graph
from torchlens.split.optimizers import get_optimizer_adapter

pytestmark = pytest.mark.tf_backend

_OPTIMIZERS = [
    pytest.param("SGD", {}, id="SGD"),
    pytest.param("SGD", {"momentum": 0.8}, id="SGD-momentum"),
    pytest.param("Adam", {}, id="Adam"),
    pytest.param("Adam", {"amsgrad": True}, id="Adam-AMSGrad"),
    pytest.param("AdamW", {"weight_decay": 0.025}, id="AdamW"),
]


def _mlp(tf: Any) -> Any:
    """Build the deterministic 128/64/10 model with stable Keras paths."""

    return tf.keras.Sequential(
        [
            tf.keras.Input((16,), dtype="float64"),
            tf.keras.layers.Dense(128, dtype="float64", activation="relu", name="stem"),
            tf.keras.layers.Dense(64, dtype="float64", activation="relu", name="middle"),
            tf.keras.layers.Dense(10, dtype="float64", name="head"),
        ],
        name="trajectory",
    )


def _parameters(runtime: Any, owner: str) -> dict[str, Any]:
    """Select the exact public parameter bindings for one optimizer."""

    return {
        key: values[owner]
        for key, values in runtime.parameter_bindings().items()
        if owner in values
    }


def _assert_states(runtime: Any, optimizers: Any, oracle: Any, reference: Any) -> dict[str, float]:
    """Compare all named slot tensors of both owners against one native optimizer."""

    params = {
        f"parameter:{value.path.split('/', 1)[1]}": value for value in reference.trainable_variables
    }
    adapter = get_optimizer_adapter("tf", oracle)
    expected = adapter.export_state(oracle, params)
    errors = {}
    for owner, optimizer in optimizers.items():
        actual = adapter.export_state(optimizer, _parameters(runtime, owner))
        assert int(optimizer.iterations) == int(oracle.iterations)
        assert float(optimizer.learning_rate) == float(oracle.learning_rate)
        for key, slots in actual.slots.items():
            for left, right in zip(slots, expected.slots[key], strict=True):
                assert left.slot_name == right.slot_name
                np.testing.assert_array_equal(left.payload, right.payload)
                error = float(np.max(np.abs(left.payload - right.payload)))
                errors[left.slot_name] = max(errors.get(left.slot_name, 0.0), error)
    return errors


def _max_error(left: Any, right: Any) -> float:
    """Check exact FP64 parity and return the measured maximum error."""

    left, right = np.asarray(left), np.asarray(right)
    np.testing.assert_array_equal(left, right)
    return float(np.max(np.abs(left - right)))


@pytest.mark.parametrize(("kind", "options"), _OPTIMIZERS)
@pytest.mark.parametrize("direction", ["suffix-to-prefix", "prefix-to-suffix"])
def test_tf_fp64_50_step_optimizer_trajectory(kind, options, direction, record_property) -> None:
    """Re-cut after 25 synchronized updates and compare every loss and final state."""

    tf = pytest.importorskip("tensorflow")
    tf.keras.utils.set_random_seed(2026)
    rng = np.random.default_rng(2026)
    with tf.device("/CPU:0"):
        model, reference = _mlp(tf), _mlp(tf)
        initial = [rng.normal(0, 0.06, array.shape) for array in model.get_weights()]
        model.set_weights(initial)
        reference.set_weights(initial)
        inputs = tf.constant(rng.normal(0, 0.5, (50, 4, 16)), tf.float64)
        targets = tf.constant(rng.normal(0, 0.25, (50, 4, 10)), tf.float64)
        graph = prepare_graph(model, inputs[0], backend="tf", features=SplitFeatures(training=True))
        dag = graph.compute_dag()

        def cut(layer: str):
            ids = frozenset(
                call_id
                for call_id in call_ids(graph, ops=frozenset({"relu"}))
                if graph.trace_graph.node_by_id[
                    next(node.output_node_ids[0] for node in dag.nodes if node.call_id == call_id)
                ].module_path
                == layer
            )
            assert len(ids) == 1
            return dag.to_partition(close_call_ids(dag, ids))

        first, second = (cut("stem"), cut("middle"))
        if direction == "prefix-to-suffix":
            first, second = second, first
        runtime = graph.materialize(graph.analyze(first))
        cls = getattr(tf.keras.optimizers, kind)
        optimizers = {owner: cls(learning_rate=0.005, **options) for owner in ("prefix", "suffix")}
        oracle = cls(learning_rate=0.005, **options)

        def loss_fn(output, target):
            return tf.reduce_mean(tf.square(output - target))

        losses = []
        max_loss_error = 0.0
        recut_before = {}
        recut_after = {}
        for step, (batch, target) in enumerate(zip(inputs, targets, strict=True)):
            if step == 15:
                for optimizer in (*optimizers.values(), oracle):
                    optimizer.learning_rate.assign(0.002)
            if step == 25:
                recut_before = _assert_states(runtime, optimizers, oracle, reference)
                plan = runtime.plan_optimizer_state_migration(second, optimizers=optimizers)
                assert plan.supported, plan.unsupported_reasons
                moved = {row.state_id for row in plan.entries if row.action == "transfer"}
                assert moved == {"parameter:middle/kernel", "parameter:middle/bias"}
                snapshot = runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
                runtime = runtime.repartition(second)
                result = runtime.restore_optimizer_state(snapshot)
                assert result.initialized_slots == 0
                optimizers = result.optimizers
                recut_after = _assert_states(runtime, optimizers, oracle, reference)
                assert {int(optimizer.iterations) for optimizer in optimizers.values()} == {25}
            with tf.GradientTape() as tape:
                expected = loss_fn(reference(batch), target)
            variables = reference.trainable_variables
            oracle.apply_gradients(zip(tape.gradient(expected, variables), variables, strict=True))
            boundary = runtime.run_training_prefix(batch)
            loss, grads = runtime.train_suffix(
                boundary, target, optimizer=optimizers["suffix"], loss_fn=loss_fn
            )
            runtime.backward_prefix(boundary, grads, optimizer=optimizers["prefix"])
            losses.append(float(loss))
            max_loss_error = max(max_loss_error, _max_error(loss, expected))

        assert {int(optimizer.iterations) for optimizer in optimizers.values()} == {50}
        slots = _assert_states(runtime, optimizers, oracle, reference)
        parameter_error = max(
            _max_error(a, b)
            for a, b in zip(model.get_weights(), reference.get_weights(), strict=True)
        )
        output = runtime.run_suffix(runtime.run_prefix(inputs[-1]))
        output_error = _max_error(output, reference(inputs[-1]))

    record_property(
        "trajectory",
        json.dumps(
            {
                "optimizer": kind,
                "options": options,
                "direction": direction,
                "losses": losses,
                "max_loss_error": max_loss_error,
                "max_parameter_error": parameter_error,
                "max_slot_errors": slots,
                "max_slot_errors_before_recut": recut_before,
                "max_slot_errors_after_recut": recut_after,
                "max_output_error": output_error,
                "iterations": {owner: int(opt.iterations) for owner, opt in optimizers.items()},
            }
        ),
    )
