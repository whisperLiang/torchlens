"""Optimizer snapshot integrity and transactional refusal cases for TensorFlow."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dag_helpers import call_ids, close_call_ids

from torchlens.split import OptimizerStateTensor, SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.optimizers import get_optimizer_adapter

pytestmark = pytest.mark.tf_backend


def _values(runtime: Any, optimizers: Any) -> tuple[np.ndarray, ...]:
    """Copy live model values and every source optimizer variable for mutation checks."""

    variables = [
        *runtime.model.variables,
        *(var for opt in optimizers.values() for var in opt.variables),
    ]
    return tuple(np.array(value.numpy(), copy=True) for value in variables)


def _unchanged(before: tuple[np.ndarray, ...], runtime: Any, optimizers: Any) -> None:
    """Require failure to preserve model values and optimizer variable inventories."""

    for old, new in zip(before, _values(runtime, optimizers), strict=True):
        np.testing.assert_array_equal(old, new)


def _graph(tf: Any, device: str = "/CPU:0") -> tuple[Any, Any, Any]:
    """Return a trained-state-compatible model, source runtime, and nontrivial cut."""

    with tf.device(device):
        model = tf.keras.Sequential(
            [
                tf.keras.Input((4,), dtype="float64"),
                tf.keras.layers.Dense(
                    4, dtype="float64", activation="relu", name="stem", kernel_initializer="ones"
                ),
                tf.keras.layers.Dense(2, dtype="float64", name="head", kernel_initializer="ones"),
            ],
            name="migration",
        )
        inputs = tf.ones((2, 4), tf.float64)
        graph = prepare_graph(model, inputs, backend="tf", features=SplitFeatures(training=True))
        dag = graph.compute_dag()
        cut = dag.to_partition(close_call_ids(dag, call_ids(graph, ops=frozenset({"relu"}))))
        return graph.materialize(graph.analyze(dag.to_partition(()))), cut, inputs


@pytest.fixture
def source():
    """Build both target owners with complete history and a reusable snapshot."""

    tf = pytest.importorskip("tensorflow")
    runtime, cut, inputs = _graph(tf)
    optimizers = {"suffix": tf.keras.optimizers.Adam(0.01)}
    runtime.train_suffix(
        runtime.run_training_prefix(inputs),
        tf.zeros((2, 2), tf.float64),
        optimizer=optimizers["suffix"],
    )
    plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
    assert plan.supported, plan.unsupported_reasons
    return SimpleNamespace(
        tf=tf,
        runtime=runtime,
        cut=cut,
        inputs=inputs,
        optimizers=optimizers,
        snapshot=runtime.export_optimizer_state(optimizers=optimizers, plan=plan),
        target=runtime.repartition(cut),
    )


def _tamper(snapshot, problem):
    """Corrupt exactly one requested field of an otherwise valid snapshot."""

    headers = {
        "backend": ("backend", "torch"),
        "partition": ("target_partition", "partition:wrong"),
        "schema": ("schema_version", 0),
        "graph": ("graph_id", "other-graph"),
        "manifest": ("checksum", "0" * 64),
    }
    if problem in headers:
        key, value = headers[problem]
        return replace(snapshot, **{key: value})
    if problem == "parameter":
        return replace(snapshot, parameter_specs={})
    state = snapshot.optimizers["prefix"]
    if problem == "config":
        config = {key: value for key, value in state.config.items() if key != "amsgrad"}
        return replace(
            snapshot, optimizers={**snapshot.optimizers, "prefix": replace(state, config=config)}
        )
    key = next(iter(state.slots))
    first, *rest = state.slots[key]
    if problem == "missing":
        changed = tuple(rest)
    elif problem == "unknown":
        wrong = OptimizerStateTensor.from_array(key, "unknown", first.payload)
        changed = (wrong, *rest)
    elif problem == "shape":
        wrong = OptimizerStateTensor.from_array(key, first.slot_name, np.zeros((3, 5)))
        changed = (wrong, *rest)
    elif problem == "dtype":
        wrong = OptimizerStateTensor.from_array(
            key, first.slot_name, first.payload.astype("float32")
        )
        changed = (wrong, *rest)
    else:
        wrong = replace(first, payload=first.payload + 1)
        changed = (wrong, *rest)
    return replace(
        snapshot,
        optimizers={
            **snapshot.optimizers,
            "prefix": replace(state, slots={**state.slots, key: changed}),
        },
    )


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        ("backend", "backend"),
        ("partition", "target_partition"),
        ("schema", "schema_version"),
        ("graph", "graph_id"),
        ("manifest", "manifest checksum"),
        ("parameter", "parameter IDs"),
        ("config", "Invalid optimizer configuration"),
        ("missing", "Missing or unknown"),
        ("unknown", "Missing or unknown"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("checksum", "Checksum"),
    ],
)
def test_invalid_snapshot_refuses_before_constructing_target(source, problem, reason) -> None:
    """A corrupt snapshot cannot call the optimizer factory or touch live source values."""

    before = _values(source.runtime, source.optimizers)

    def unexpected(parameters, config):
        pytest.fail("snapshot validation ran after optimizer construction")

    with pytest.raises(SplitUnsupportedError, match=reason):
        source.target.restore_optimizer_state(
            _tamper(source.snapshot, problem), optimizer_factories={"prefix": unexpected}
        )
    _unchanged(before, source.runtime, source.optimizers)


@pytest.mark.parametrize("divergence", ["iterations", "learning_rate", "config"])
def test_merge_refuses_incompatible_keras_histories(source, divergence) -> None:
    """A recut may not combine incompatible iterations, rates, or hyperparameters."""

    runtime = source.target
    optimizers = dict(runtime.restore_optimizer_state(source.snapshot).optimizers)
    if divergence == "iterations":
        optimizers["prefix"].iterations.assign_add(1)
    elif divergence == "learning_rate":
        optimizers["prefix"].learning_rate.assign(0.004)
    else:
        optimizers["prefix"].beta_1 = 0.65
    before = _values(runtime, optimizers)
    plan = runtime.plan_optimizer_state_migration(
        source.runtime.request.partition, optimizers=optimizers
    )
    assert not plan.supported
    assert divergence in " ".join(plan.unsupported_reasons)
    _unchanged(before, runtime, optimizers)


@pytest.mark.parametrize(
    "mode", ["EMA", "gradient accumulation", "loss scaling", "custom", "callable"]
)
def test_unsupported_keras_modes_refuse_without_mutation(source, mode) -> None:
    """Reject native modes that cannot be reconstructed with the complete supported schema."""

    keras = source.tf.keras

    class Custom(keras.optimizers.Adam):
        """An unregistered custom Adam subclass."""

    variants = {
        "EMA": lambda: keras.optimizers.Adam(use_ema=True),
        "gradient accumulation": lambda: keras.optimizers.Adam(gradient_accumulation_steps=2),
        "loss scaling": lambda: keras.optimizers.LossScaleOptimizer(keras.optimizers.Adam()),
        "custom": Custom,
        "callable": lambda: keras.optimizers.Adam(
            learning_rate=lambda: pytest.fail("evaluated rate")
        ),
    }
    optimizers = {"suffix": variants[mode]()}
    before = _values(source.runtime, optimizers)
    plan = source.runtime.plan_optimizer_state_migration(source.cut, optimizers=optimizers)
    assert not plan.supported
    assert "unsupported" in " ".join(
        plan.unsupported_reasons
    ) or "no registered adapter" in " ".join(plan.unsupported_reasons)
    _unchanged(before, source.runtime, optimizers)


def test_restore_exception_after_first_native_write_is_transactional(source, monkeypatch) -> None:
    """A failure after one staged optimizer was fully restored leaves its source intact."""

    adapter = get_optimizer_adapter("tf", source.optimizers["suffix"])
    native_restore = adapter.restore_state
    seen = []

    def fail_second(optimizer, state, params):
        seen.append(optimizer)
        if len(seen) == 2:
            raise RuntimeError("injected second optimizer failure")
        native_restore(optimizer, state, params)

    monkeypatch.setattr(adapter, "restore_state", fail_second)
    before = _values(source.runtime, source.optimizers)
    with pytest.raises(SplitUnsupportedError, match="injected second optimizer failure"):
        source.target.restore_optimizer_state(source.snapshot)
    assert len(seen) == 2
    assert int(seen[0].iterations) == 1
    _unchanged(before, source.runtime, source.optimizers)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("wrong_type", "incompatible Keras optimizer type"),
        ("already_built", "fresh, unbuilt"),
        ("wrong_config", "incompatible Keras optimizer configuration"),
    ],
)
def test_factory_must_build_fresh_compatible_keras_optimizer(source, model, message) -> None:
    """Validate target factory output before building or assigning its resources."""

    keras = source.tf.keras
    old = source.optimizers["suffix"]
    factories = {
        "wrong_type": lambda parameters, config: keras.optimizers.SGD(),
        "already_built": lambda parameters, config: old,
        "wrong_config": lambda parameters, config: keras.optimizers.Adam(0.9),
    }
    before = _values(source.runtime, source.optimizers)
    with pytest.raises(SplitUnsupportedError, match=message):
        source.target.restore_optimizer_state(
            source.snapshot, optimizer_factories={"prefix": factories[model]}
        )
    _unchanged(before, source.runtime, source.optimizers)


def test_shared_tensorflow_parameter_refuses_duplicate_optimizers() -> None:
    """One resource read on both sides cannot be assigned two update histories."""

    tf = pytest.importorskip("tensorflow")
    layer = tf.keras.layers.Dense(4, dtype="float64", activation="relu", name="shared")
    model_input = tf.keras.Input((4,), dtype="float64")
    model = tf.keras.Model(model_input, layer(layer(model_input)))
    graph = prepare_graph(
        model, tf.ones((2, 4), tf.float64), backend="tf", features=SplitFeatures(training=True)
    )
    dag = graph.compute_dag()
    first_relu = next(
        call.call_id
        for call in dag.nodes
        if graph.trace_graph.node_by_id[call.output_node_ids[0]].op_type == "relu"
    )
    cut = dag.to_partition(close_call_ids(dag, frozenset({first_relu})))
    runtime = graph.materialize(graph.analyze(cut))
    optimizers = {owner: tf.keras.optimizers.Adam() for owner in ("prefix", "suffix")}
    before = _values(runtime, optimizers)
    plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
    assert not plan.supported
    assert "unique named parameter ownership" in " ".join(plan.unsupported_reasons)
    _unchanged(before, runtime, optimizers)


def test_serialized_keras_schedule_and_unbuilt_optimizer_resume() -> None:
    """An unbuilt optimizer can initialize new slots without losing schedule semantics."""

    tf = pytest.importorskip("tensorflow")
    runtime, cut, inputs = _graph(tf)
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=2, decay_rate=0.5)
    optimizers = {"suffix": tf.keras.optimizers.Adam(learning_rate=schedule)}
    plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
    assert plan.supported, plan.unsupported_reasons
    assert {row.action for row in plan.entries} == {"initialize"}
    snapshot = runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
    target = runtime.repartition(cut)
    result = target.restore_optimizer_state(
        snapshot,
        optimizer_factories={
            "prefix": lambda parameters, config: tf.keras.optimizers.Adam.from_config(config)
        },
    )
    assert result.initialized_slots == 8
    assert result.restored_parameters == ()
    for _ in range(3):
        boundary = target.run_training_prefix(inputs)
        _, grads = target.train_suffix(
            boundary, tf.zeros((2, 2), tf.float64), optimizer=result.optimizers["suffix"]
        )
        target.backward_prefix(boundary, grads, optimizer=result.optimizers["prefix"])
    for optimizer in result.optimizers.values():
        assert int(optimizer.iterations) == 3
        assert float(optimizer.learning_rate) == float(schedule(3))


@pytest.mark.parametrize(
    ("source_device", "target_device"), [("/CPU:0", "/GPU:0"), ("/GPU:0", "/CPU:0")]
)
def test_tf_optimizer_moves_across_existing_native_devices(source_device, target_device) -> None:
    """Restore portable Keras optimizer bytes onto the native target resources."""

    tf = pytest.importorskip("tensorflow")
    if not tf.config.list_logical_devices("GPU"):
        pytest.skip("TensorFlow GPU unavailable")
    source, cut, inputs = _graph(tf, source_device)
    with tf.device(source_device):
        optimizer = tf.keras.optimizers.Adam()
        source.train_suffix(
            source.run_training_prefix(inputs), tf.zeros((2, 2), tf.float64), optimizer=optimizer
        )
        plan = source.plan_optimizer_state_migration(cut, optimizers={"suffix": optimizer})
        snapshot = source.export_optimizer_state(optimizers={"suffix": optimizer}, plan=plan)
    target, target_cut, inputs = _graph(tf, target_device)
    target.model.set_weights(source.model.get_weights())
    target = target.repartition(target_cut)
    with tf.device(target_device):
        restored = target.restore_optimizer_state(snapshot)
        for name, value in restored.optimizers.items():
            native = get_optimizer_adapter("tf", value).export_state(
                value,
                {
                    key: slots[name]
                    for key, slots in target.parameter_bindings().items()
                    if name in slots
                },
            )
            for key, slots in native.slots.items():
                assert tuple(slot.checksum for slot in slots) == tuple(
                    slot.checksum for slot in snapshot.optimizers[name].slots[key]
                )
        boundary = target.run_training_prefix(inputs)
        _, grads = target.train_suffix(
            boundary, tf.zeros((2, 2), tf.float64), optimizer=restored.optimizers["suffix"]
        )
        target.backward_prefix(boundary, grads, optimizer=restored.optimizers["prefix"])
        assert {int(value.iterations) for value in restored.optimizers.values()} == {2}
