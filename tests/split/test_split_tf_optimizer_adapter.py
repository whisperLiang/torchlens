"""Native Keras adapter slot and factory contracts."""

from __future__ import annotations

import numpy as np
import pytest

from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.optimizers import get_optimizer_adapter

pytestmark = pytest.mark.tf_backend


@pytest.mark.parametrize(
    ("kind", "options", "slots"),
    [
        ("SGD", {}, ()),
        ("SGD", {"momentum": 0.8}, ("momentum",)),
        ("Adam", {}, ("momentum", "velocity")),
        ("Adam", {"amsgrad": True}, ("momentum", "velocity", "velocity_hat")),
        ("AdamW", {}, ("momentum", "velocity")),
    ],
)
@pytest.mark.parametrize("built", [False, True])
def test_keras_state_follows_resource_identity(kind, options, slots, built) -> None:
    """Reversed variable order and identical shapes cannot swap moment histories."""
    tf = pytest.importorskip("tensorflow")
    opt = getattr(tf.keras.optimizers, kind)(learning_rate=0.01, **options)
    variables = {key: tf.Variable([1.0, 1.0], dtype="float64") for key in ("a", "b")}
    if built:
        opt.apply_gradients(
            [(tf.constant([g, g], tf.float64), variables[k]) for k, g in (("b", 2.0), ("a", 1.0))]
        )
        opt.learning_rate.assign(0.0025)
    adapter = get_optimizer_adapter("tensorflow", opt)
    snapshot = adapter.export_state(opt, variables)
    assert all(expected == slots for expected in snapshot.expected_slots.values())
    assert all(len(state) == (len(slots) if built else 0) for state in snapshot.slots.values())
    target = {key: tf.Variable(v.numpy(), dtype=v.dtype) for key, v in variables.items()}
    adapter.validate_restore(snapshot, target)
    restored = adapter.build_optimizer(snapshot, target, None)
    adapter.restore_state(restored, snapshot, target)
    assert int(restored.iterations) == int(opt.iterations) == int(built)
    assert float(restored.learning_rate) == float(opt.learning_rate)
    assert restored is not opt
    if built:
        final = adapter.export_state(restored, target)
        for key in target:
            for src, dst in zip(snapshot.slots[key], final.slots[key], strict=True):
                np.testing.assert_array_equal(dst.payload, src.payload)


def test_keras_rejects_unmapped_optimizer_variables() -> None:
    """Live state outside the verified slot and global map prevents export."""
    tf = pytest.importorskip("tensorflow")
    opt = tf.keras.optimizers.Adam()
    variable = tf.Variable(1.0)
    opt.build([variable])
    opt.add_variable(shape=(), name="unknown")
    with pytest.raises(SplitUnsupportedError, match="unmapped"):
        get_optimizer_adapter("tf", opt).export_state(opt, {"parameter:weight": variable})


def test_keras_rejects_parameter_with_identical_shape_and_different_resource() -> None:
    """Shape and dtype equality cannot substitute for native optimizer membership."""

    tf = pytest.importorskip("tensorflow")
    optimizer = tf.keras.optimizers.Adam()
    original, replacement = tf.Variable([1.0]), tf.Variable([1.0])
    optimizer.apply_gradients([(tf.constant([1.0]), original)])
    before = [np.array(var.numpy(), copy=True) for var in optimizer.variables]
    with pytest.raises(SplitUnsupportedError, match="ownership differs"):
        get_optimizer_adapter("tf", optimizer).export_state(optimizer, {"weight": replacement})
    for old, new in zip(before, optimizer.variables, strict=True):
        np.testing.assert_array_equal(old, new.numpy())
