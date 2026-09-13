"""Singleton-batch Keras residuals stay real additions in split replay."""

from __future__ import annotations

from typing import Any

import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.split._tf_capture import batch_stable_keras_add


@pytest.mark.tf_backend
@pytest.mark.parametrize("point", ["before:add", "after:add"])
def test_singleton_spatial_residual_replays_and_restores_keras(point: str) -> None:
    """Retain batch-changing residual shape, values, and original Keras callable.

    Parameters
    ----------
    point:
        Cut on either side of the residual addition.
    """

    tf = pytest.importorskip("tensorflow")
    from keras.src.backend.tensorflow import numpy as keras_numpy

    inputs = tf.keras.Input(shape=(1, 1, 4))
    residual = tf.keras.layers.Add()([inputs, inputs * 2])
    model = tf.keras.Model(inputs, tf.keras.layers.ReLU()(residual))
    original = keras_numpy.add
    with tf.device("/CPU:0"):
        runtime = tl.split.prepare(model, tf.ones((2, 1, 1, 4)), split_request(point, backend="tf"))
        assert keras_numpy.add is original
        assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
        for batch in (1, 2, 3):
            value = tf.reshape(tf.range(batch * 4, dtype=tf.float32), (batch, 1, 1, 4))
            tf.debugging.assert_equal(runtime.replay(value), model(value))
            assert keras_numpy.add is original


@pytest.mark.tf_backend
def test_keras_add_scope_delegates_other_operands_and_restores_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not replace dtype/broadcast behavior or leak the scoped function.

    Parameters
    ----------
    monkeypatch:
        Restore the delegated Keras function after inspecting its calls.
    """

    tf = pytest.importorskip("tensorflow")
    from keras.src.backend.tensorflow import numpy as keras_numpy

    original = keras_numpy.add
    calls: list[tuple[Any, Any]] = []

    def delegate(x1: Any, x2: Any) -> Any:
        """Record operands delegated to the original Keras implementation."""

        calls.append((x1, x2))
        return original(x1, x2)

    monkeypatch.setattr(keras_numpy, "add", delegate)
    with pytest.raises(RuntimeError, match="sentinel"):
        with batch_stable_keras_add():
            tf.debugging.assert_equal(keras_numpy.add(tf.ones((2, 4)), tf.ones((2, 1))), 2.0)
            tf.debugging.assert_equal(keras_numpy.add(tf.ones((1, 4)), 2.0), 3.0)
            assert len(calls) == 2
            raise RuntimeError("sentinel")
    assert keras_numpy.add is delegate


@pytest.mark.tf_backend
def test_keras_subclass_probe_preserves_generated_layer_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reconstructed subclass keeps independent weights and stable capture paths.

    Parameters
    ----------
    monkeypatch:
        Restore the test-only Keras serialization registration.
    """

    tf = pytest.importorskip("tensorflow")

    class ProbeModel(tf.keras.Model):
        """Recreate unnamed layers through Keras' subclass configuration path."""

        def __init__(self, **kwargs: Any) -> None:
            """Let Keras choose globally unique layer names."""

            super().__init__(**kwargs)
            self.hidden = tf.keras.layers.Dense(4, activation="relu")
            self.head = tf.keras.layers.Dense(2)

        def call(self, x: Any) -> Any:
            """Apply two independently named dense layers."""

            return self.head(self.hidden(x))

    monkeypatch.setitem(tf.keras.utils.get_custom_objects(), "ProbeModel", ProbeModel)
    with tf.device("/CPU:0"):
        model = ProbeModel()
        sample = tf.ones((2, 3))
        model(sample)
        names = tuple(layer.name for layer in model.layers)
        weights = tuple(variable.numpy().copy() for variable in model.weights)
        runtime = tl.split.prepare(model, sample, split_request("50%", backend="tf"))
        assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
        assert tuple(layer.name for layer in model.layers) == names
        for variable, expected in zip(model.weights, weights, strict=True):
            tf.debugging.assert_equal(variable, expected)
        for batch in (1, 2, 3):
            x = tf.reshape(tf.range(batch * 3, dtype=tf.float32), (batch, 3)) / 10
            tf.debugging.assert_near(runtime.replay(x), model(x))
