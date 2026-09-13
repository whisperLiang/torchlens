"""Scoped TensorFlow lowering for batch-polymorphic split captures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any


def preserve_probe_layer_names(model: Any, clone: Any) -> None:
    """Preserve capture addresses on an isolated Keras reconstruction.

    Parameters
    ----------
    model, clone:
        Original model and its independent deep copy. Subclass constructors
        can regenerate default layer names; only the clone's names are changed.
    """

    import tensorflow as tf

    if not isinstance(model, tf.keras.layers.Layer):
        return
    seen: set[int] = set()

    def restore(source: Any, target: Any) -> None:
        """Copy names through corresponding layers, preserving shared children."""

        if id(source) in seen:
            return
        seen.add(id(source))
        if source is target or type(source) is not type(target):
            raise ValueError("Keras probe copy did not preserve independent layer structure.")
        source_layers = tuple(getattr(source, "_layers", ()))
        target_layers = tuple(getattr(target, "_layers", ()))
        if len(source_layers) != len(target_layers):
            raise ValueError("Keras probe copy changed the layer inventory.")
        target.name = source.name
        for source_layer, target_layer in zip(source_layers, target_layers, strict=True):
            restore(source_layer, target_layer)

    restore(model, clone)


@contextmanager
def batch_stable_keras_add() -> Iterator[None]:
    """Keep dense residual and trailing-bias additions batch independent.

    Yields
    ------
    None
        A capture scope whose residual additions keep the same native topology.

    Notes
    -----
    Keras optimizes a tensor shaped ``(1, 1, 1, C)`` to a bias vector. For a
    residual addition that tensor is actually a batch-bearing activation: the
    canonical B=1 graph would capture Squeeze/BiasAdd, which is inexecutable at
    B=2. Equal-shaped, same-dtype dense tensors require neither broadcasting nor
    dtype promotion, so ``tf.add`` expresses their exact operation at every B.
    A trailing-axis bias follows the same rule even when its B=1 shape happens
    to equal the activation shape. Other operands still use Keras' original
    dtype/sparse/broadcast rules.
    The original function is restored on both success and failure; ordinary
    TensorFlow captures and user model calls outside this scope are unchanged.
    """

    import tensorflow as tf
    from keras.src.backend.tensorflow import numpy as keras_numpy

    original = keras_numpy.add

    @wraps(original)
    def add(x1: Any, x2: Any) -> Any:
        """Delegate dense equal-shape addition without singleton-batch fusion.

        Parameters
        ----------
        x1, x2:
            Keras addition operands, with native semantics preserved.

        Returns
        -------
        Any
            The sum, using Keras' original implementation for other operands.
        """

        if (
            isinstance(x1, tf.Tensor)
            and isinstance(x2, tf.Tensor)
            and x1.dtype == x2.dtype
            and x1.shape.rank is not None
            and x1.shape.rank > 1
            and x1.shape.is_fully_defined()
            and x2.shape.is_fully_defined()
            and (
                x1.shape == x2.shape
                or (
                    x2.shape.rank is not None
                    and 0 < x2.shape.rank <= x1.shape.rank
                    and x2.shape[-1] == x1.shape[-1]
                    and all(int(dim) == 1 for dim in x2.shape[:-1])
                )
            )
        ):
            return tf.add(x1, x2)
        return original(x1, x2)

    try:
        keras_numpy.add = add
        yield
    finally:
        if keras_numpy.add is add:
            keras_numpy.add = original
