"""Singleton JAX batched products preserve their batch axes during split capture."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.split._jax_capture import batch_stable_matmul


@pytest.mark.parametrize("point", ["25%", "50%", "75%"])
def test_jax_attention_replays_singleton_and_larger_batches(point: str) -> None:
    """Compare nonuniform attention outputs across three cut positions.

    Parameters
    ----------
    point:
        Relative position of the split in the attention graph.
    """

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    def model(x: Any) -> Any:
        """Use both operator and public-function forms of batched matmul."""

        scores = (x @ jnp.swapaxes(x, -1, -2)) / 2
        return jnp.matmul(jax.nn.softmax(scores, axis=-1), x) + x

    with jax.default_device(jax.devices("cpu")[0]):
        runtime = tl.split.prepare(model, jnp.ones((2, 3, 4)), split_request(point, backend="jax"))
        assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
        for batch in (1, 2, 3):
            x = jnp.arange(batch * 12, dtype=jnp.float32).reshape(batch, 3, 4) / 20
            np.testing.assert_allclose(runtime.replay(x), model(x), rtol=1e-5, atol=1e-6)


def test_jax_matmul_scope_delegates_broadcasts_and_restores_on_error() -> None:
    """Keep native promotion/broadcast behavior and restore both entry points."""

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax._src.numpy import tensor_contractions

    public = jnp.matmul
    operator = tensor_contractions.matmul
    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3)
        b = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)
        expected = public(a, b)
        with pytest.raises(RuntimeError, match="sentinel"), batch_stable_matmul():
            np.testing.assert_allclose(jnp.matmul(a, b), expected)
            mixed = jnp.ones((1, 3, 2), dtype=jnp.float16)
            np.testing.assert_allclose(jnp.matmul(a, mixed), public(a, mixed))
            raise RuntimeError("sentinel")
        assert jnp.matmul is public
        assert tensor_contractions.matmul is operator
