"""Singleton JAX batched products preserve their batch axes during split capture."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import numpy as np
import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.split._jax_capture import batch_stable_matmul


def _jax_matmul_bindings() -> list[tuple[Any, str]]:
    """Identify public and installed operator bindings for restoration assertions.

    Returns
    -------
    list[tuple[Any, str]]
        Module/class owners and their directly owned matmul attribute names.
    """

    import jax.numpy as jnp
    from jax._src import core
    from jax._src.array import ArrayImpl
    from jax._src.numpy import tensor_contractions

    bindings = [(jnp, "matmul"), (tensor_contractions, "matmul")]
    for name in ("matmul", "rmatmul"):
        bindings.append((ArrayImpl, f"__{name}__"))
        for class_name in ("ShapedArray", "DShapedArray"):
            owner = getattr(core, class_name, None)
            if owner is not None and f"_{name}" in vars(owner):
                bindings.append((owner, f"_{name}"))
    return bindings


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
    """Keep native promotion/broadcast behavior and restore nested operator scopes."""

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    bindings = _jax_matmul_bindings()
    originals = [vars(owner)[name] for owner, name in bindings]
    public = jnp.matmul
    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3)
        b = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)
        expected = public(a, b)
        with pytest.raises(RuntimeError, match="sentinel"), batch_stable_matmul():
            np.testing.assert_allclose(jnp.matmul(a, b), expected)
            np.testing.assert_allclose(a @ b, expected)
            mixed = jnp.ones((1, 3, 2), dtype=jnp.float16)
            np.testing.assert_allclose(jnp.matmul(a, mixed), public(a, mixed))
            np.testing.assert_allclose(a @ mixed, public(a, mixed))
            outer = [vars(owner)[name] for owner, name in bindings]
            with pytest.raises(RuntimeError, match="inner"), batch_stable_matmul():
                np.testing.assert_allclose(a @ b, expected)
                raise RuntimeError("inner")
            assert all(
                vars(owner)[name] is prior
                for (owner, name), prior in zip(bindings, outer, strict=True)
            )
            raise RuntimeError("sentinel")
        assert all(
            vars(owner)[name] is prior
            for (owner, name), prior in zip(bindings, originals, strict=True)
        )


@pytest.mark.parametrize("reflected", [False, True])
def test_jax_matmul_operators_preserve_batch_axes_eager_and_traced(
    reflected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eager and traced operators keep batch dimensions for B=1 and B=2.

    Parameters
    ----------
    reflected:
        Exercise NumPy's delegation to the JAX right operand's reflected operator.
    monkeypatch:
        Record the native dot dimensions without changing the calculation.
    """

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    original_dot = jax.lax.dot_general
    dimensions = []

    def dot(a: Any, b: Any, dimension_numbers: Any, **kwargs: Any) -> Any:
        """Record contracting and batch dimensions while executing the native dot.

        Parameters
        ----------
        a, b, dimension_numbers, **kwargs:
            Unchanged native dot arguments.
        """

        dimensions.append(dimension_numbers)
        return original_dot(a, b, dimension_numbers, **kwargs)

    monkeypatch.setattr(jax.lax, "dot_general", dot)
    with jax.default_device(jax.devices("cpu")[0]), batch_stable_matmul():
        for batch in (1, 2):
            left = np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 3)
            right = jnp.arange(batch * 12, dtype=jnp.float32).reshape(batch, 3, 4)

            def product(x: Any) -> Any:
                """Exercise a tracer as the operator receiver.

                Parameters
                ----------
                x:
                    Left operand for matmul, right operand for reflected matmul.
                """

                return left @ x if reflected else x @ right

            argument = right if reflected else jnp.asarray(left)
            expected = np.matmul(left, np.asarray(right))
            dimensions.clear()
            np.testing.assert_allclose(product(argument), expected)
            assert dimensions == [(((2,), (1,)), ((0,), (0,)))]
            graph = jax.make_jaxpr(product)(argument).jaxpr
            assert [equation.primitive.name for equation in graph.eqns] == ["dot_general"]
            assert graph.eqns[0].params["dimension_numbers"] == (
                ((2,), (1,)),
                ((0,), (0,)),
            )


def test_jax_matmul_scope_preserves_operand_dispatch() -> None:
    """Keep custom reflection, JAX conversion, and invalid-operand refusals."""

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    sentinel = object()

    class Reflected:
        """An unknown operand must get its own reflected operation."""

        def __rmatmul__(self, other: Any) -> Any:
            """Return the dispatch sentinel.

            Parameters
            ----------
            other:
                Left operand whose operator declined this object.
            """

            return sentinel

    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.ones((1, 2, 3), dtype=jnp.float32)
        b = jnp.ones((1, 3, 4), dtype=jnp.float32)

        class Convertible:
            """An operand recognized through JAX's array conversion protocol."""

            def __jax_array__(self) -> Any:
                """Return the native right operand."""

                return b

        with batch_stable_matmul():
            assert a @ Reflected() is sentinel
            assert type(a).__matmul__(a, object()) is NotImplemented
            assert type(a).__rmatmul__(a, object()) is NotImplemented
            np.testing.assert_allclose(a @ Convertible(), a @ b)
            with pytest.raises(TypeError, match="unsupported operand"):
                a @ [[1]]
            with pytest.raises(TypeError, match="unsupported operand"):
                [[1]] @ a


def test_jax_matmul_concurrent_scopes_serialize_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A waiting capture enters only after an exceptional owner restores bindings.

    Parameters
    ----------
    monkeypatch:
        Observe lock contention without changing which lock protects the scopes.
    """

    pytest.importorskip("jax")
    from torchlens.split import _jax_capture

    first_inside = Event()
    first_release = Event()
    second_attempted = Event()
    second_blocked = []
    lock = _jax_capture._MATMUL_CAPTURE_LOCK
    bindings = _jax_matmul_bindings()
    originals = [vars(owner)[name] for owner, name in bindings]

    class ObservedLock:
        """Report the second thread's actual nonblocking acquisition attempt."""

        def __enter__(self) -> None:
            """Acquire the production lock, exposing contention deterministically."""

            if first_inside.is_set():
                acquired = lock.acquire(blocking=False)
                second_blocked.append(not acquired)
                second_attempted.set()
                if acquired:
                    return
            lock.acquire()

        def __exit__(self, *exc: Any) -> None:
            """Release the production lock even when the capture raises.

            Parameters
            ----------
            *exc:
                Context-manager exception information.
            """

            lock.release()

    monkeypatch.setattr(_jax_capture, "_MATMUL_CAPTURE_LOCK", ObservedLock())

    def first_capture() -> None:
        """Keep the first scope open until the second thread attempts entry."""

        with batch_stable_matmul():
            first_inside.set()
            assert first_release.wait(timeout=10), "first capture was never released"
            raise RuntimeError("thread sentinel")

    def second_capture() -> None:
        """Check the second scope owns temporary bindings after acquiring the lock."""

        with batch_stable_matmul():
            assert first_release.is_set(), "concurrent matmul scopes overlapped"
            assert all(
                vars(owner)[name] is not prior
                for (owner, name), prior in zip(bindings, originals, strict=True)
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_capture)
        try:
            assert first_inside.wait(timeout=10), "first capture never entered"
            second = executor.submit(second_capture)
            assert second_attempted.wait(timeout=10), "second capture never attempted entry"
            assert second_blocked == [True], "capture lock did not cover the complete scope"
        finally:
            first_release.set()
        with pytest.raises(RuntimeError, match="thread sentinel"):
            first.result(timeout=10)
        second.result(timeout=10)
    assert all(
        vars(owner)[name] is prior for (owner, name), prior in zip(bindings, originals, strict=True)
    )
