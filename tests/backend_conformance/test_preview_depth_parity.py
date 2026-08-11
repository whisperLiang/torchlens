"""Preview parity for compute_input_output_distances (torch Step 4).

Every preview backend must honor the same public opt-in torch honors:
input/output hop distances plus ancestor/descendant lineage sets, with the
effective value stored on ``trace.mark_layer_depths``. Single-pass previews
must also store the EFFECTIVE ``recurrence_detection`` (False — they never
group), while JAX keeps its real grouping value.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.backend_parity


def _assert_depths(trace, expect_recurrence: bool) -> None:
    ops = list(trace.layer_list)
    with_depth = [
        op for op in ops if getattr(op, "min_distance_from_input", None) is not None
    ]
    assert with_depth, "depth flood populated no op"
    assert trace.mark_layer_depths is True
    assert trace.recurrence_detection is expect_recurrence
    assert any(getattr(op, "input_ancestors", None) for op in ops)
    assert any(getattr(op, "output_descendants", None) for op in ops)


@pytest.mark.backend_paddle
def test_paddle_depth_parity() -> None:
    paddle = pytest.importorskip("paddle")
    import torchlens as tl

    class M(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = paddle.nn.Linear(4, 3)
            self.l2 = paddle.nn.Linear(3, 2)

        def forward(self, x):
            return self.l2(paddle.nn.functional.relu(self.l1(x)))

    trace = tl.trace(
        M(), paddle.ones([1, 4]), backend="paddle", compute_input_output_distances=True
    )
    _assert_depths(trace, expect_recurrence=False)
    base = tl.trace(M(), paddle.ones([1, 4]), backend="paddle")
    assert base.mark_layer_depths is False
    assert base.recurrence_detection is False


@pytest.mark.backend_tinygrad
def test_tinygrad_depth_parity() -> None:
    pytest.importorskip("tinygrad")
    import torchlens as tl
    from tinygrad import Tensor

    def model(x):
        return ((x + 1.0).relu() * 2.0).sum()

    trace = tl.trace(
        model,
        Tensor([1.0, -2.0, 3.0]),
        backend="tinygrad",
        compute_input_output_distances=True,
    )
    _assert_depths(trace, expect_recurrence=False)


@pytest.mark.backend_jax
def test_jax_depth_parity() -> None:
    jnp = pytest.importorskip("jax.numpy")
    import torchlens as tl

    def model(x):
        return jnp.tanh(x @ jnp.ones((4, 3))) @ jnp.ones((3, 2))

    trace = tl.trace(
        model, jnp.ones((1, 4)), backend="jax", compute_input_output_distances=True
    )
    _assert_depths(trace, expect_recurrence=True)


@pytest.mark.backend_mlx
def test_mlx_depth_parity() -> None:
    mx = pytest.importorskip("mlx.core")
    mnn = pytest.importorskip("mlx.nn")
    import torchlens as tl

    class M(mnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = mnn.Linear(4, 3)
            self.l2 = mnn.Linear(3, 2)

        def __call__(self, x):
            return self.l2(mnn.relu(self.l1(x)))

    trace = tl.trace(
        M(), mx.ones((1, 4)), backend="mlx", compute_input_output_distances=True
    )
    _assert_depths(trace, expect_recurrence=False)


@pytest.mark.tf_backend
def test_tf_depth_parity() -> None:
    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    import torchlens as tl

    model = keras.Sequential(
        [keras.layers.Dense(3, activation="relu"), keras.layers.Dense(2)]
    )
    inputs = tf.ones((1, 4))
    model(inputs)
    trace = tl.trace(
        model, inputs, backend="tf", compute_input_output_distances=True
    )
    _assert_depths(trace, expect_recurrence=False)
