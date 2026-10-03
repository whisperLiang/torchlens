"""MLX split gradients and private optimizer state against native unsplit training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from v2_helpers import split_request

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
nn = pytest.importorskip("mlx.nn", exc_type=ImportError)

import mlx.optimizers as optim  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError  # noqa: E402

pytestmark = pytest.mark.backend_mlx


class _Mlp(nn.Module):
    """Two parameterized layers with an explicit, differentiable cut."""

    def __init__(self) -> None:
        """Initialize fixed weights so numerical comparisons are deterministic."""

        super().__init__()
        self.hidden = nn.Linear(4, 6)
        self.output = nn.Linear(6, 2)
        self.hidden.weight = mx.arange(24, dtype=mx.float32).reshape(6, 4) / 30 - 0.3
        self.hidden.bias = mx.arange(6, dtype=mx.float32) / 10
        self.output.weight = mx.arange(12, dtype=mx.float32).reshape(2, 6) / 20 - 0.2
        self.output.bias = mx.array([0.1, -0.1])

    def __call__(self, x: Any) -> Any:
        """Evaluate the MLP around its maximum boundary."""

        return self.output(mx.maximum(self.hidden(x), 0))


def _input(batch: int = 3) -> Any:
    """Return mixed-sign batch rows that exercise the activation derivative."""

    return mx.arange(batch * 4, dtype=mx.float32).reshape(batch, 4) / 5 - 1


def _loss(output: Any, targets: Any) -> Any:
    """Compute the same mean squared loss for split and native paths."""

    return mx.mean(mx.square(output - targets))


def _assert_close(left: Any, right: Any) -> None:
    """Compare evaluated values and gradients at float32 tolerances."""

    mx.eval(left, right)
    assert left.shape == right.shape
    assert bool(mx.allclose(left, right, atol=2e-5, rtol=2e-5))


@pytest.mark.parametrize("optimizer_cls", [optim.SGD, optim.Adam])
def test_mlx_optimizer_steps_match_unsplit_and_preserve_source(optimizer_cls: Any) -> None:
    """Repeated split steps match native losses, gradients and updated model values."""

    from mlx.utils import tree_flatten

    model, reference = _Mlp(), _Mlp()
    x, targets = _input(), mx.array([[0.2, -0.5], [1, 0.3], [-0.7, 0.4]])
    source = dict(tree_flatten(model.parameters()))
    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    prefix_optimizer = optimizer_cls(learning_rate=0.01)
    suffix_optimizer = optimizer_cls(learning_rate=0.01)
    reference_optimizer = optimizer_cls(learning_rate=0.01)
    native_grad = nn.value_and_grad(reference, lambda a, b: _loss(reference(a), b))
    for _step in range(3):
        expected_input = mx.grad(lambda a: _loss(reference(a), targets))(x)
        expected_loss, expected_grads = native_grad(x, targets)
        boundary = runtime.run_training_prefix(x)
        suffix = runtime.train_suffix_result(boundary, targets, optimizer=suffix_optimizer)
        prefix = runtime.backward_prefix(
            boundary, suffix.boundary_grads, optimizer=prefix_optimizer
        )
        _assert_close(suffix.loss, expected_loss)
        _assert_close(prefix["inputs"][0], expected_input)
        actual_grads = {**prefix["parameter_grads"], **suffix.parameter_grads}
        flat_expected = dict(tree_flatten(expected_grads))
        assert actual_grads.keys() == flat_expected.keys()
        for name in flat_expected:
            _assert_close(actual_grads[name], flat_expected[name])
        reference_optimizer.update(reference, expected_grads)
        mx.eval(reference.parameters())
        _assert_close(runtime.replay(x), reference(x))
        assert prefix["optimizer_applied"] and suffix.optimizer_pending
    for name, value in tree_flatten(model.parameters()):
        assert value is source[name]
    assert runtime.prefix_parameters() and runtime.suffix_parameters()
    migrated = runtime.with_placement(tl.split.PlacementPlan.on("cpu"))
    recut = migrated.at(tl.split.before("maximum"))
    _assert_close(migrated.replay(x), reference(x))
    _assert_close(recut.replay(x), reference(x))


def test_mlx_gradients_match_at_every_cut_and_changed_batch() -> None:
    """Multi-output cuts retain all contributions to the native input gradient."""

    def model(x: Any) -> Any:
        """Merge two native split outputs with different derivative weights."""

        left, right = mx.split(mx.maximum(x, 0), 2, axis=-1)
        return mx.add(mx.multiply(left, 2), right)

    seed = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    for candidate in seed.split_points().supported:
        runtime = seed.at(candidate.point)
        x = _input(5)
        targets = mx.ones((5, 2))
        expected = mx.grad(lambda a, targets=targets: _loss(model(a), targets))(x)
        boundary = runtime.run_training_prefix(x)
        result = runtime.train_suffix_result(boundary, targets)
        prefix = runtime.backward_prefix(boundary, result.boundary_grads)
        _assert_close(prefix["inputs"][0], expected)
        assert result.parameter_grads == {}
        assert prefix["parameter_grads"] == {}
        assert not prefix["optimizer_applied"] and not result.optimizer_applied


def test_mlx_nested_and_keyword_input_gradients() -> None:
    """Functional prefix differentiation returns gradients in each original input tree."""

    def model(pair: dict[str, Any], *, offset: Any) -> Any:
        """Consume nested positional arrays and an independent keyword array."""

        return mx.multiply(mx.maximum(mx.add(pair["a"], offset), 0), pair["b"])

    x = _input(2)
    pair, offset = {"b": mx.add(x, 2), "a": x}, mx.ones_like(x)
    runtime = tl.split.prepare(
        model,
        pair,
        split_request(
            "after:maximum",
            backend="mlx",
            trainable=True,
            batch_axes={"/args/0/a": 0, "/args/0/b": 0, "/kwargs/offset": 0},
        ),
        input_kwargs={"offset": offset},
    )
    targets = mx.zeros_like(x)
    _value, expected = mx.value_and_grad(
        lambda a, b: _loss(model(a, offset=b), targets), argnums=(0, 1)
    )(pair, offset)
    boundary = runtime.run_training_prefix(pair)
    suffix = runtime.train_suffix_result(boundary, targets, loss_fn=_loss)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    for name in pair:
        _assert_close(prefix["inputs"][0][name], expected[0][name])
    _assert_close(prefix["input_kwargs"]["offset"], expected[1])


def test_mlx_shared_input_gradient_survives_explicit_placement() -> None:
    """Moving tied input leaves preserves the single AD root and accumulated gradient."""

    def model(x: Any, y: Any) -> Any:
        """Consume one array through two aliased argument positions."""

        return mx.multiply(mx.maximum(x, 0), y)

    x = _input(2)
    runtime = tl.split.prepare(
        model,
        (x, x),
        split_request(
            "after:maximum",
            backend="mlx",
            trainable=True,
            placement=tl.split.PlacementPlan.on("cpu"),
        ),
    )
    targets = mx.ones_like(x)
    boundary = runtime.run_training_prefix(x, x)
    suffix = runtime.train_suffix_result(boundary, targets)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    expected = mx.grad(lambda a: _loss(model(a, a), targets))(x)
    _assert_close(prefix["inputs"][0], expected)
    assert prefix["inputs"][0] is prefix["inputs"][1]


def test_mlx_native_conv_and_layernorm_parameter_gradients() -> None:
    """Native modules requiring typed self attributes replay and differentiate correctly."""

    from mlx.utils import tree_flatten

    class Model(nn.Module):
        """Use convolution and normalization on opposite sides of the cut."""

        def __init__(self) -> None:
            """Create native modules with non-array configuration attributes."""

            super().__init__()
            self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
            self.norm = nn.LayerNorm(3)

        def __call__(self, x: Any) -> Any:
            """Reduce spatial axes after normalization."""

            return mx.mean(self.norm(mx.maximum(self.conv(x), 0)), axis=(1, 2))

    model = Model()
    x = mx.arange(64, dtype=mx.float32).reshape(2, 4, 4, 2) / 50
    runtime = tl.split.prepare(
        model, x, split_request("after:maximum", backend="mlx", trainable=True)
    )
    targets = mx.ones((2, 3))
    _value, expected = nn.value_and_grad(model, lambda: _loss(model(x), targets))()
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(boundary, targets)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    actual = {**prefix["parameter_grads"], **suffix.parameter_grads}
    assert actual.keys() == dict(tree_flatten(expected)).keys()
    for name, value in tree_flatten(expected):
        _assert_close(actual[name], value)


def test_mlx_training_rejects_detached_stale_and_invalid_gradients() -> None:
    """Prefix recomputation refuses boundaries that cannot attest the original forward."""

    runtime = tl.split.prepare(
        _Mlp(), _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    x, targets = _input(2), mx.ones((2, 2))
    detached = runtime.run_prefix(x)
    result = runtime.train_suffix_result(detached, targets)
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(detached, result.boundary_grads)
    boundary = runtime.run_training_prefix(x)
    result = runtime.train_suffix_result(boundary, targets)
    key = next(iter(result.boundary_grads))
    with pytest.raises(SplitBoundaryError, match="Unknown"):
        runtime.backward_prefix(boundary, {"invented": mx.ones((2, 6))})
    with pytest.raises(SplitBoundaryError, match="shape or dtype"):
        runtime.backward_prefix(boundary, {key: mx.ones((2, 7))})
    runtime.backward_prefix(boundary, result.boundary_grads, optimizer=optim.SGD(0.01))
    with pytest.raises(SplitBoundaryError, match="updated training prefix"):
        runtime.backward_prefix(boundary, result.boundary_grads)


def test_mlx_default_classification_loss_and_frozen_parameters() -> None:
    """Default cross entropy differentiates only unfrozen native parameter leaves."""

    model = _Mlp()
    model.hidden.freeze(keys="weight")
    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    x, targets = _input(3), mx.array([0, 1, 0])
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(boundary, targets)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    _assert_close(suffix.loss, nn.losses.cross_entropy(model(x), targets, reduction="mean"))
    assert set(prefix["parameter_grads"]) == {"hidden.bias"}


def test_mlx_custom_nested_loss_and_non_scalar_refusal() -> None:
    """Explicit scalar loss functions consume nested final outputs, refusing vectors."""

    def model(x: Any) -> Any:
        """Retain two output containers and a non-array literal."""

        y = mx.maximum(x, 0)
        return {"pair": mx.split(y, 2, axis=-1), "literal": "kept"}

    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    boundary = runtime.run_training_prefix(_input(2))
    result = runtime.train_suffix_result(
        boundary,
        None,
        loss_fn=lambda output, _: mx.sum(output["pair"][0]) + mx.sum(output["pair"][1]),
    )
    prefix = runtime.backward_prefix(boundary, result.boundary_grads)
    _assert_close(prefix["inputs"][0], (_input(2) > 0).astype(mx.float32))
    with pytest.raises(SplitUnsupportedError, match="scalar"):
        runtime.train_suffix_result(boundary, None, loss_fn=lambda output, _: output["pair"][0])


def test_mlx_training_boundary_snapshots_input_values() -> None:
    """In-place caller input changes cannot alter a previously captured prefix VJP."""

    def model(x: Any) -> Any:
        """Use a sign-sensitive derivative with a sign-insensitive prefix value."""

        return mx.add(mx.multiply(x, x), 1)

    x = _input(2)
    original = x.astype(x.dtype)
    mx.eval(original)
    runtime = tl.split.prepare(
        model, x, split_request("after:multiply", backend="mlx", trainable=True)
    )
    targets = mx.zeros_like(x)
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(boundary, targets)
    x[:] = -x
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    expected = mx.grad(lambda a: _loss(model(a), targets))(original)
    _assert_close(prefix["inputs"][0], expected)


def test_mlx_cached_training_boundary_is_suffix_only(tmp_path: Path) -> None:
    """Training cache payloads lose prefix identity and input snapshots before saving."""

    from torchlens.split.cache import load_boundary, save_boundary

    runtime = tl.split.prepare(
        _Mlp(),
        _input(2),
        split_request("after:maximum", backend="mlx", trainable=True, boundary_cache=True),
    )
    boundary = runtime.run_training_prefix(_input(2))
    path = tmp_path / "training.boundary"
    save_boundary(boundary, path, adapter=runtime.adapter)
    loaded = load_boundary(path, adapter=runtime.adapter)
    assert not loaded.metadata["supports_prefix_backward"]
    assert "mlx_prefix_owner" not in loaded.metadata
    assert "prefix_inputs" not in loaded.metadata
    suffix = runtime.train_suffix_result(loaded, mx.ones((2, 2)))
    assert suffix.parameter_grads
    with pytest.raises(SplitUnsupportedError, match="run_training_prefix"):
        runtime.backward_prefix(loaded, suffix.boundary_grads)


def test_mlx_prefix_rejects_inplace_parameter_mutation() -> None:
    """External writes through exposed parameter arrays invalidate the training boundary."""

    runtime = tl.split.prepare(
        _Mlp(), _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    boundary = runtime.run_training_prefix(_input(2))
    suffix = runtime.train_suffix_result(boundary, mx.ones((2, 2)))
    parameter = runtime.prefix_parameters()[0]
    parameter[:] = -parameter
    with pytest.raises(SplitBoundaryError, match="prefix state changed"):
        runtime.backward_prefix(boundary, suffix.boundary_grads)


def test_mlx_native_attention_parameter_and_input_gradients() -> None:
    """Nested native submodules retain their classes and functional parameter roots."""

    from mlx.utils import tree_flatten

    class Model(nn.Module):
        """Place a native attention call with typed child modules in the suffix."""

        def __init__(self) -> None:
            """Create one native attention block."""

            super().__init__()
            self.attention = nn.MultiHeadAttention(dims=4, num_heads=2)

        def __call__(self, x: Any) -> Any:
            """Attend over the prefix activation and pool its sequence dimension."""

            hidden = mx.maximum(x, 0)
            return mx.mean(self.attention(hidden, hidden, hidden), axis=1)

    model = Model()
    x = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4) / 10 - 1
    targets = mx.ones((2, 4))
    runtime = tl.split.prepare(
        model, x, split_request("after:maximum", backend="mlx", trainable=True)
    )
    expected_input = mx.grad(lambda a: _loss(model(a), targets))(x)
    _value, expected_params = nn.value_and_grad(model, lambda: _loss(model(x), targets))()
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(boundary, targets)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    _assert_close(prefix["inputs"][0], expected_input)
    for name, value in tree_flatten(expected_params):
        _assert_close(suffix.parameter_grads[name], value)


def test_mlx_integer_embedding_inputs_are_not_gradient_roots() -> None:
    """Embedding parameters differentiate while integer input gradients remain absent."""

    from mlx.utils import tree_flatten

    class Model(nn.Module):
        """Learn from an integer token batch across a floating activation boundary."""

        def __init__(self) -> None:
            """Create a native embedding table and output projection."""

            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.output = nn.Linear(4, 2)

        def __call__(self, tokens: Any) -> Any:
            """Pool embedded tokens after a differentiable activation."""

            return self.output(mx.mean(mx.maximum(self.embedding(tokens), 0), axis=1))

    model = Model()
    tokens = mx.array([[1, 2, 3], [4, 1, 6]], dtype=mx.int32)
    targets = mx.ones((2, 2))
    runtime = tl.split.prepare(
        model, tokens, split_request("after:maximum", backend="mlx", trainable=True)
    )
    _value, expected = nn.value_and_grad(model, lambda: _loss(model(tokens), targets))()
    boundary = runtime.run_training_prefix(tokens)
    suffix = runtime.train_suffix_result(boundary, targets)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    assert prefix["inputs"] == (None,)
    actual = {**prefix["parameter_grads"], **suffix.parameter_grads}
    for name, value in tree_flatten(expected):
        _assert_close(actual[name], value)
