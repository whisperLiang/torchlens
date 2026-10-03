"""Native oracles for MLX tied parameters, running buffers, and random-mask replay."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from v2_helpers import split_request

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
nn = pytest.importorskip("mlx.nn", exc_type=ImportError)

import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

import torchlens as tl  # noqa: E402

pytestmark = pytest.mark.backend_mlx


def _input(batch: int = 5) -> Any:
    """Construct nonuniform samples with nonlinear feature correlations."""

    value = mx.arange(batch * 4, dtype=mx.float32).reshape(batch, 4)
    return mx.sin(value / 3) + value / 20


def _loss(output: Any, targets: Any) -> Any:
    """Use an additive mean squared native loss."""

    return mx.mean(mx.square(output - targets))


def _close(left: Any, right: Any) -> None:
    """Compare values, including gradients and optimizer state, at float32 tolerance."""

    mx.eval(left, right)
    assert left.shape == right.shape
    if mx.issubdtype(left.dtype, mx.integer):
        assert bool(mx.array_equal(left, right))
        return
    assert bool(mx.allclose(left, right, atol=3e-5, rtol=3e-5))


def _native_gradients(model: Any, x: Any, targets: Any) -> tuple[Any, Any]:
    """Differentiate inputs and parameters together, executing stateful forward once."""

    def loss(values: Any) -> Any:
        """Bind native parameter roots and evaluate the full unsplit program."""

        model.update(values["parameters"])
        return _loss(model(values["inputs"]), targets)

    result = mx.value_and_grad(loss)({"inputs": x, "parameters": model.trainable_parameters()})
    mx.eval(result, model.parameters())
    return result


def _runtime_values(runtime: Any) -> dict[str, Any]:
    """Inspect every consumed logical parameter and running buffer."""

    result = {}
    for segment in (runtime.segments.prefix, runtime.segments.suffix):
        binding = segment._binding
        binding.bound_values()
        for entry in binding.state.entries():
            name = binding._parameters.get(entry.source_id, ("", False))[0]
            if name:
                result[name] = entry.value
    return result


class _TiedModel(nn.Module):
    """Reuse one linear module with exclusive modules on either side of the cut."""

    def __init__(self) -> None:
        """Create shared and exclusive parameter families."""

        super().__init__()
        self.first = nn.Linear(4, 4)
        self.shared = nn.Linear(4, 4)
        self.last = nn.Linear(4, 4)

    def __call__(self, x: Any) -> Any:
        """Consume the shared weights twice around a differentiable activation."""

        return self.last(self.shared(mx.maximum(self.shared(self.first(x)), 0)))


class _StatefulModel(nn.Module):
    """Share train-mode BatchNorm and linear state across two independent dropouts."""

    def __init__(self) -> None:
        """Create native running statistics and stochastic layers."""

        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.norm = nn.BatchNorm(4, momentum=0.2)
        self.left_dropout = nn.Dropout(0.25)
        self.right_dropout = nn.Dropout(0.3)

    def __call__(self, x: Any) -> Any:
        """Run each shared state consumer once on its side of the cut."""

        hidden = self.left_dropout(self.norm(self.shared(x)))
        return self.shared(self.norm(self.right_dropout(mx.maximum(hidden, 0))))


@pytest.mark.parametrize("optimizer_cls", [optim.SGD, optim.Adam, optim.AdamW])
@pytest.mark.parametrize("shared_optimizer", [False, True])
def test_mlx_tied_parameter_steps_match_native(optimizer_cls: Any, shared_optimizer: bool) -> None:
    """Momentum, Adam bias correction and weight decay act once on summed tied gradients."""

    mx.random.seed(73)
    model = _TiedModel()
    reference = deepcopy(model)
    source = dict(tree_flatten(model.parameters()))
    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    kwargs = {"momentum": 0.8} if optimizer_cls is optim.SGD else {}
    prefix_opt = optimizer_cls(learning_rate=0.01, **kwargs)
    suffix_opt = prefix_opt if shared_optimizer else optimizer_cls(learning_rate=0.01, **kwargs)
    reference_opt = optimizer_cls(learning_rate=0.01, **kwargs)
    targets = mx.cos(_input())
    for _step in range(3):
        value, gradients = _native_gradients(reference, _input(), targets)
        boundary = runtime.run_training_prefix(_input())
        suffix = runtime.train_suffix_result(boundary, targets, optimizer=suffix_opt)
        assert suffix.optimizer_pending and not suffix.optimizer_applied
        prefix = runtime.backward_prefix(boundary, suffix.boundary_grads, optimizer=prefix_opt)
        _close(suffix.loss, value)
        _close(prefix["inputs"][0], gradients["inputs"])
        expected = dict(tree_flatten(gradients["parameters"]))
        assert prefix["all_parameter_grads"].keys() == expected.keys()
        for name, grad in expected.items():
            _close(prefix["all_parameter_grads"][name], grad)
        assert set(prefix["shared_parameter_grads"]) == {"shared.weight", "shared.bias"}
        assert prefix["optimizer_step_count"] == (1 if shared_optimizer else 2)
        reference_opt.update(reference, gradients["parameters"])
        for name, value in tree_flatten(reference.parameters()):
            _close(_runtime_values(runtime)[name], value)
        _close(runtime.replay(_input()), reference(_input()))
    assert int(prefix_opt.step.item()) == int(suffix_opt.step.item()) == 3
    assert all(value is source[name] for name, value in tree_flatten(model.parameters()))
    migrated = runtime.with_placement(tl.split.PlacementPlan.on(mx.new_stream(mx.cpu)))
    recut = migrated.at(tl.split.before("maximum"))
    _close(recut.replay(_input()), reference(_input()))
    assert all(candidate.training_supported for candidate in runtime.split_points().supported)


@pytest.mark.parametrize("boundary_point", ["before:maximum", "after:maximum"])
def test_mlx_batchnorm_dropout_and_ties_match_native(boundary_point: str) -> None:
    """Running statistics advance once per native call and VJPs reuse the original masks."""

    mx.random.seed(41)
    model = _StatefulModel()
    reference = deepcopy(model)
    source = dict(tree_flatten(model.parameters()))
    initial_rng = tuple(value.astype(value.dtype) for value in mx.random.state)
    mx.eval(initial_rng)
    runtime = tl.split.prepare(
        model, _input(2), split_request(boundary_point, backend="mlx", trainable=True)
    ).with_placement(
        tl.split.PlacementPlan(
            prefix=tl.split.DevicePlacement(mx.new_stream(mx.cpu)),
            suffix=tl.split.DevicePlacement(mx.new_stream(mx.cpu)),
        )
    )
    assert runtime.batch_validation["status"] == "passed"
    assert runtime.validate_equivalence(model, (_input(),))
    for left, right in zip(mx.random.state, initial_rng, strict=True):
        _close(left, right)
    optimizer = optim.Adam(learning_rate=0.01)
    native_optimizer = optim.Adam(learning_rate=0.01)
    targets = mx.cos(_input())
    for step in range(3):
        mx.random.seed(200 + step)
        expected_value, expected = _native_gradients(reference, _input(), targets)
        expected_rng = tuple(value.astype(value.dtype) for value in mx.random.state)
        mx.eval(expected_rng)
        mx.random.seed(200 + step)
        boundary = runtime.run_training_prefix(_input())
        suffix = runtime.train_suffix_result(boundary, targets, optimizer=optimizer)
        prefix = runtime.backward_prefix(boundary, suffix.boundary_grads, optimizer=optimizer)
        _close(suffix.loss, expected_value)
        _close(prefix["inputs"][0], expected["inputs"])
        for name, value in tree_flatten(expected["parameters"]):
            _close(prefix["all_parameter_grads"][name], value)
        native_optimizer.update(reference, expected["parameters"])
        for name, value in tree_flatten(reference.parameters()):
            _close(_runtime_values(runtime)[name], value)
        for left, right in zip(mx.random.state, expected_rng, strict=True):
            _close(left, right)
    assert all(value is source[name] for name, value in tree_flatten(model.parameters()))
    _close(model.norm.running_mean, mx.zeros((4,)))
    _close(model.norm.running_var, mx.ones((4,)))
    migrated = runtime.with_placement(tl.split.PlacementPlan.on(mx.new_stream(mx.cpu)))
    recut = migrated.at(tl.split.before("maximum"))
    for name, value in tree_flatten(reference.parameters()):
        _close(_runtime_values(recut)[name], value)


def test_mlx_explicit_random_producers_replay_fresh_then_reuse_vjp_keys() -> None:
    """Random core producers participate in the graph and preserve forward/VJP RNG semantics."""

    def model(x: Any) -> Any:
        """Use two independent random draws around the activation cut."""

        noisy = mx.multiply(x, mx.random.uniform(shape=x.shape))
        return mx.multiply(mx.maximum(noisy, 0), mx.random.normal(shape=x.shape))

    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    assert runtime.batch_validation["status"] == "passed"
    x, targets = _input(), mx.ones((5, 4))
    mx.random.seed(87)
    expected = mx.value_and_grad(lambda value: _loss(model(value), targets))(x)
    mx.eval(expected)
    expected_rng = tuple(value.astype(value.dtype) for value in mx.random.state)
    mx.eval(expected_rng)
    mx.random.seed(87)
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(boundary, targets)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    _close(suffix.loss, expected[0])
    _close(prefix["inputs"][0], expected[1])
    for left, right in zip(mx.random.state, expected_rng, strict=True):
        _close(left, right)


def test_mlx_stateful_gradients_at_every_valid_cut() -> None:
    """Cuts through either normalization or random region preserve native training semantics."""

    mx.random.seed(52)
    model = _StatefulModel()
    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    targets = mx.cos(_input())
    for candidate in runtime.split_points().supported:
        reference = deepcopy(model)
        split = runtime.at(candidate.point)
        mx.random.seed(82)
        expected_value, gradients = _native_gradients(reference, _input(), targets)
        mx.random.seed(82)
        boundary = split.run_training_prefix(_input())
        suffix = split.train_suffix_result(boundary, targets)
        prefix = split.backward_prefix(boundary, suffix.boundary_grads)
        _close(suffix.loss, expected_value)
        _close(prefix["inputs"][0], gradients["inputs"])
        for name, value in tree_flatten(gradients["parameters"]):
            _close(prefix["all_parameter_grads"][name], value)
        for name, value in tree_flatten(reference.parameters()):
            _close(_runtime_values(split)[name], value)


def test_mlx_tied_aliases_in_distinct_modules_and_suffix_optimizer_owner() -> None:
    """Distinct module addresses sharing an array retain one gradient and one Adam update."""

    class Model(nn.Module):
        """Expose a tied matrix through two independently addressed native modules."""

        def __init__(self) -> None:
            """Bind both modules to the same matrix, without duplicate biases."""

            super().__init__()
            self.left = nn.Linear(4, 4, bias=False)
            self.right = nn.Linear(4, 4, bias=False)
            self.right.weight = self.left.weight

        def __call__(self, x: Any) -> Any:
            """Use each module once across the activation cut."""

            return self.right(mx.maximum(self.left(x), 0))

    mx.random.seed(25)
    model = Model()
    source = model.left.weight
    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    targets = mx.cos(_input())

    def native_loss(values: Any) -> Any:
        """Use a unique weight root in the native unsplit functional oracle."""

        hidden = mx.maximum(values["input"] @ mx.transpose(values["weight"]), 0)
        return _loss(hidden @ mx.transpose(values["weight"]), targets)

    value, gradients = mx.value_and_grad(native_loss)({"input": _input(), "weight": source})
    boundary = runtime.run_training_prefix(_input())
    optimizer = optim.Adam(0.01)
    suffix = runtime.train_suffix_result(boundary, targets, optimizer=optimizer)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    assert len(prefix["all_parameter_grads"]) == 1
    _close(suffix.loss, value)
    _close(prefix["inputs"][0], gradients["input"])
    _close(next(iter(prefix["all_parameter_grads"].values())), gradients["weight"])
    expected = optim.Adam(0.01).apply_gradients({"weight": gradients["weight"]}, {"weight": source})
    _close(runtime.prefix_parameters()[0], expected["weight"])
    _close(runtime.suffix_parameters()[0], expected["weight"])
    assert int(optimizer.step.item()) == 1
    assert model.left.weight is model.right.weight is source


def test_mlx_readonly_random_state_container_restores_native_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native PRNG replay works with the read-only state container introduced by MLX 0.32."""

    from torchlens.backends.mlx._call_state import mlx_rng_scope, snapshot_mlx_rng

    native_state = mx.random.state

    class ReadOnlyState:
        """Expose current native keys by iteration without a container setter."""

        def __iter__(self) -> Any:
            """Iterate the current thread's generator keys."""

            return iter(native_state)

    monkeypatch.setattr(mx.random, "state", ReadOnlyState())
    mx.random.seed(98)
    before = snapshot_mlx_rng()
    first = mx.random.uniform(shape=(8,))
    mx.eval(first)
    after = snapshot_mlx_rng()
    with mlx_rng_scope(before):
        replayed = mx.random.uniform(shape=(8,))
        mx.eval(replayed)
        _close(replayed, first)
    for expected, actual in zip(after, snapshot_mlx_rng(), strict=True):
        assert bool(mx.array_equal(expected, actual))


def test_mlx_batchnorm_minimum_batch_uses_independent_probe() -> None:
    """A native minimum batch of two retains sampled validation at a different batch."""

    class Model(_StatefulModel):
        """Apply the minimum batch required by current native BatchNorm versions."""

        def __call__(self, x: Any) -> Any:
            """Reject singleton batches before invoking the stateful native program."""

            if x.shape[0] < 2:
                raise ValueError("BatchNorm training requires more than one value per channel")
            return super().__call__(x)

    mx.random.seed(22)
    model = Model()
    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    assert runtime.traced_batch_size == 2
    assert runtime.batch_validation["probe_batch_size"] == 3
    assert runtime.batch_validation["status"] == "passed"
    assert runtime.validate_equivalence(model, (_input(),))
    _close(model.norm.running_mean, mx.zeros((4,)))
    _close(model.norm.running_var, mx.ones((4,)))


@pytest.mark.parametrize("prefix_device,suffix_device", [("cpu", "gpu"), ("gpu", "cpu")])
def test_mlx_stateful_tied_updates_across_native_devices(
    prefix_device: str, suffix_device: str
) -> None:
    """Native CPU/GPU placements share state and sum tied gradients on the optimizer owner."""

    try:
        mx.default_stream(mx.gpu)
    except ValueError:
        pytest.skip("Installed MLX runtime has no GPU backend")
    mx.random.seed(43)
    model = _StatefulModel()
    reference = deepcopy(model)
    runtime = tl.split.prepare(
        model,
        _input(2),
        split_request(
            "after:maximum",
            backend="mlx",
            trainable=True,
            placement=tl.split.PlacementPlan.across(prefix_device, suffix_device),
        ),
    )
    targets = mx.cos(_input())
    mx.random.seed(63)
    expected_value, gradients = _native_gradients(reference, _input(), targets)
    mx.random.seed(63)
    optimizer = optim.Adam(0.01)
    boundary = runtime.run_training_prefix(_input())
    suffix = runtime.train_suffix_result(boundary, targets, optimizer=optimizer)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads, optimizer=optimizer)
    _close(suffix.loss, expected_value)
    _close(prefix["inputs"][0], gradients["inputs"])
    for name, value in tree_flatten(gradients["parameters"]):
        _close(prefix["all_parameter_grads"][name], value)
    optim.Adam(0.01).update(reference, gradients["parameters"])
    for name, value in tree_flatten(reference.parameters()):
        _close(_runtime_values(runtime)[name], value)
    assert int(optimizer.step.item()) == 1
