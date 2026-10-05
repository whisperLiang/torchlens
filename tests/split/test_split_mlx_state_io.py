"""Native oracles for trained MLX state export and aggregate-state synchronization."""

from __future__ import annotations

from typing import Any

import pytest
from v2_helpers import split_request

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
nn = pytest.importorskip("mlx.nn", exc_type=ImportError)

import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError  # noqa: E402

pytestmark = pytest.mark.backend_mlx


class _Model(nn.Module):
    """Reuse a list-held module around the cut, retaining frozen and unused state."""

    def __init__(self) -> None:
        """Initialize deterministic native weights for independent training oracles."""

        super().__init__()
        self.blocks = [nn.Linear(4, 4)]
        self.output = nn.Linear(4, 2)
        self.unused = nn.Linear(4, 1)
        for index, (_name, value) in enumerate(tree_flatten(self.parameters())):
            value[:] = mx.arange(value.size, dtype=value.dtype).reshape(value.shape) / 40 + (
                index / 20 - 0.1
            )
        self.output.freeze(keys="bias")

    def __call__(self, inputs: Any) -> Any:
        """Consume tied module state in both segments."""

        return self.output(self.blocks[0](mx.maximum(self.blocks[0](inputs), 0)))


def _inputs(batch: int = 3) -> Any:
    """Return finite mixed-sign inputs with nonzero model gradients."""

    return mx.arange(batch * 4, dtype=mx.float32).reshape(batch, 4) / 7 - 0.5


def _loss(output: Any, targets: Any) -> Any:
    """Compute an additive native mean squared loss."""

    return mx.mean(mx.square(output - targets))


def _assert_state(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare every named state leaf, including unconsumed and frozen arrays."""

    assert actual.keys() == expected.keys()
    mx.eval(actual, expected)
    for name, value in expected.items():
        assert actual[name].shape == value.shape
        assert actual[name].dtype == value.dtype
        assert bool(mx.allclose(actual[name], value, atol=3e-5, rtol=3e-5)), name


def test_export_trained_state_matches_native_and_is_independent() -> None:
    """Export learns from owned runtime state while snapshots and the source stay isolated."""

    model, reference = _Model(), _Model()
    source = dict(tree_flatten(model.parameters()))
    runtime = tl.split.prepare(
        model, _inputs(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    optimizer, native_optimizer = optim.Adam(0.01), optim.Adam(0.01)
    native_grad = nn.value_and_grad(reference, lambda x, y: _loss(reference(x), y))
    targets = mx.ones((3, 2))
    for _step in range(2):
        _value, gradients = native_grad(_inputs(), targets)
        boundary = runtime.run_training_prefix(_inputs())
        suffix = runtime.train_suffix_result(boundary, targets, optimizer=optimizer)
        runtime.backward_prefix(boundary, suffix.boundary_grads, optimizer=optimizer)
        native_optimizer.update(reference, gradients)
    expected = dict(tree_flatten(reference.parameters()))
    exported = runtime.state_dict()
    _assert_state(exported, expected)
    assert not bool(mx.array_equal(exported["blocks.0.weight"], source["blocks.0.weight"]))
    assert all(value is source[name] for name, value in tree_flatten(model.parameters()))
    exported["blocks.0.weight"][:] = 100
    _assert_state(runtime.state_dict(), expected)
    assert bool(mx.allclose(runtime.replay(_inputs()), reference(_inputs()), atol=3e-5))


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_load_aggregate_updates_existing_segments_and_migrates(
    materialized: bool, training: bool
) -> None:
    """Load before or after lazy binding, including source replacements and repeated rounds."""

    model, reference = _Model(), _Model()
    source = dict(tree_flatten(model.parameters()))
    runtime = tl.split.prepare(
        model,
        _inputs(2),
        split_request(
            "after:maximum",
            backend="mlx",
            trainable=training,
            placement=tl.split.PlacementPlan.on("cpu"),
        ),
    )
    if materialized:
        runtime.replay(_inputs())
    if materialized and training:
        boundary = runtime.run_training_prefix(_inputs())
        step = runtime.train_suffix_result(boundary, mx.ones((3, 2)), optimizer=optim.SGD(0.02))
        runtime.backward_prefix(boundary, step.boundary_grads, optimizer=optim.SGD(0.02))
    for shift in (0.25, -0.1):
        aggregate = {name: value + shift for name, value in tree_flatten(reference.parameters())}
        reference.update(tree_unflatten(list(aggregate.items())))
        # Native MLX replaces arrays; an already captured runtime keeps its own identities.
        model.update(tree_unflatten(list(aggregate.items())))
        runtime.load_state_dict(dict(tree_flatten(model.parameters())))
        _assert_state(runtime.state_dict(), aggregate)
        assert bool(mx.allclose(runtime.replay(_inputs()), reference(_inputs()), atol=3e-5))
        assert all(value is aggregate[name] for name, value in tree_flatten(model.parameters()))
        for updated in (
            runtime.with_placement(tl.split.PlacementPlan.on("cpu")),
            runtime.at(tl.split.before("maximum")),
        ):
            _assert_state(updated.state_dict(), aggregate)
            assert bool(mx.allclose(updated.replay(_inputs()), reference(_inputs()), atol=3e-5))
        aggregate["blocks.0.weight"][:] = 100
        assert not bool(
            mx.array_equal(runtime.state_dict()["blocks.0.weight"], aggregate["blocks.0.weight"])
        )
    assert all(value is not source[name] for name, value in tree_flatten(model.parameters()))


def test_training_after_aggregate_matches_native_across_rounds() -> None:
    """Each new round differentiates loaded weights and retains caller-owned optimizer state."""

    model, reference = _Model(), _Model()
    source = dict(tree_flatten(model.parameters()))
    runtime = tl.split.prepare(
        model, _inputs(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    optimizer, native_optimizer = optim.Adam(0.01), optim.Adam(0.01)
    native_grad = nn.value_and_grad(reference, lambda x, y: _loss(reference(x), y))
    targets = mx.ones((3, 2))
    for shift in (0.2, -0.1):
        old = runtime.run_training_prefix(_inputs())
        old_suffix = runtime.train_suffix_result(old, targets, optimizer=optimizer)
        aggregate = {name: value + shift for name, value in tree_flatten(reference.parameters())}
        runtime.load_state_dict(aggregate)
        reference.update(tree_unflatten(list(aggregate.items())))
        with pytest.raises(SplitBoundaryError, match="updated training prefix"):
            runtime.backward_prefix(old, old_suffix.boundary_grads, optimizer=optimizer)
        expected_loss, expected_gradients = native_grad(_inputs(), targets)
        fresh = runtime.run_training_prefix(_inputs())
        suffix = runtime.train_suffix_result(fresh, targets, optimizer=optimizer)
        runtime.backward_prefix(fresh, suffix.boundary_grads, optimizer=optimizer)
        assert bool(mx.allclose(suffix.loss, expected_loss, atol=3e-5))
        native_optimizer.update(reference, expected_gradients)
        _assert_state(runtime.state_dict(), dict(tree_flatten(reference.parameters())))
    assert all(value is source[name] for name, value in tree_flatten(model.parameters()))


def test_plain_callable_refuses_named_state_exchange() -> None:
    """Unnamed callable literals cannot masquerade as a native module state inventory."""

    def model(inputs: Any) -> Any:
        """Keep the callable's replay path valid independently of named state exchange."""

        return mx.multiply(mx.maximum(inputs, 0), 2)

    runtime = tl.split.prepare(model, _inputs(2), split_request("after:maximum", backend="mlx"))
    for method, args in ((runtime.state_dict, ()), (runtime.load_state_dict, ({},))):
        with pytest.raises(SplitUnsupportedError) as exc:
            method(*args)
        assert exc.value.context.reason == "named_state_unavailable"


class _Aliases(nn.Module):
    """Expose distinct native paths to one shared parameter across the cut."""

    def __init__(self) -> None:
        """Tie two module weights by array identity."""

        super().__init__()
        self.left, self.right = nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
        self.right.weight = self.left.weight

    def __call__(self, inputs: Any) -> Any:
        """Use the tied parameter in both generated segments."""

        return self.right(mx.maximum(self.left(inputs), 0))


def test_alias_names_round_trip_and_conflicts_refuse_transactionally() -> None:
    """Export retains every tied name; loading cannot silently untie conflicting values."""

    model = _Aliases()
    runtime = tl.split.prepare(
        model, _inputs(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    exported = runtime.state_dict()
    assert exported.keys() == {"left.weight", "right.weight"}
    assert exported["left.weight"] is exported["right.weight"]
    aggregate = {
        "left.weight": exported["left.weight"] + 0.1,
        "right.weight": exported["right.weight"] + 0.1,
    }
    runtime.load_state_dict(aggregate)
    _assert_state(runtime.state_dict(), aggregate)
    original = runtime.state_dict()
    with pytest.raises(SplitUnsupportedError, match="conflicting"):
        runtime.load_state_dict({**aggregate, "right.weight": aggregate["right.weight"] + 1})
    _assert_state(runtime.state_dict(), original)
    assert model.left.weight is model.right.weight
    runtime.prefix_parameters()[0][:] = 100
    with pytest.raises(SplitUnsupportedError, match="divergent"):
        runtime.state_dict()


@pytest.mark.parametrize("invalid", ["missing", "unexpected", "shape", "dtype", "type"])
def test_invalid_state_keeps_values_and_training_boundary(invalid: str) -> None:
    """Reject the whole aggregate before any owned state or boundary version changes."""

    runtime = tl.split.prepare(
        _Model(), _inputs(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    boundary = runtime.run_training_prefix(_inputs())
    suffix = runtime.train_suffix_result(boundary, mx.ones((3, 2)))
    original = runtime.state_dict()
    aggregate = {name: value + 0.2 for name, value in original.items()}
    name = "output.weight"
    if invalid == "missing":
        aggregate.pop(name)
    elif invalid == "unexpected":
        aggregate["unknown.weight"] = mx.ones((4, 4))
    elif invalid == "shape":
        aggregate[name] = mx.ones((1, 1))
    elif invalid == "dtype":
        aggregate[name] = aggregate[name].astype(mx.float16)
    else:
        aggregate[name] = 1
    with pytest.raises(SplitUnsupportedError):
        runtime.load_state_dict(aggregate)
    _assert_state(runtime.state_dict(), original)
    # Rejected aggregates must not invalidate otherwise usable connected steps.
    runtime.backward_prefix(boundary, suffix.boundary_grads)


def test_placement_failure_does_not_partially_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed suffix copy cannot leave the prefix on a different aggregate round."""

    runtime = tl.split.prepare(
        _Model(), _inputs(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    original = runtime.state_dict()
    binding = runtime.segments.suffix._binding
    adapter = binding.adapter
    replicate = adapter.replicate_state
    calls = 0
    prefix_count = len(runtime.segments.prefix._state.entries())

    def fail_on_suffix(value: Any, device: Any, *, trainable: bool) -> Any:
        """Fail after prefix staging to exercise publication atomicity."""

        nonlocal calls
        calls += 1
        if calls > prefix_count:
            raise RuntimeError("simulated suffix placement failure")
        return replicate(value, device, trainable=trainable)

    monkeypatch.setattr(adapter, "replicate_state", fail_on_suffix)
    with pytest.raises(RuntimeError, match="suffix placement"):
        runtime.load_state_dict({name: value + 0.2 for name, value in original.items()})
    _assert_state(runtime.state_dict(), original)


def test_batchnorm_buffers_export_and_buffer_load_invalidates_old_boundary() -> None:
    """Running state comes from the runtime; a buffer-only change invalidates old pullbacks."""

    class Model(nn.Module):
        """Exercise frozen parameters and mutable buffers with a train-mode prefix."""

        def __init__(self) -> None:
            """Create a native normalization module with only frozen affine parameters."""

            super().__init__()
            self.norm = nn.BatchNorm(4)
            self.norm.freeze()

        def __call__(self, inputs: Any) -> Any:
            """Apply a nontrivial suffix after updating native running statistics."""

            hidden = mx.maximum(self.norm(inputs), 0)
            return mx.multiply(hidden, hidden)

    model, reference = Model(), Model()
    runtime = tl.split.prepare(
        model, _inputs(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    boundary = runtime.run_training_prefix(_inputs())
    suffix = runtime.train_suffix_result(boundary, mx.ones((3, 4)))
    reference(_inputs())
    expected = dict(tree_flatten(reference.parameters()))
    _assert_state(runtime.state_dict(), expected)
    assert not bool(mx.array_equal(model.norm.running_mean, expected["norm.running_mean"]))
    aggregate = {**expected, "norm.running_mean": expected["norm.running_mean"] + 0.2}
    runtime.load_state_dict(aggregate)
    _assert_state(runtime.state_dict(), aggregate)
    with pytest.raises(SplitBoundaryError, match="updated training prefix"):
        runtime.backward_prefix(boundary, suffix.boundary_grads)
    fresh = runtime.run_training_prefix(_inputs())
    fresh_suffix = runtime.train_suffix_result(fresh, mx.ones((3, 4)))
    runtime.backward_prefix(fresh, fresh_suffix.boundary_grads)
