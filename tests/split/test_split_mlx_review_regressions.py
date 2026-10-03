"""MLX output occurrences and native optimizer-tree regression oracles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from v2_helpers import split_request

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
nn = pytest.importorskip("mlx.nn", exc_type=ImportError)

import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx.containers import iter_arrays_with_paths  # noqa: E402

pytestmark = pytest.mark.backend_mlx


def _input(batch: int) -> Any:
    """Supply mixed-sign inputs at independently varied batch sizes."""

    return mx.arange(batch * 4, dtype=mx.float32).reshape(batch, 4) / 5 - 1


def _close(actual: Any, expected: Any) -> None:
    """Compare native float32 arrays including their full shapes."""

    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert bool(mx.allclose(actual, expected, atol=2e-5, rtol=2e-5))


@pytest.mark.parametrize("container", ["dict", "tuple", "nested"])
def test_mlx_repeated_outputs_replay_and_cache_at_every_cut(container: str, tmp_path: Path) -> None:
    """Every final occurrence survives changed batches, cuts and cached boundaries."""

    def model(x: Any) -> Any:
        """Repeat an intermediate, a native split leaf, and the original input."""

        hidden = mx.maximum(x, 0)
        left, right = mx.split(hidden, 2, axis=-1)
        total = mx.add(left, right)
        if container == "dict":
            return {"a": total, "b": total}
        if container == "tuple":
            return (total, total, "mlx")
        return {"pair": [right, total, right], "early": (hidden, hidden), "input": [x, x]}

    seed = tl.split.prepare(
        model,
        _input(2),
        split_request("before:split_1_3", backend="mlx", boundary_cache=True),
    )
    assert seed.batch_validation["status"] == "passed"
    for index, candidate in enumerate(seed.split_points().supported):
        runtime = seed.at(candidate.point)
        for batch in (1, 2, 5):
            x = _input(batch)
            expected = iter_arrays_with_paths(model(x), runtime.adapter.is_tensor)
            boundary = runtime.run_prefix(x)
            cache_path = tmp_path / f"{index}-{batch}.boundary"
            runtime.save_boundary(boundary, cache_path)
            actual = iter_arrays_with_paths(
                runtime.run_suffix(runtime.load_boundary(cache_path)), runtime.adapter.is_tensor
            )
            assert [path for _value, path in actual] == [path for _value, path in expected]
            for (value, _path), (reference, _expected_path) in zip(actual, expected, strict=True):
                _close(value, reference)
            for i, (reference, _path) in enumerate(expected):
                for j, (other, _other_path) in enumerate(expected):
                    if reference is other:
                        assert actual[i][0] is actual[j][0]


def test_mlx_repeated_output_gradients_accumulate() -> None:
    """Distinct loss uses of the same final array sum into one native pullback."""

    def model(x: Any) -> Any:
        """Return a native split leaf twice through different container paths."""

        _left, right = mx.split(mx.maximum(x, 0), 2, axis=-1)
        return {"a": right, "nested": [right]}

    def loss(output: Any, _targets: Any) -> Any:
        """Give the two occurrences different contributions to the scalar loss."""

        return mx.mean(mx.square(output["a"])) + 3 * mx.mean(output["nested"][0])

    runtime = tl.split.prepare(
        model, _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    x = _input(5)
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(boundary, None, loss_fn=loss)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads)
    _close(suffix.loss, loss(model(x), None))
    _close(prefix["inputs"][0], mx.grad(lambda value: loss(model(value), None))(x))


def test_mlx_output_occurrence_metadata_survives_portable_save(tmp_path: Path) -> None:
    """The existing root ModuleCall schema archives all paths, including duplicates."""

    def model(x: Any) -> Any:
        """Return a repeated array through nested builtin containers."""

        value = mx.maximum(x, 0)
        return {"first": value, "nested": [value, {"last": value}]}

    trace = tl.trace(model, _input(2), backend="mlx")
    root = trace.modules["self"].ops[0]
    assert root.output_paths == (("first",), ("nested", 0), ("nested", 1, "last"))
    assert len(root.output_ops) == 3 and len(set(root.output_ops)) == 1
    path = tmp_path / "output-occurrences.tlspec"
    trace.save(path)
    loaded = tl.load(path).modules["self"].ops[0]
    assert loaded.output_paths == root.output_paths
    assert loaded.output_ops == root.output_ops
    assert loaded.output_structure == root.output_structure


class _GroupedModel(nn.Module):
    """A list of modules with a parameterized layer reused across the split."""

    def __init__(self) -> None:
        """Use fixed nested parameters for independent training comparisons."""

        super().__init__()
        self.first = nn.Linear(4, 4)
        self.blocks = [nn.Linear(4, 4)]
        self.output = nn.Linear(4, 2)
        for index, (_name, value) in enumerate(tree_flatten(self.parameters())):
            value[:] = mx.arange(value.size, dtype=value.dtype).reshape(value.shape) / 30 + (
                index / 20 - 0.1
            )

    def __call__(self, x: Any) -> Any:
        """Reuse the second block in the prefix and suffix."""

        hidden = self.blocks[0](self.first(x))
        return self.output(self.blocks[0](mx.maximum(hidden, 0)))


def _optimizer() -> Any:
    """Assign biases to SGD and weights to Adam using native full-path filters."""

    return optim.MultiOptimizer(
        [optim.SGD(learning_rate=0.03, momentum=0.9), optim.Adam(learning_rate=0.01)],
        [lambda name, _value: name.endswith(".bias")],
    )


def _loss(output: Any, targets: Any) -> Any:
    """Use the same scalar loss on native and split training paths."""

    return mx.mean(mx.square(output - targets))


@pytest.mark.parametrize("shared_optimizer", [False, True])
@pytest.mark.parametrize("placement", [None, ("cpu", "gpu"), ("gpu", "cpu")])
def test_mlx_multi_optimizer_matches_nested_tied_native_training(
    shared_optimizer: bool, placement: tuple[str, str] | None
) -> None:
    """Native grouped optimizers retain nested paths and update tied weights once."""

    if placement is not None and not any(
        callable(available := getattr(getattr(mx, name, None), "is_available", None))
        and available()
        for name in ("metal", "cuda")
    ):
        pytest.skip("Native MLX GPU backend is unavailable")
    model, reference = _GroupedModel(), _GroupedModel()
    original = dict(tree_flatten(model.parameters()))
    snapshots = {name: value.astype(value.dtype) for name, value in original.items()}
    mx.eval(snapshots)
    runtime = tl.split.prepare(
        model,
        _input(2),
        split_request(
            "after:maximum",
            backend="mlx",
            trainable=True,
            placement=tl.split.PlacementPlan.across(*placement) if placement else None,
        ),
    )
    prefix_optimizer, native_optimizer = _optimizer(), _optimizer()
    suffix_optimizer = prefix_optimizer if shared_optimizer else _optimizer()
    native_grad = nn.value_and_grad(reference, lambda x, targets: _loss(reference(x), targets))
    x, targets = _input(3), mx.cos(_input(3)[:, :2])
    for _step in range(3):
        expected_input = mx.grad(lambda value: _loss(reference(value), targets))(x)
        expected_loss, expected_grads = native_grad(x, targets)
        boundary = runtime.run_training_prefix(x)
        suffix = runtime.train_suffix_result(boundary, targets, optimizer=suffix_optimizer)
        prefix = runtime.backward_prefix(
            boundary, suffix.boundary_grads, optimizer=prefix_optimizer
        )
        _close(suffix.loss, expected_loss)
        _close(prefix["inputs"][0], expected_input)
        for name, gradient in tree_flatten(expected_grads):
            _close(prefix["all_parameter_grads"][name], gradient)
        assert prefix["optimizer_step_count"] == (1 if shared_optimizer else 2)
        native_optimizer.update(reference, expected_grads)
        mx.eval(reference.parameters())
        actual = {
            name: value
            for segment in (runtime.segments.prefix, runtime.segments.suffix)
            for name, value in segment._binding.parameters().items()
        }
        for name, value in tree_flatten(reference.parameters()):
            _close(actual[name], value)
        _close(runtime.replay(x), reference(x))
    for name, value in tree_flatten(model.parameters()):
        assert value is original[name]
        _close(value, snapshots[name])


def test_mlx_multi_optimizer_detached_suffix_matches_native_updates() -> None:
    """Suffix-only steps restore nested list paths before native optimizer filtering."""

    runtime = tl.split.prepare(
        _GroupedModel(), _input(2), split_request("after:maximum", backend="mlx", trainable=True)
    )
    optimizer, reference_optimizer = _optimizer(), _optimizer()
    reference_parameters = dict(runtime.segments.suffix._binding.parameters())
    x, targets = _input(3), mx.ones((3, 2))
    for _step in range(3):
        boundary = runtime.run_prefix(x)
        result = runtime.train_suffix_result(boundary, targets, optimizer=optimizer)
        assert result.optimizer_applied and not result.optimizer_pending
        reference_tree = reference_optimizer.apply_gradients(
            tree_unflatten(list(result.parameter_grads.items())),
            tree_unflatten(list(reference_parameters.items())),
        )
        reference_parameters = dict(tree_flatten(reference_tree))
        for name, value in runtime.segments.suffix._binding.parameters().items():
            _close(value, reference_parameters[name])
