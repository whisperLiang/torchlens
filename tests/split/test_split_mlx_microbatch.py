"""Native MLX microbatch oracles for logical gradients and optimizer steps."""

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


class _Model(nn.Module):
    """Share parameters around a transposed batch boundary."""

    def __init__(self) -> None:
        """Create one native linear projection."""

        super().__init__()
        self.shared = nn.Linear(4, 4)

    def __call__(self, x: Any) -> Any:
        """Transpose the batch axis twice around the cut."""

        hidden = mx.transpose(mx.maximum(self.shared(x), 0))
        return self.shared(mx.transpose(mx.multiply(hidden, 2)))


def _close(left: Any, right: Any) -> None:
    """Compare native values and gradients at float32 tolerances."""

    mx.eval(left, right)
    assert bool(mx.allclose(left, right, atol=3e-5, rtol=3e-5))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("size", [1, 2, 8])
def test_mlx_microbatches_match_logical_gradients(reduction: str, size: int) -> None:
    """Uneven chunks sum tied gradients and advance Adam once across both segments."""

    mx.random.seed(8)
    model = _Model()
    reference = deepcopy(model)
    x = mx.arange(20, dtype=mx.float32).reshape(5, 4) / 7 - 0.8
    targets = {"answer": mx.cos(x)}
    runtime = tl.split.prepare(
        model,
        x,
        split_request(
            "after:transpose_1",
            backend="mlx",
            trainable=True,
            placement=tl.split.PlacementPlan.on(mx.new_stream(mx.cpu)),
        ),
    )

    def loss(output: Any, target: Any) -> Any:
        """Apply the requested additive reduction over nested targets."""

        squared = mx.square(output - target["answer"])
        return mx.mean(squared) if reduction == "mean" else mx.sum(squared)

    def native_loss(values: Any) -> Any:
        """Execute the native full batch with independent parameter/input roots."""

        reference.update(values["parameters"])
        return loss(reference(values["inputs"]), targets)

    expected_value, expected = mx.value_and_grad(native_loss)(
        {"inputs": x, "parameters": reference.trainable_parameters()}
    )
    optimizer = optim.Adam(0.01)
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(
        boundary,
        targets,
        loss_fn=loss,
        optimizer=optimizer,
        microbatch_size=size,
        microbatch_reduction=reduction,
    )
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads, optimizer=optimizer)
    _close(suffix.loss, expected_value)
    _close(prefix["inputs"][0], expected["inputs"])
    for name, value in tree_flatten(expected["parameters"]):
        _close(prefix["all_parameter_grads"][name], value)
    optim.Adam(0.01).update(reference, expected["parameters"])
    _close(runtime.replay(x), reference(x))
    assert int(optimizer.step.item()) == 1
    assert prefix["optimizer_step_count"] == 1


def test_mlx_cached_microbatch_updates_suffix_once() -> None:
    """Detached prefix features support one immediate suffix optimizer commit."""

    model = _Model()
    x = mx.arange(20, dtype=mx.float32).reshape(5, 4) / 10
    runtime = tl.split.prepare(
        model, x, split_request("after:maximum", backend="mlx", trainable=True)
    )
    optimizer = optim.SGD(0.01)
    seen: list[tuple[int, int, int]] = []

    def slicer(target: Any, start: int, end: int, batch: int) -> Any:
        """Record and slice custom native targets by the declared logical batch."""

        seen.append((start, end, batch))
        return target[start:end]

    boundary = runtime.run_prefix(x)
    result = runtime.train_suffix_result(
        boundary, mx.ones((5, 4)), optimizer=optimizer, microbatch_size=2, target_slicer=slicer
    )
    assert seen == [(0, 2, 5), (2, 4, 5), (4, 5, 5)]
    assert result.optimizer_applied and not result.optimizer_pending
    assert int(optimizer.step.item()) == 1


def test_mlx_stateful_microbatches_match_native_sequential_chunks() -> None:
    """BatchNorm and Dropout buffers, masks and tied gradients follow native chunk execution."""

    class Model(nn.Module):
        """Share a projection around a stateful suffix."""

        def __init__(self) -> None:
            """Create a reused projection and native running/random state."""

            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.norm = nn.BatchNorm(4, momentum=0.2)
            self.dropout = nn.Dropout(0.3)

        def __call__(self, value: Any) -> Any:
            """Evaluate the prefix once and the stateful suffix for each chunk."""

            return self.shared(self.dropout(self.norm(mx.maximum(self.shared(value), 0))))

    mx.random.seed(7)
    model = Model()
    reference = deepcopy(model)
    x = mx.sin(mx.arange(20, dtype=mx.float32).reshape(5, 4) / 3)
    targets = mx.cos(x)
    runtime = tl.split.prepare(
        model, x, split_request("after:maximum", backend="mlx", trainable=True)
    )

    def native_loss(values: Any) -> Any:
        """Compute one weighted logical objective over sequential native chunks."""

        reference.update(values["parameters"])
        hidden = mx.maximum(reference.shared(values["input"]), 0)
        total = mx.array(0.0)
        for start in range(0, 5, 3):
            end = min(start + 3, 5)
            output = reference.shared(reference.dropout(reference.norm(hidden[start:end])))
            total = total + mx.mean(mx.square(output - targets[start:end])) * ((end - start) / 5)
        return total

    mx.random.seed(91)
    value, gradients = mx.value_and_grad(native_loss)(
        {"input": x, "parameters": reference.trainable_parameters()}
    )
    mx.eval(value, gradients, reference.parameters())
    saved_rng = [item.astype(item.dtype) for item in mx.random.state]
    mx.eval(saved_rng)
    mx.random.seed(91)
    boundary = runtime.run_training_prefix(x)
    optimizer = optim.Adam(0.01)
    suffix = runtime.train_suffix_result(boundary, targets, optimizer=optimizer, microbatch_size=3)
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads, optimizer=optimizer)
    _close(suffix.loss, value)
    _close(prefix["inputs"][0], gradients["input"])
    for name, value in tree_flatten(gradients["parameters"]):
        _close(prefix["all_parameter_grads"][name], value)
    optim.Adam(0.01).update(reference, gradients["parameters"])
    binding = runtime.segments.suffix._binding
    for entry in binding.state.entries():
        name = binding._parameters.get(entry.source_id, ("", False))[0]
        if name:
            _close(entry.value, dict(tree_flatten(reference.parameters()))[name])
    assert all(bool(mx.array_equal(left, right)) for left, right in zip(mx.random.state, saved_rng))
    assert int(optimizer.step.item()) == 1
