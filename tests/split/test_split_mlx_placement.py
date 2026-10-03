"""MLX scoped stream placement and conditional native CPU/GPU integration."""

from __future__ import annotations

from functools import wraps
from typing import Any

import pytest
from v2_helpers import split_request

mx = pytest.importorskip("mlx.core", exc_type=ImportError)

import torchlens as tl  # noqa: E402
from torchlens.split.errors import SplitUnsupportedError  # noqa: E402

pytestmark = pytest.mark.backend_mlx


def _gpu_available() -> bool:
    """Return whether this MLX build exposes an available GPU backend."""

    return any(
        callable(available := getattr(getattr(mx, name, None), "is_available", None))
        and available()
        for name in ("metal", "cuda")
    )


def _model(x: Any) -> Any:
    """Return a differentiable graph with functional and core operations."""

    import mlx.nn as nn

    return mx.multiply(nn.relu(mx.maximum(x, 0)), 2)


@pytest.mark.parametrize("device", ["cpu", "cpu:0", mx.cpu, mx.Device(mx.cpu)])
def test_mlx_accepts_native_cpu_devices_and_preserves_ambient_stream(device: Any) -> None:
    """CPU placements execute under a scope and restore the caller's stream afterward."""

    x = mx.arange(8, dtype=mx.float32).reshape(2, 4) - 2
    ambient = mx.default_stream(mx.cpu)
    runtime = tl.split.prepare(
        _model,
        x,
        split_request("after:maximum", backend="mlx", placement=tl.split.PlacementPlan.on(device)),
    )
    output, expected = runtime.replay(x), _model(x)
    mx.eval(output, expected)
    assert bool(mx.allclose(output, expected))
    assert mx.default_stream(mx.cpu) == ambient
    assert runtime.run_prefix(x).to(device).metadata["supports_prefix_backward"] is False


def test_mlx_distinct_segment_streams_override_captured_streams() -> None:
    """Explicit core arguments and native calls both obey their segment's stream."""

    import mlx.nn as nn

    captured_stream = mx.default_stream(mx.cpu)

    class Model(nn.Module):
        """Use native modules around an explicitly streamed core operation."""

        def __init__(self) -> None:
            """Create the segment-local parameterized operations."""

            super().__init__()
            self.first = nn.Linear(4, 4)
            self.last = nn.Linear(4, 2)

        def __call__(self, x: Any) -> Any:
            """Evaluate the middle operation on a stream recorded during capture."""

            return self.last(mx.maximum(self.first(x), 0, stream=captured_stream))

    model = Model()
    x = mx.ones((2, 4))
    seed = tl.split.prepare(model, x, split_request("after:maximum", backend="mlx"))
    prefix_stream, suffix_stream = mx.new_stream(mx.cpu), mx.new_stream(mx.cpu)
    ambient = mx.default_stream(mx.cpu)
    observed: list[tuple[str, Any, Any]] = []
    for node in seed.trace_graph.nodes:
        capture = node.target
        if capture is None or not callable(getattr(capture, "func", None)):
            continue
        original = capture.func
        if getattr(original, "_placement_observer", False):
            continue

        @wraps(original)
        def observe(
            *args: Any, _func: Any = original, _id: str = node.canonical_id, **kwargs: Any
        ) -> Any:
            """Record the active native stream and any explicit core stream argument."""

            observed.append((_id, mx.default_stream(mx.cpu), kwargs.get("stream")))
            return _func(*args, **kwargs)

        observe._placement_observer = True
        object.__setattr__(capture, "func", observe)
    runtime = seed.with_placement(tl.split.PlacementPlan.across(prefix_stream, suffix_stream))
    output, expected = runtime.replay(x), model(x)
    mx.eval(output, expected)
    assert bool(mx.allclose(output, expected))
    assert observed
    for node_id, active, explicit in observed:
        chosen = prefix_stream if node_id in runtime.plan.prefix_node_ids else suffix_stream
        assert active == chosen
        if explicit is not None:
            assert explicit == chosen
    assert mx.default_stream(mx.cpu) == ambient


@pytest.mark.parametrize("device", ["xpu", "cpu:1", mx.Device(mx.cpu, 1)])
def test_mlx_invalid_devices_refuse_before_execution(device: Any) -> None:
    """Unknown devices and ignored CPU indexes never become successful placements."""

    x = mx.ones((2, 4))
    runtime = tl.split.prepare(_model, x, split_request("after:maximum", backend="mlx"))
    with pytest.raises(SplitUnsupportedError, match="Cannot use MLX split device"):
        runtime.with_placement(tl.split.PlacementPlan.on(device))


@pytest.mark.parametrize("name", ["gpu", "metal", "cuda"])
def test_mlx_unavailable_gpu_backends_refuse_explicit_placement(name: str) -> None:
    """Backend-specific GPU strings cannot silently fall back to another backend."""

    native = getattr(getattr(mx, name, None), "is_available", None)
    available = _gpu_available() if name == "gpu" else callable(native) and native()
    if available:
        pytest.skip(f"MLX {name} backend is available on this host")
    runtime = tl.split.prepare(
        _model, mx.ones((2, 4)), split_request("after:maximum", backend="mlx")
    )
    with pytest.raises(SplitUnsupportedError, match="unavailable|without gpu backend"):
        runtime.with_placement(tl.split.PlacementPlan.across("cpu", name))


def test_mlx_execution_failure_restores_stream() -> None:
    """A failed captured operation unwinds the per-segment stream scope."""

    runtime = tl.split.prepare(
        _model, mx.ones((2, 4)), split_request("after:maximum", backend="mlx")
    ).with_placement(tl.split.PlacementPlan.on(mx.new_stream(mx.cpu)))
    ambient = mx.default_stream(mx.cpu)
    target = next(
        node.target
        for node in runtime.trace_graph.nodes
        if node.canonical_id in runtime.plan.suffix_node_ids and node.target is not None
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Raise inside the selected stream to exercise scope restoration."""

        raise ValueError("planned operation failure")

    object.__setattr__(target, "func", fail)
    with pytest.raises(SplitUnsupportedError, match="planned operation failure"):
        runtime.replay(mx.ones((2, 4)))
    assert mx.default_stream(mx.cpu) == ambient


@pytest.mark.skipif(not _gpu_available(), reason="MLX GPU backend is unavailable")
@pytest.mark.parametrize("ambient_device", [mx.cpu, mx.gpu])
def test_mlx_cross_device_scopes_restore_both_device_defaults(ambient_device: Any) -> None:
    """Heterogeneous stream scopes restore both native defaults for either ambient device."""

    previous_device = mx.default_device()
    cpu_stream, gpu_stream = mx.default_stream(mx.cpu), mx.default_stream(mx.gpu)
    try:
        mx.set_default_device(ambient_device)
        runtime = tl.split.prepare(
            _model,
            mx.ones((2, 4)),
            split_request(
                "after:maximum",
                backend="mlx",
                placement=tl.split.PlacementPlan.across(
                    mx.new_stream(mx.cpu), mx.new_stream(mx.gpu)
                ),
            ),
        )
        result = runtime.replay(mx.ones((5, 4)))
        assert bool(mx.array_equal(result, mx.full((5, 4), 2)))
        assert mx.default_device() == mx.Device(ambient_device)
        assert mx.default_stream(mx.cpu) == cpu_stream
        assert mx.default_stream(mx.gpu) == gpu_stream
    finally:
        mx.set_default_stream(cpu_stream)
        mx.set_default_stream(gpu_stream)
        mx.set_default_device(previous_device)


@pytest.mark.skipif(not _gpu_available(), reason="MLX GPU backend is unavailable")
@pytest.mark.parametrize("prefix_device,suffix_device", [("cpu", "gpu"), ("gpu", "cpu")])
def test_mlx_native_cross_device_replay_and_gradients(
    prefix_device: str,
    suffix_device: str,
) -> None:
    """Native heterogeneous execution matches unsplit outputs and input gradients."""

    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    class Model(nn.Module):
        """Train native parameterized layers on opposite devices."""

        def __init__(self) -> None:
            """Create parameters consumed by each segment."""

            super().__init__()
            self.first = nn.Linear(4, 4)
            self.last = nn.Linear(4, 2)

        def __call__(self, x: Any) -> Any:
            """Evaluate the two native layers across an explicit boundary."""

            return self.last(mx.maximum(self.first(x), 0))

    model = Model()
    x = mx.arange(12, dtype=mx.float32).reshape(3, 4) - 2
    targets = mx.ones((3, 2))
    runtime = tl.split.prepare(
        model,
        x,
        split_request(
            "after:maximum",
            backend="mlx",
            trainable=True,
            placement=tl.split.PlacementPlan.across(prefix_device, suffix_device),
        ),
    )
    boundary = runtime.run_training_prefix(x)
    suffix = runtime.train_suffix_result(boundary, targets, optimizer=optim.SGD(0.01))
    prefix = runtime.backward_prefix(boundary, suffix.boundary_grads, optimizer=optim.SGD(0.01))
    expected = mx.grad(lambda a: mx.mean(mx.square(model(a) - targets)))(x)
    _loss, expected_params = nn.value_and_grad(
        model, lambda: mx.mean(mx.square(model(x) - targets))
    )()
    mx.eval(prefix["inputs"][0], expected)
    assert bool(mx.allclose(prefix["inputs"][0], expected, atol=1e-4, rtol=1e-4))
    actual_params = {**prefix["parameter_grads"], **suffix.parameter_grads}
    for name, expected_param in tree_flatten(expected_params):
        assert bool(mx.allclose(actual_params[name], expected_param, atol=1e-4, rtol=1e-4))
    optim.SGD(0.01).update(model, expected_params)
    assert bool(mx.allclose(runtime.replay(x), model(x), atol=1e-4, rtol=1e-4))
