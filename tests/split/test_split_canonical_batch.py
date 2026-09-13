"""Canonical small-batch capture across native split backends."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import pipeline
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError


def _check_native_backend(backend: str) -> None:
    """Check cyclic resizing and batch-independent capture on one backend."""

    if backend == "jax":
        native = pytest.importorskip("jax.numpy")
        from torchlens.split.adapters.jax import JaxSplitAdapter

        adapter: Any = JaxSplitAdapter()
        make = native.array

        def relu(x: Any) -> Any:
            """Apply the JAX boundary operation."""

            return native.maximum(x, 0)

        boundary = "after:max"
    elif backend == "tf":
        native = pytest.importorskip("tensorflow")
        from torchlens.split.adapters.tf import TfSplitAdapter

        adapter = TfSplitAdapter()
        make = native.constant
        relu = native.nn.relu
        boundary = "after:relu"
    elif backend == "paddle":
        native = pytest.importorskip("paddle")
        from torchlens.split.adapters.paddle import PaddleSplitAdapter

        adapter = PaddleSplitAdapter()
        native.set_device("cpu")
        make = native.to_tensor

        def relu(x: Any) -> Any:
            """Resolve the Paddle operation after capture wrappers are installed."""

            return native.nn.functional.relu(x)

        boundary = "after:relu"
    else:
        native = pytest.importorskip("tinygrad")
        from torchlens.split.adapters.tinygrad import TinygradSplitAdapter

        adapter = TinygradSplitAdapter()

        def make(values: Any) -> Any:
            """Create a realized tinygrad input."""

            return native.Tensor(values).realize()

        def relu(x: Any) -> Any:
            """Apply the tinygrad boundary operation."""

            return x.relu()

        boundary = "after:where"

    for dtype in (np.float32, np.int32, np.bool_):
        values = np.arange(12).reshape(3, 4).astype(dtype)
        source = make(values)
        for axis in (0, -1):
            for batch in (1, 2, 7):
                resized = adapter.resize_batch(source, axis=axis, batch_size=batch)
                payload = resized.numpy() if hasattr(resized, "numpy") else np.asarray(resized)
                expected = np.take(values, np.arange(batch) % values.shape[axis], axis=axis)
                np.testing.assert_array_equal(payload, expected)
                assert resized.dtype == source.dtype
                assert str(
                    getattr(resized, "place", None) or getattr(resized, "device", None)
                ) == str(getattr(source, "place", None) or getattr(source, "device", None))
        original = source.numpy() if hasattr(source, "numpy") else np.asarray(source)
        np.testing.assert_array_equal(original, values)

    def model(x: Any) -> Any:
        """Compute an elementwise function without a fixed batch extent."""

        return relu(x) * 2.0 + 1.0

    for axes, shape in ((None, (4,)), ({}, (4,)), ({}, (7, 4))):
        static_input = make(np.ones(shape, dtype=np.float32))
        static = tl.split.prepare(
            model,
            static_input,
            split_request(boundary, backend=backend, batch_axes=axes),
        )
        program = static.trace_graph.shape_program
        assert program.input_batch_axes == {}
        assert program.traced_input_shapes == {"/args/0": shape}
        assert program.witness_batch_sizes == ()
        assert static.batch_validation["probe_batch_size"] is None
        assert adapter.allclose(
            static.replay(static_input), model(static_input), atol=1e-5, rtol=1e-4
        )
        with pytest.raises(SplitBoundaryError, match="non-batch input"):
            static.run_prefix(make(np.ones((shape[0] + 1, *shape[1:]), dtype=np.float32)))

    runtimes = [
        tl.split.prepare(
            model,
            make(np.ones((batch, 4), dtype=np.float32)),
            split_request(boundary, backend=backend),
        )
        for batch in (32, 8)
    ]
    assert all(runtime.traced_batch_size == 1 for runtime in runtimes)
    assert all(runtime.batch_validation["status"] == "passed" for runtime in runtimes)
    assert runtimes[0].graph_identity == runtimes[1].graph_identity
    assert runtimes[0].split_id == runtimes[1].split_id
    for batch in (1, 3, 8):
        x = make(np.arange(batch * 4, dtype=np.float32).reshape(batch, 4) - 2)
        actual = runtimes[0].replay(x)
        expected_output = model(x)
        assert adapter.allclose(actual, expected_output, atol=1e-5, rtol=1e-5)

    def branch_model(x: Any) -> Any:
        """Keep tensor topology stable while changing the multiplier at B=2."""

        return relu(x) * (3.0 if x.shape[0] >= 2 else 2.0)

    restricted = tl.split.prepare(
        branch_model,
        make(np.ones((8, 4), dtype=np.float32)),
        split_request(boundary, backend=backend),
    )
    assert restricted.batch_validation["status"] == "failed"
    assert "numeric mismatch" in restricted.batch_validation["reason"]
    singleton = make(np.ones((1, 4), dtype=np.float32))
    assert adapter.allclose(
        restricted.replay(singleton), branch_model(singleton), atol=1e-5, rtol=1e-5
    )
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        restricted.replay(make(np.ones((2, 4), dtype=np.float32)))


@pytest.mark.heavy
@pytest.mark.parametrize("backend", ["jax", "tf", "tinygrad", "paddle"])
def test_native_canonical_capture(backend: str) -> None:
    """Large examples produce B=1 captures with stable identities and replay."""

    module = "tensorflow" if backend == "tf" else backend
    if find_spec(module) is None:
        pytest.skip(f"{module!r} is not installed.")
    # Isolate native runtime state, following the existing Paddle split tests.
    env = os.environ.copy()
    if backend != "tinygrad":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}\n"
            "from test_split_canonical_batch import _check_native_backend\n"
            f"_check_native_backend({backend!r})",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"{backend} canonical batch subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_rebatch_failure_does_not_capture_original_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken backend resize implementation cannot silently use a large batch."""

    torch = pytest.importorskip("torch")
    from torchlens.split.adapters.torch import TorchSplitAdapter

    adapter = TorchSplitAdapter()
    captures: list[int] = []

    def broken_resize(value: Any, axis: int, batch_size: int) -> Any:
        """Simulate an adapter missing its canonical-batch implementation."""

        raise NotImplementedError("resize unavailable")

    def capture(model: Any, inputs: tuple[Any, ...], spec: Any, **kwargs: Any) -> Any:
        """Record any unintended fallback capture."""

        captures.append(int(inputs[0].shape[0]))
        return object()

    monkeypatch.setattr(adapter, "resize_batch", broken_resize)
    monkeypatch.setattr(pipeline, "capture_model", capture)
    with pytest.raises(SplitUnsupportedError, match="Cannot construct canonical batch B=1"):
        pipeline.capture_canonical_model(
            object(), (torch.ones(32, 4),), split_request("after:relu"), adapter=adapter
        )
    assert captures == []


@pytest.mark.parametrize("accept_two", [False, True])
def test_canonical_capture_only_retries_at_two(
    monkeypatch: pytest.MonkeyPatch, accept_two: bool
) -> None:
    """Only a real B=1 capture failure permits B=2, never the original B=32."""

    torch = pytest.importorskip("torch")
    from torchlens.split.adapters.torch import TorchSplitAdapter

    captures: list[int] = []
    result = object()

    def capture(model: Any, inputs: tuple[Any, ...], spec: Any, **kwargs: Any) -> Any:
        """Reject the singleton batch as a batch-sensitive model would."""

        batch = int(inputs[0].shape[0])
        captures.append(batch)
        if batch == 1 or not accept_two:
            raise ValueError(f"model rejects B={batch}")
        return result

    monkeypatch.setattr(pipeline, "capture_model", capture)
    args = (object(), (torch.ones(32, 4),), split_request("after:relu"))
    if accept_two:
        actual, inputs, _kwargs, batch_spec = pipeline.capture_canonical_model(
            *args, adapter=TorchSplitAdapter()
        )
        assert actual is result
        assert inputs[0].shape == (2, 4)
        assert batch_spec.canonical_batch_size == 2
    else:
        with pytest.raises(ValueError, match="model rejects B=2"):
            pipeline.capture_canonical_model(*args, adapter=TorchSplitAdapter())
    assert captures == [1, 2]
