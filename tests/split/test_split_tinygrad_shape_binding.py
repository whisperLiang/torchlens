"""Regression coverage for tinygrad's captured versus runtime shape literals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from v2_helpers import split_request

import torchlens as tl


def test_tinygrad_reduce_does_not_rewrite_bound_parent_shape() -> None:
    """A reduced singleton axis must not become a second dynamic batch axis."""

    tinygrad = pytest.importorskip("tinygrad")
    Tensor = tinygrad.Tensor

    def model(x: Any) -> Any:
        """Reduce a non-batch axis before removing its singleton dimension."""

        return x.mean(axis=1)

    sample = Tensor.ones(2, 4, 4, device="CPU").realize()
    for point in ("25%", "50%", "75%"):
        runtime = tl.split.prepare(model, sample, split_request(point, backend="tinygrad"))
        assert runtime.trace_graph.shape_program.batch_probe.status == "passed"
        for batch in (1, 2, 4):
            values = np.arange(batch * 16, dtype=np.float32).reshape(batch, 4, 4)
            x = Tensor(values, device="CPU").realize()
            actual = runtime.replay(x).realize()
            assert actual.shape == (batch, 4)
            np.testing.assert_allclose(actual.numpy(), values.mean(axis=1))


def test_tinygrad_constant_linear_weights_broadcast_at_runtime_batch() -> None:
    """Constant-backed weights need an explicit broadcast at low-level MULs."""

    tinygrad = pytest.importorskip("tinygrad")
    Tensor = tinygrad.Tensor
    weight = Tensor.ones(4, 6, device="CPU").realize()

    def model(x: Any) -> Any:
        """Apply a fixed constant-backed dense weight."""

        return (x @ weight.T).relu()

    sample = Tensor.ones(2, 6, device="CPU").realize()
    runtime = tl.split.prepare(model, sample, split_request("50%", backend="tinygrad"))
    assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
    for point in ("25%", "50%", "75%"):
        selected = runtime.at(split_request(point, backend="tinygrad").point)
        for batch in (1, 2, 3):
            data = np.arange(batch * 6, dtype=np.float32).reshape(batch, 6) - 4
            x = Tensor(data, device="CPU").realize()
            expected = np.maximum(data @ np.ones((6, 4), dtype=np.float32), 0)
            np.testing.assert_allclose(selected.replay(x).numpy(), expected)


def test_tinygrad_padding_does_not_broadcast_its_parameter_input() -> None:
    """A PAD must receive the unpadded state, not an output-sized expansion."""

    tinygrad = pytest.importorskip("tinygrad")
    Tensor = tinygrad.Tensor

    class PaddedState:
        """Add a padded, buffer-backed parameter to a batched input."""

        def __init__(self) -> None:
            """Create a non-scalar parameter with a singleton padding axis."""

            self.weight = Tensor(
                np.arange(4, dtype=np.float32).reshape(1, 1, 4), device="CPU"
            ).realize()

        def __call__(self, x: Any) -> Any:
            """Pad once, then rely on native elementwise broadcasting."""

            return x + self.weight.pad((None, (0, 2), None))

    model = PaddedState()
    sample = Tensor.ones(2, 3, 4, device="CPU").realize()
    runtime = tl.split.prepare(model, sample, split_request("50%", backend="tinygrad"))
    assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
    for node in runtime.trace_graph.compute_nodes:
        for kind in ("before", "after"):
            selected = runtime.at(
                split_request(f"{kind}:{node.canonical_id}", backend="tinygrad").point
            )
            for batch in (1, 3):
                data = np.arange(batch * 12, dtype=np.float32).reshape(batch, 3, 4)
                x = Tensor(data, device="CPU").realize()
                expected = data + np.pad(
                    np.arange(4, dtype=np.float32).reshape(1, 1, 4), ((0, 0), (0, 2), (0, 0))
                )
                np.testing.assert_allclose(selected.replay(x).numpy(), expected)
