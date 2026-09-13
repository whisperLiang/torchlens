"""Static input shapes and explicit batch-axis semantics for split replay."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.split import SplitFeatures, SplitRequest, pipeline
from torchlens.split.errors import SplitBoundaryError


@pytest.fixture
def capture_shapes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """Observe the shapes passed to actual model captures."""

    shapes: list[tuple[int, ...]] = []
    original = pipeline.capture_model

    def capture(model: Any, inputs: tuple[Any, ...], spec: Any, **kwargs: Any) -> Any:
        """Record captures without relying on restored model attributes."""

        shapes.append(tuple(inputs[0].shape))
        return original(model, inputs, spec, **kwargs)

    monkeypatch.setattr(pipeline, "capture_model", capture)
    return shapes


def test_batch_axis_serialization_distinguishes_auto_and_static() -> None:
    """None means inference, while an empty mapping explicitly disables batching."""

    assert SplitFeatures().batch_axes is None
    for axes in (None, {}, {"/args/0": 0}):
        features = SplitFeatures(batch_axes=axes)
        serialized = json.loads(json.dumps(features.as_dict()))
        assert serialized["batch_axes"] == axes
        assert SplitFeatures(**serialized).batch_axes == axes


@pytest.mark.parametrize("axes", [None, {}], ids=["automatic", "explicit-static"])
@pytest.mark.parametrize("width", [1, 4], ids=["singleton-feature", "feature-vector"])
def test_linear_vector_is_never_resized_or_probed(
    axes: dict[str, int] | None,
    width: int,
    capture_shapes: list[tuple[int, ...]],
) -> None:
    """A legal unbatched Linear input retains every feature, even at width one."""

    model = torch.nn.Linear(width, 3)
    example = torch.randn(width)
    request = SplitRequest(point=tl.split.after("linear"), features=SplitFeatures(batch_axes=axes))
    runtime = tl.split.prepare(model, example, request)

    assert capture_shapes == [(width,)]
    assert runtime.trace_graph.shape_program.input_batch_axes == {}
    assert runtime.trace_graph.shape_program.traced_input_shapes == {"/args/0": (width,)}
    assert runtime.trace_graph.shape_program.witness_batch_sizes == ()
    assert runtime.batch_validation["probe_batch_size"] is None
    value = torch.randn(width)
    torch.testing.assert_close(runtime.replay(value), model(value))
    assert runtime.replay(value).shape == (3,)
    assert capture_shapes == [(width,)]


def test_explicit_static_matrix_has_exact_runtime_shape_guards(
    capture_shapes: list[tuple[int, ...]],
) -> None:
    """No-axis matrix inputs are not normalized, probed, or silently rebatched."""

    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
    request = SplitRequest(
        point=tl.split.after("relu"),
        features=SplitFeatures(batch_axes={}, training=True),
    )
    example = torch.randn(7, 4)
    runtime = tl.split.prepare(model, example, request)

    assert capture_shapes == [(7, 4)]
    assert runtime.trace_graph.shape_program.input_batch_axes == {}
    torch.testing.assert_close(runtime.replay(example), model(example))
    for run in (runtime.run_prefix, runtime.run_training_prefix, runtime.replay):
        for shape in ((2, 4), (7, 5), (4,)):
            with pytest.raises(SplitBoundaryError, match="shape|dimension|rank"):
                run(torch.randn(shape))
    assert capture_shapes == [(7, 4)]


def test_rank_one_batch_requires_explicit_axis(
    capture_shapes: list[tuple[int, ...]],
) -> None:
    """A genuine batch of scalars still supports canonical capture and B=2 probing."""

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply independent scalar operations to every batch item."""

            return torch.relu(x) * 2

    model = Model()
    request = SplitRequest(
        point=tl.split.after("relu"), features=SplitFeatures(batch_axes={"/args/0": 0})
    )
    runtime = tl.split.prepare(model, torch.randn(4), request)
    assert capture_shapes == [(1,), (2,)]
    assert runtime.trace_graph.shape_program.input_batch_axes == {"/args/0": 0}
    assert runtime.batch_validation["status"] == "passed"
    for batch in (1, 2, 7):
        value = torch.randn(batch)
        torch.testing.assert_close(runtime.replay(value), model(value))
    assert capture_shapes == [(1,), (2,)]


def test_nested_static_inputs_do_not_require_a_batch_declaration() -> None:
    """An empty mapping explicitly supports unbatched tensor leaves in containers."""

    class Model(torch.nn.Module):
        def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            """Read a fixed-shape feature vector from a nested container."""

            return torch.relu(inputs["features"]) * 2

    model = Model()
    request = SplitRequest(point=tl.split.after("relu"), features=SplitFeatures(batch_axes={}))
    value = {"features": torch.randn(4)}
    runtime = tl.split.prepare(model, value, request)
    assert runtime.trace_graph.shape_program.input_batch_axes == {}
    assert runtime.trace_graph.shape_program.traced_input_shapes == {"/args/0/features": (4,)}
    assert runtime.trace_graph.shape_program.witness_batch_sizes == ()
    torch.testing.assert_close(runtime.replay(value), model(value))
    with pytest.raises(SplitBoundaryError, match="shape|dimension|rank"):
        runtime.run_prefix({"features": torch.randn(5)})


def test_scalar_auto_input_remains_static(capture_shapes: list[tuple[int, ...]]) -> None:
    """Rank-zero input tensors have no batch axis to normalize or probe."""

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Preserve scalar rank across simple arithmetic."""

            return torch.relu(x) * 2

    model = Model()
    runtime = tl.split.prepare(model, torch.tensor(3.0), SplitRequest(point=tl.split.after("relu")))
    assert capture_shapes == [()]
    assert runtime.trace_graph.shape_program.input_batch_axes == {}
    torch.testing.assert_close(runtime.replay(torch.tensor(4.0)), model(torch.tensor(4.0)))
    with pytest.raises(SplitBoundaryError, match="shape|dimension|rank"):
        runtime.run_prefix(torch.tensor([4.0]))
