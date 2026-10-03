"""MLX split replay, dynamic batch, input binding and boundary cache regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from v2_helpers import split_request

mx = pytest.importorskip("mlx.core", exc_type=ImportError)

import torchlens as tl  # noqa: E402
from torchlens.split.errors import SplitBoundaryError  # noqa: E402

pytestmark = pytest.mark.backend_mlx


def _input(batch: int) -> Any:
    """Return distinct rows that reveal stale captured values and scalar rewrites."""

    return mx.arange(batch * 4, dtype=mx.float32).reshape((batch, 4)) - 3


def _split_merge(x: Any) -> Any:
    """Return a deterministic multi-output graph with an integer scalar operand."""

    hidden = mx.maximum(x, 0)
    left, right = mx.split(hidden, 2, axis=-1)
    return mx.add(mx.multiply(left, 2), right)


def test_mlx_replays_every_supported_cut_at_changed_batches() -> None:
    """One MLX capture supports cuts before and after each multi-output call."""

    seed = tl.split.prepare(_split_merge, _input(2), split_request("after:maximum", backend="mlx"))
    graph = seed.trace_graph
    assert graph.shape_program.batch_probe.status == "passed"
    report = seed.split_points()
    assert report.supported
    for candidate in report.supported:
        runtime = seed.at(candidate.point)
        assert runtime.trace_graph is graph
        for batch in (1, 3, 7):
            x = _input(batch)
            replayed, expected = runtime.replay(x), _split_merge(x)
            mx.eval(replayed, expected)
            assert replayed.shape == expected.shape == (batch, 2)
            assert bool(mx.allclose(replayed, expected))


def test_mlx_reshape_rewrites_only_the_shape_operand() -> None:
    """Reshape batch extents change while integer scalar operands stay literal."""

    def model(x: Any) -> Any:
        hidden = mx.maximum(x, 0)
        reshaped = mx.reshape(hidden, shape=(hidden.shape[0], -1))
        return mx.multiply(reshaped, 2)

    runtime = tl.split.prepare(model, _input(2), split_request("after:maximum", backend="mlx"))
    x = _input(7)
    replayed, expected = runtime.replay(x), model(x)
    mx.eval(replayed, expected)
    assert replayed.shape == expected.shape == (7, 4)
    assert bool(mx.allclose(replayed, expected))


def test_mlx_binds_nested_inputs_and_keyword_arrays() -> None:
    """Input dictionary insertion order cannot change replay argument binding."""

    def model(pair: dict[str, Any], *, extra: dict[str, Any]) -> Any:
        hidden = mx.maximum(mx.subtract(pair["a"], pair["z"]), 0)
        return mx.add(hidden, extra["offset"])

    x = _input(2)
    request = split_request(
        "after:maximum",
        backend="mlx",
        batch_axes={"/args/0/a": 0, "/args/0/z": 0, "/kwargs/extra/offset": 0},
    )
    runtime = tl.split.prepare(
        model,
        {"z": x, "a": mx.multiply(x, 3)},
        request,
        input_kwargs={"extra": {"offset": mx.add(x, 1)}},
    )
    x = _input(3)
    pair = {"a": mx.multiply(x, 3), "z": x}
    kwargs = {"extra": {"offset": mx.add(x, 1)}}
    replayed, expected = runtime.replay(pair, input_kwargs=kwargs), model(pair, **kwargs)
    mx.eval(replayed, expected)
    assert replayed.shape == expected.shape == (3, 4)
    assert bool(mx.allclose(replayed, expected))


def test_mlx_shared_input_identities_must_be_preserved() -> None:
    """Breaking an input alias refuses before replay can use the wrong parent."""

    def model(x: Any, y: Any) -> Any:
        return mx.subtract(mx.maximum(x, 0), y)

    example = _input(2)
    runtime = tl.split.prepare(
        model, (example, example), split_request("after:maximum", backend="mlx")
    )
    x = _input(3)
    replayed, expected = runtime.replay(x, x), model(x, x)
    mx.eval(replayed, expected)
    assert bool(mx.allclose(replayed, expected))
    with pytest.raises(SplitBoundaryError, match="shared during capture"):
        runtime.replay(x, mx.add(x, 1))


def test_mlx_replays_native_module_calls_with_captured_parameters() -> None:
    """A generated suffix preserves the parameters of a captured native MLX module."""

    import mlx.nn as nn

    class Model(nn.Module):
        """Small native MLX MLP split between its linear modules."""

        def __init__(self) -> None:
            """Create the two parameterized layers."""

            super().__init__()
            self.hidden = nn.Linear(4, 6)
            self.output = nn.Linear(6, 2)

        def __call__(self, x: Any) -> Any:
            """Evaluate the native layers around the split point."""

            return self.output(mx.maximum(self.hidden(x), 0))

    model = Model()
    runtime = tl.split.prepare(model, _input(2), split_request("after:maximum", backend="mlx"))
    for batch in (1, 3):
        x = _input(batch)
        replayed, expected = runtime.replay(x), model(x)
        mx.eval(replayed, expected)
        assert replayed.shape == expected.shape == (batch, 2)
        assert bool(mx.allclose(replayed, expected))


def test_mlx_boundary_cache_round_trips_split_outputs(tmp_path: Path) -> None:
    """Native MLX boundary values survive signed cache save/load and suffix replay."""

    runtime = tl.split.prepare(
        _split_merge,
        _input(2),
        split_request("after:split_1_3", backend="mlx", boundary_cache=True),
    )
    x = _input(3)
    boundary = runtime.run_prefix(x)
    assert len(boundary.tensors) == 2
    cache_path = tmp_path / "mlx_boundary"
    runtime.save_boundary(boundary, cache_path)
    loaded = runtime.load_boundary(cache_path)
    assert all(isinstance(value, mx.array) for value in loaded.tensors.values())
    replayed, expected = runtime.run_suffix(loaded), _split_merge(x)
    mx.eval(replayed, expected)
    assert replayed.shape == expected.shape == (3, 2)
    assert bool(mx.allclose(replayed, expected))


def test_mlx_reconstructs_one_final_leaf_from_a_multi_output_call() -> None:
    """A final model path need not match the native split call's leaf path."""

    def model(x: Any) -> Any:
        hidden = mx.maximum(x, 0)
        left, right = mx.split(hidden, 2, axis=-1)
        return {"right": right, "sum": mx.add(left, right), "meta": ("mlx", 2, None, [])}

    runtime = tl.split.prepare(model, _input(2), split_request("after:maximum", backend="mlx"))
    x = _input(3)
    replayed, expected = runtime.replay(x), model(x)
    assert tuple(replayed) == tuple(expected) == ("right", "sum", "meta")
    assert replayed["meta"] == expected["meta"]
    for key in ("right", "sum"):
        mx.eval(replayed[key], expected[key])
        assert bool(mx.allclose(replayed[key], expected[key]))
