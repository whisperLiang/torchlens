"""Canonical input resizing must not silently erase shared-storage semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from v2_helpers import split_request

import torchlens as tl
from torchlens.split.adapters.torch import TorchSplitAdapter
from torchlens.split.batching import rebatch_inputs
from torchlens.split.errors import SplitUnsupportedError


@pytest.mark.filterwarnings(
    "ignore:TorchLens cannot verify the captured model-input semantics:"
    "torchlens._errors.TorchLensCaptureGapWarning"
)
def test_same_geometry_aliases_preserve_the_captured_branch() -> None:
    """Aliased in-place reads retain their meaning at B=1, B=2 and extrapolated B."""

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Observe an in-place write through the other input's shared storage."""

            x.add_(1)
            return y * (3 if y[0, 0].item() > 1 else 2)

    example = torch.ones(4, 4)
    model = Model()
    runtime = tl.split.prepare(
        model, (example, example.view_as(example)), split_request("after:add")
    )
    assert runtime.batch_validation["status"] == "passed"
    torch.testing.assert_close(example, torch.ones_like(example))
    for batch in (1, 2, 4):
        value = torch.ones(batch, 4)
        actual = runtime.replay(value, value.view_as(value))
        expected_input = torch.ones(batch, 4)
        expected = model(expected_input, expected_input.view_as(expected_input))
        torch.testing.assert_close(actual, expected)


def test_rebatch_preserves_duplicate_identity_and_views_across_input_trees() -> None:
    """One alias family spans positional, nested, and keyword leaves."""

    source = torch.ones(4, 3)
    view = source.view_as(source)
    args, kwargs = rebatch_inputs(
        (view, {"x": source}),
        {"same": source},
        axes={"/args/0": 0, "/args/1/x": 0, "/kwargs/same": 0},
        batch_size=2,
        adapter=TorchSplitAdapter(),
    )
    assert args[1]["x"] is kwargs["same"]
    assert args[0] is not args[1]["x"]
    args[0].add_(2)
    torch.testing.assert_close(kwargs["same"], torch.full((2, 3), 3.0))
    torch.testing.assert_close(source, torch.ones_like(source))


@pytest.mark.parametrize("kind", ["offset", "transpose", "separate-storage", "axes"])
def test_unsupported_alias_geometry_refuses_before_model_execution(kind: str) -> None:
    """Never mistake independent Torch storage wrappers for disjoint memory."""

    base = torch.ones(4, 5)
    left, right = base[:, :4], base[:, 1:]
    axes = {"/args/0": 0, "/args/1": 0}
    if kind == "transpose":
        left = torch.ones(4, 4)
        right = left.t()
    elif kind == "separate-storage":
        values = np.ones(20, dtype=np.float32)
        left = torch.from_numpy(values[:16]).reshape(4, 4)
        right = torch.from_numpy(values[1:17]).reshape(4, 4)
    elif kind == "axes":
        left = torch.ones(4, 4)
        right = left.view_as(left)
        axes["/args/1"] = 1
    with pytest.raises(SplitUnsupportedError, match="input alias relationship"):
        rebatch_inputs((left, right), None, axes=axes, batch_size=1, adapter=TorchSplitAdapter())


def test_partial_overlap_prepare_refuses_before_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported alias is refused before any user model execution."""

    from torchlens.split import pipeline

    def unexpected_capture(*args: Any, **kwargs: Any) -> Any:
        """Fail if rebatching already altered the input relationship."""

        pytest.fail("capture must not run for unsupported overlapping input views")

    monkeypatch.setattr(pipeline, "capture_model", unexpected_capture)
    source = torch.ones(4, 5)
    with pytest.raises(SplitUnsupportedError, match="input alias relationship"):
        tl.split.prepare(torch.nn.Identity(), (source[:, :4], source[:, 1:]), split_request("50%"))


def test_disjoint_views_of_one_storage_remain_independent() -> None:
    """Sharing an allocation alone does not require an alias refusal."""

    base = torch.ones(4, 8)
    args, _ = rebatch_inputs(
        (base[:, :4], base[:, 4:]),
        None,
        axes={"/args/0": 0, "/args/1": 0},
        batch_size=2,
        adapter=TorchSplitAdapter(),
    )
    args[0].add_(1)
    torch.testing.assert_close(args[1], torch.ones(2, 4))
