"""Backend-neutral dynamic-batch shape-program contracts."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.split import SplitFeatures, SplitRequest, percent
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError


def _request(*, batch_axes: dict[str, int]) -> SplitRequest:
    """Return a strict dynamic-batch request for shape-program tests."""

    return SplitRequest(
        point=percent(50),
        backend="torch",
        features=SplitFeatures(dynamic_batch=(1, 5), batch_axes=batch_axes),
    )


def test_explicit_multi_input_batch_axes_propagate_through_permute() -> None:
    """Different input axes bind one B and propagate through axis movement."""

    class Model(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            aligned = right.permute(1, 0, 2)
            return (left + aligned).reshape(left.shape[0], -1) * 2.0

    model = Model().eval()
    left = torch.ones(2, 3, 4)
    right = torch.ones(3, 2, 4)
    runtime = tl.split.prepare(
        model,
        (left, right),
        _request(batch_axes={"/args/0": 0, "/args/1": 1}),
    )

    for batch in (1, 3, 5):
        replay_left = torch.ones(batch, 3, 4)
        replay_right = torch.ones(3, batch, 4)
        torch.testing.assert_close(
            runtime.replay(replay_left, replay_right),
            model(replay_left, replay_right),
        )


def test_kwargs_batch_axis_uses_json_pointer_and_runtime_override() -> None:
    """Keyword tensor leaves participate in explicit batch consistency checks."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
            return (values * mask).reshape(values.shape[0], -1)

    model = Model().eval()
    values = torch.ones(2, 3, 4)
    mask = torch.ones(2, 3, 4)
    runtime = tl.split.prepare(
        model,
        values,
        _request(batch_axes={"/args/0": 0, "/kwargs/mask": 0}),
        input_kwargs={"mask": mask},
    )

    replay_values = torch.ones(4, 3, 4)
    replay_mask = torch.ones(4, 3, 4)
    torch.testing.assert_close(
        runtime.replay(replay_values, input_kwargs={"mask": replay_mask}),
        model(replay_values, mask=replay_mask),
    )
    with pytest.raises(SplitBoundaryError, match="disagree"):
        runtime.replay(replay_values, input_kwargs={"mask": torch.ones(3, 3, 4)})


def test_explicit_mode_rejects_unique_undeclared_batch_dimension() -> None:
    """A matching dimension alone cannot prove an omitted tensor carries batch."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return values * mask

    model = Model().eval()
    with pytest.raises(SplitUnsupportedError, match="undeclared tensor paths"):
        tl.split.prepare(
            model,
            (torch.ones(2, 3), torch.ones(2, 3)),
            _request(batch_axes={"/args/0": 0}),
        )


def test_explicit_mode_rejects_ambiguous_undeclared_batch_input() -> None:
    """An omitted tensor with multiple candidate batch axes requires a declaration."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return values + mask[:, 0]

    with pytest.raises(SplitUnsupportedError, match="undeclared tensor paths"):
        tl.split.prepare(
            Model().eval(),
            (torch.ones(2, 3), torch.ones(2, 2, 3)),
            _request(batch_axes={"/args/0": 0}),
        )


def test_validate_equivalence_uses_prepared_and_override_kwargs() -> None:
    """Equivalence validation invokes both paths with the same keyword arguments."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
            return values * scale

    model = Model().eval()
    values = torch.ones(2, 3)
    runtime = tl.split.prepare(
        model,
        values,
        _request(batch_axes={"/args/0": 0, "/kwargs/scale": 0}),
        input_kwargs={"scale": torch.full((2, 3), 2.0)},
    )

    assert runtime.validate_equivalence(model, (values,))
    assert runtime.validate_equivalence(
        model,
        (values,),
        input_kwargs={"scale": torch.full((2, 3), 3.0)},
    )


def test_reshape_retains_batch_product_expression() -> None:
    """A reshape dimension derived from B times a fixed axis is solved at runtime."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values.reshape(values.shape[0] * values.shape[1], -1) + 1.0

    model = Model().eval()
    example = torch.ones(2, 3, 4)
    runtime = tl.split.prepare(
        model,
        example,
        _request(batch_axes={"/args/0": 0}),
    )

    for batch in (1, 4, 5):
        values = torch.ones(batch, 3, 4)
        output = runtime.replay(values)
        assert output.shape == (batch * 3, 4)
        torch.testing.assert_close(output, model(values))


def test_explicit_batch_axes_reject_unknown_and_non_batch_changes() -> None:
    """Invalid declarations and runtime non-batch changes fail structurally."""

    model = torch.nn.Sequential(torch.nn.Flatten(start_dim=1), torch.nn.Linear(12, 2)).eval()
    example = torch.ones(2, 3, 4)
    with pytest.raises(SplitUnsupportedError, match="Unknown dynamic-batch input paths"):
        tl.split.prepare(
            model,
            example,
            _request(batch_axes={"/args/9": 0}),
        )

    runtime = tl.split.prepare(
        model,
        example,
        _request(batch_axes={"/args/0": 0}),
    )
    with pytest.raises(SplitBoundaryError, match="non-batch dimension"):
        runtime.replay(torch.ones(3, 4, 4))


def test_auto_inference_rejects_nested_ambiguous_inputs() -> None:
    """Conservative auto mode requires explicit paths for nested input trees."""

    class Model(torch.nn.Module):
        def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
            return values["x"] * 2.0

    with pytest.raises(SplitUnsupportedError, match="could not be inferred conservatively"):
        tl.split.prepare(
            Model().eval(),
            {"x": torch.ones(2, 3)},
            _request(batch_axes={}),
        )
