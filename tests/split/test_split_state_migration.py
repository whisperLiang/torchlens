"""Owned split state survives placement changes and guards stale boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError
from torchlens.split.runtime import _state_values_fingerprint
from torchlens.split.state import SegmentState


class StateMigrationMlp(nn.Module):
    """Small deterministic model with trainable state on each side of a cut."""

    def __init__(self) -> None:
        """Construct two affine layers around a named activation."""

        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)
        with torch.no_grad():
            for value in self.parameters():
                value.fill_(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model with positive, nonzero prefix gradients."""

        return self.fc2(self.relu(self.fc1(x)))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_trained_state_survives_replacement_and_recut(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebuilt executables preserve learned values without modifying capture state."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for real cross-device migration.")
    if device == "cpu":
        monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = StateMigrationMlp()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:relu", trainable=True, placement=PlacementPlan.on(device)),
    )
    prefix = runtime.prefix_parameters()
    suffix = runtime.suffix_parameters()
    boundary = runtime.run_training_prefix(inputs)
    _, gradients = runtime.train_suffix(
        boundary,
        torch.zeros(3, 3, device=device),
        optimizer=torch.optim.SGD(suffix, lr=0.01),
    )
    runtime.backward_prefix(boundary, gradients, optimizer=torch.optim.SGD(prefix, lr=0.01))
    expected = runtime.replay(inputs).detach().cpu()
    assert not torch.equal(expected, model(inputs))

    rebound = runtime.with_placement(runtime.placement)
    assert [id(value) for value in rebound.prefix_parameters()] == [id(value) for value in prefix]
    assert [id(value) for value in rebound.suffix_parameters()] == [id(value) for value in suffix]
    torch.testing.assert_close(rebound.replay(inputs).cpu(), expected)

    for placement in (PlacementPlan.on("cpu"), PlacementPlan.unplaced()):
        moved = runtime.with_placement(placement)
        assert all(value.device.type == "cpu" for value in moved.prefix_parameters())
        assert len(moved.prefix_parameters()) == 2
        torch.testing.assert_close(moved.replay(inputs), expected)

    for point in (tl.split.before("fc1"), tl.split.after("fc2")):
        recut = runtime.at(point)
        torch.testing.assert_close(recut.replay(inputs).cpu(), expected)
        assert len(recut.prefix_parameters()) + len(recut.suffix_parameters()) == 4
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)


@pytest.mark.parametrize("training_boundary", [False, True])
def test_owned_updates_invalidate_direct_and_cached_boundaries(
    training_boundary: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replica update changes the execution fingerprint, including saved training roots."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = StateMigrationMlp()
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:relu", trainable=True, placement=PlacementPlan.on("cpu")),
    )
    boundary = (
        runtime.run_training_prefix(inputs) if training_boundary else runtime.run_prefix(inputs)
    )
    cache_path = tmp_path / "boundary"
    runtime.save_boundary(boundary, cache_path)
    cached = runtime.load_boundary(cache_path)
    assert cached.metadata["state_fingerprint"] == boundary.metadata["state_fingerprint"]
    assert cached.metadata["state_prefix_kind"] == boundary.metadata["state_prefix_kind"]
    runtime.run_suffix(cached)

    parameters = runtime.prefix_parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    for value in parameters:
        value.grad = torch.ones_like(value)
    optimizer.step()
    updated = runtime.run_prefix(inputs)
    assert updated.metadata["state_fingerprint"] != boundary.metadata["state_fingerprint"]
    runtime.run_suffix(updated)
    for stale in (boundary, cached):
        with pytest.raises(SplitBoundaryError, match="state_fingerprint"):
            runtime.run_suffix(stale)
        with pytest.raises(SplitBoundaryError, match="state_fingerprint"):
            runtime.train_suffix(stale, torch.zeros(3, 3))


def test_captured_and_training_prefix_fingerprints_remain_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached training boundary retains its source kind when backward state is removed."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    runtime = tl.split.prepare(
        StateMigrationMlp(),
        torch.ones(3, 4),
        split_request("after:relu", placement=PlacementPlan.on("cpu")),
    )
    inputs = torch.ones(3, 4)
    inference = runtime.run_prefix(inputs)
    parameters = runtime.prefix_parameters()
    with torch.no_grad():
        for value in parameters:
            value.add_(1)
    training = runtime.run_training_prefix(inputs)
    assert training.metadata["state_fingerprint"] != inference.metadata["state_fingerprint"]
    runtime.run_suffix(inference)
    cache_path = tmp_path / "training-boundary"
    runtime.save_boundary(training, cache_path)
    cached = runtime.load_boundary(cache_path)
    assert cached.metadata["supports_prefix_backward"] is False
    torch.testing.assert_close(runtime.run_suffix(cached), runtime.run_suffix(training))


@pytest.mark.parametrize("divergent", [False, True])
def test_recut_refuses_merging_distinct_inference_and_training_prefix_state(
    divergent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving a dual-prefix operation into the common suffix cannot discard learned state."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(
        StateMigrationMlp(),
        inputs,
        split_request("after:relu", placement=PlacementPlan.on("cpu")),
    )
    parameters = runtime.prefix_parameters()
    if divergent:
        with torch.no_grad():
            for value in parameters:
                value.add_(1)
    expected_inference = runtime.replay(inputs).detach()
    expected_training = runtime.run_suffix(runtime.run_training_prefix(inputs)).detach()
    assert torch.equal(expected_inference, expected_training) is not divergent

    for rebuilt in (runtime.at(runtime.request.point), runtime.with_placement(runtime.placement)):
        torch.testing.assert_close(rebuilt.replay(inputs), expected_inference)
        torch.testing.assert_close(
            rebuilt.run_suffix(rebuilt.run_training_prefix(inputs)), expected_training
        )

    if divergent:
        with pytest.raises(SplitUnsupportedError, match="divergent inference and training"):
            runtime.at(tl.split.before("fc1"))
    else:
        recut = runtime.at(tl.split.before("fc1"))
        torch.testing.assert_close(recut.replay(inputs), expected_inference)
        torch.testing.assert_close(
            recut.run_suffix(recut.run_training_prefix(inputs)), expected_training
        )

    assert [id(value) for value in runtime.prefix_parameters()] == [
        id(value) for value in parameters
    ]
    torch.testing.assert_close(runtime.replay(inputs), expected_inference)
    torch.testing.assert_close(
        runtime.run_suffix(runtime.run_training_prefix(inputs)), expected_training
    )


def test_recut_moves_updated_suffix_into_both_prefix_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reverse recut preserves learned suffix state in captured and training prefixes."""

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    model = StateMigrationMlp()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(
        model,
        inputs,
        split_request("after:relu", placement=PlacementPlan.on("cpu")),
    )
    runtime.run_prefix(inputs)
    parameters = runtime.suffix_parameters()
    assert len(parameters) == 2
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    for value in parameters:
        value.grad = torch.ones_like(value)
    optimizer.step()
    expected = runtime.replay(inputs).detach()
    assert not torch.equal(expected, model(inputs))

    recut = runtime.at(tl.split.after("fc2"))
    torch.testing.assert_close(recut.replay(inputs), expected)
    torch.testing.assert_close(recut.run_suffix(recut.run_training_prefix(inputs)), expected)
    assert {id(value) for value in parameters} <= {id(value) for value in recut.prefix_parameters()}
    torch.testing.assert_close(runtime.replay(inputs), expected)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)


@pytest.mark.parametrize("divergent", [False, True])
def test_recut_preserves_tied_identity_or_refuses_divergent_replicas(
    divergent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared parameter cannot silently choose one of two different trained values."""

    class TiedModel(nn.Module):
        """Reuse one affine module on both sides of a split."""

        def __init__(self) -> None:
            """Construct a tied weight and two uniquely named activations."""

            super().__init__()
            self.shared = nn.Linear(4, 4, bias=False)
            self.relu = nn.ReLU()
            self.tail = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Consume the same weight before and after the boundary."""

            return self.tail(self.shared(self.relu(self.shared(x))))

    monkeypatch.setattr(SegmentState, "_already_placed", lambda self, value: False)
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(
        TiedModel(),
        inputs,
        split_request("after:relu", trainable=True, placement=PlacementPlan.on("cpu")),
    )
    if divergent:
        with torch.no_grad():
            runtime.prefix_parameters()[0].add_(1)
    expected = runtime.replay(inputs)
    same_cut = runtime.at(runtime.request.point)
    torch.testing.assert_close(same_cut.replay(inputs), expected)
    if divergent:
        with pytest.raises(SplitUnsupportedError, match="divergent replicas"):
            runtime.at(tl.split.after("tail"))
    else:
        recut = runtime.at(tl.split.after("tail"))
        assert len(recut.prefix_parameters()) == 1
        torch.testing.assert_close(recut.replay(inputs), expected)


def test_bfloat16_state_fingerprint_hashes_unabridged_payload() -> None:
    """An update outside the printed tensor summary must still invalidate a boundary."""

    values = torch.ones(2000, dtype=torch.bfloat16)
    previous = _state_values_fingerprint({"weight": values})
    values[1000] = 2
    assert _state_values_fingerprint({"weight": values}) != previous
