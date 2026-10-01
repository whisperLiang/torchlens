"""Owned split state survives placement changes and structural boundary reuse."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan
from torchlens.split.boundary import ReplayBoundary
from torchlens.split.cache import _cache_secret
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError
from torchlens.split.state import SegmentState
from torchlens.user_funcs import _store_authenticated_capture_cache


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
def test_owned_suffix_updates_allow_direct_and_cached_boundaries(
    training_boundary: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suffix update can consume an old boundary with the same structural ABI."""

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
    assert "state_fingerprint" not in cached.metadata
    assert "state_prefix_kind" not in cached.metadata
    previous = runtime.run_suffix(cached).detach().clone()

    parameters = runtime.suffix_parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    for value in parameters:
        value.grad = torch.ones_like(value)
    optimizer.step()
    expected = runtime.run_suffix(runtime.run_prefix(inputs))
    assert not torch.equal(expected, previous)
    for old in (boundary, cached):
        torch.testing.assert_close(runtime.run_suffix(old), expected)
        runtime.train_suffix(old, torch.zeros(3, 3))


def test_cached_training_boundary_keeps_backward_metadata_without_state_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached training boundaries retain their graph ABI after dropping backward state."""

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
    assert "state_fingerprint" not in training.metadata
    assert "state_prefix_kind" not in training.metadata
    assert "state_fingerprint" not in inference.metadata
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
    analysis = runtime.analyze(tl.split.after("tail"))
    target_node_id = analysis.plan.target_node_id
    if divergent:
        with torch.no_grad():
            runtime.prefix_parameters()[0].add_(1)
    candidate = next(
        candidate
        for candidate in runtime.split_points(diagnose=True).candidates
        if candidate.kind == "after" and candidate.node_id == target_node_id
    )
    assert candidate.replay_supported is not divergent
    if divergent:
        assert "divergent replicas" in candidate.unsupported_reason
        with pytest.raises(SplitUnsupportedError, match="divergent replicas"):
            runtime.analyze(tl.split.after("tail"))
    expected = runtime.replay(inputs)
    same_cut = runtime.at(runtime.request.point)
    torch.testing.assert_close(same_cut.replay(inputs), expected)
    if divergent:
        with pytest.raises(SplitUnsupportedError, match="divergent replicas"):
            runtime.at(tl.split.after("tail"))
        with pytest.raises(SplitUnsupportedError, match="divergent replicas"):
            runtime.materialize(analysis)
    else:
        recut = runtime.at(tl.split.after("tail"))
        assert len(recut.prefix_parameters()) == 1
        torch.testing.assert_close(recut.replay(inputs), expected)


def test_boundary_metadata_discards_legacy_state_fields(tmp_path: Path) -> None:
    """Legacy cached metadata cannot impose value-state compatibility checks."""

    model = StateMigrationMlp().eval()
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(model, inputs, split_request("after:relu"))
    boundary = runtime.run_prefix(inputs)
    assert "state_fingerprint" not in boundary.metadata
    assert "state_prefix_kind" not in boundary.metadata

    old_metadata = dict(boundary.metadata)
    old_metadata.update(state_fingerprint="obsolete", state_prefix_kind="inference")
    legacy = ReplayBoundary(boundary.backend, boundary.tensors, boundary.spec, old_metadata)
    assert "state_fingerprint" not in legacy.metadata
    assert "state_prefix_kind" not in legacy.metadata
    torch.testing.assert_close(runtime.run_suffix(legacy), runtime.run_suffix(boundary))

    # Simulate an authenticated pickle produced by an older release, whose
    # unpickler does not call the dataclass constructor.
    object.__setattr__(legacy, "metadata", old_metadata)
    cache_path = tmp_path / "old-boundary"
    runtime.save_boundary(boundary, cache_path)
    assert _store_authenticated_capture_cache(legacy, cache_path / "payload.pkl", _cache_secret())
    cached = runtime.load_boundary(cache_path)
    assert "state_fingerprint" not in cached.metadata
    assert "state_prefix_kind" not in cached.metadata


def test_replay_boundary_still_refuses_structural_mismatches() -> None:
    """Removing value checks leaves split, graph, dtype, and shape checks intact."""

    runtime = tl.split.prepare(
        StateMigrationMlp().eval(), torch.ones(3, 4), split_request("after:relu")
    )
    boundary = runtime.run_prefix(torch.ones(3, 4))
    for field, replacement in (("split_id", "other"), ("graph_shape_hash", "other")):
        bad_metadata = dict(boundary.metadata)
        bad_metadata[field] = replacement
        with pytest.raises(SplitBoundaryError):
            runtime.run_suffix(
                ReplayBoundary(boundary.backend, boundary.tensors, boundary.spec, bad_metadata)
            )
    key = next(iter(boundary.tensors))
    bad_tensors = dict(boundary.tensors)
    bad_tensors[key] = bad_tensors[key].to(torch.float64)
    with pytest.raises(SplitBoundaryError, match="dtype"):
        runtime.run_suffix(
            ReplayBoundary(boundary.backend, bad_tensors, boundary.spec, dict(boundary.metadata))
        )
    bad_tensors[key] = boundary.tensors[key][:1]
    with pytest.raises(SplitBoundaryError, match="shape"):
        runtime.run_suffix(
            ReplayBoundary(boundary.backend, bad_tensors, boundary.spec, dict(boundary.metadata))
        )


def test_explicit_replay_does_not_read_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit prefix and suffix calls rely on the structural boundary ABI."""

    model = StateMigrationMlp().eval()
    inputs = torch.ones(3, 4)
    runtime = tl.split.prepare(model, inputs, split_request("after:relu"))

    def unexpected_state_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("split replay must not scan model state")

    monkeypatch.setattr(model, "state_dict", unexpected_state_read)
    monkeypatch.setattr(runtime.segments.prefix, "bound_state_values", unexpected_state_read)
    monkeypatch.setattr(runtime.segments.suffix, "bound_state_values", unexpected_state_read)
    boundary = runtime.run_prefix(inputs)
    torch.testing.assert_close(runtime.run_suffix(boundary), runtime.replay(inputs))
