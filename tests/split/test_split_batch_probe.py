"""One-sample batch extrapolation and honest refusal/disclosure contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan, pipeline
from torchlens.split.batching import witness_probe_sizes
from torchlens.split.errors import SplitBoundaryError


class ThresholdModel(torch.nn.Module):
    """Keep tensor topology equal while changing a scalar on a Python branch."""

    def __init__(self, threshold: int = 2) -> None:
        """Set a branch threshold and trainable prefix."""

        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch only on batch metadata."""

        hidden = torch.relu(x * self.weight)
        return hidden * (3.0 if x.shape[0] >= self.threshold else 2.0)


def test_only_two_capture_batches_and_no_runtime_recapture(monkeypatch: pytest.MonkeyPatch) -> None:
    """B=1 plus one B=2 capture license disclosed empirical extrapolation."""

    captures: list[int] = []
    original = pipeline.capture_model

    def count_capture(model: Any, inputs: tuple[Any, ...], spec: Any, **kwargs: Any) -> Any:
        """Observe capture invocations without relying on restored model counters."""

        captures.append(int(inputs[0].shape[0]))
        return original(model, inputs, spec, **kwargs)

    monkeypatch.setattr(pipeline, "capture_model", count_capture)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    runtime = tl.split.prepare(model, torch.ones(32, 4), split_request("after:relu"))
    assert captures == [1, 2]
    assert witness_probe_sizes(1) == (2,)
    assert witness_probe_sizes(2) == ()
    report = runtime.batch_validation
    assert report["status"] == "passed"
    assert report["mode"] == "sampled_extrapolation"
    assert report["universal_proof"] is False
    assert runtime.explain_capabilities()["shape_diagnostics"]["batch_validation"] == report
    identity = runtime.graph_identity
    for batch, status in ((1, "captured"), (2, "sampled"), (8, "extrapolated")):
        x = torch.ones(batch, 4)
        boundary = runtime.run_prefix(x)
        assert boundary.metadata["runtime_batch_validation"] == status
        torch.testing.assert_close(runtime.run_suffix(boundary), model(x))
    assert runtime.graph_identity == identity
    assert captures == [1, 2]


@pytest.mark.parametrize("validation", ["strict", "permissive"])
def test_same_topology_numeric_failure_restricts_all_entrypoints(
    validation: str, tmp_path: Path
) -> None:
    """A scalar branch missed by topology checks must fail the numeric B=2 check."""

    model = ThresholdModel()
    request = split_request("after:relu", trainable=True)
    if validation != "strict":
        request = replace(request, validation="permissive")
    runtime = tl.split.prepare(model, torch.ones(32, 4), request)
    assert runtime.batch_validation["status"] == "failed"
    assert "numeric mismatch" in runtime.batch_validation["reason"]
    torch.testing.assert_close(runtime.replay(torch.ones(1, 4)), model(torch.ones(1, 4)))
    for replay in (runtime.run_prefix, runtime.run_training_prefix, runtime.replay):
        with pytest.raises(SplitBoundaryError, match="probe did not pass"):
            replay(torch.ones(2, 4))
    for derived in (runtime.at(tl.split.before("relu")), runtime.with_placement(PlacementPlan())):
        assert derived.batch_validation == runtime.batch_validation
        with pytest.raises(SplitBoundaryError, match="probe did not pass"):
            derived.replay(torch.ones(8, 4))

    boundary = runtime.run_training_prefix(torch.ones(1, 4))
    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    assert loaded.metadata["batch_validation"] == runtime.batch_validation
    assert loaded.metadata["runtime_batch_validation"] == "captured"
    forged = replace(loaded, metadata={**loaded.metadata, "runtime_batch_size": 2})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.ones_like(model.weight)
    before = model.weight.detach().clone()
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        runtime.run_suffix(forged)
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        runtime.train_suffix(forged, torch.zeros(2, 4), optimizer=optimizer)
    torch.testing.assert_close(model.weight, before)
    torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))


@pytest.mark.parametrize("kind", ["exception", "topology", "container", "dtype", "shape"])
def test_probe_failure_preserves_captured_batch(kind: str) -> None:
    """Probe failures are diagnostics, not unconditional strict prepare failures."""

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> Any:
            """Change one output property only in the probe."""

            out = torch.relu(x)
            if x.shape[0] == 2:
                if kind == "exception":
                    raise ValueError("B=2 unavailable")
                if kind == "topology":
                    return torch.sigmoid(out)
                if kind == "container":
                    return (out,)
                if kind == "dtype":
                    return out.double()
                if kind == "shape":
                    return out[:1]
            return out

    model = Model()
    runtime = tl.split.prepare(model, torch.ones(4, 4), split_request("after:relu"))
    assert runtime.batch_validation["status"] == "failed"
    assert runtime.batch_validation["reason"]
    torch.testing.assert_close(runtime.replay(torch.ones(1, 4)), model(torch.ones(1, 4)))
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        runtime.replay(torch.ones(3, 4))


def test_later_branch_is_explicitly_an_unverified_extrapolation() -> None:
    """The chosen heuristic deliberately cannot detect a branch starting at B=8."""

    model = ThresholdModel(threshold=8)
    runtime = tl.split.prepare(model, torch.ones(32, 4), split_request("after:relu"))
    assert runtime.batch_validation["status"] == "passed"
    boundary = runtime.run_prefix(torch.ones(8, 4))
    assert boundary.metadata["runtime_batch_validation"] == "extrapolated"
    assert boundary.metadata["batch_validation"]["universal_proof"] is False
    assert not torch.equal(runtime.run_suffix(boundary), model(torch.ones(8, 4)))


def test_unavailable_isolated_probe_keeps_only_captured_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that cannot be isolated is not probed through the live instance."""

    from torchlens import _capture_state_helpers

    def cannot_copy(model: Any) -> tuple[Any, None, bool]:
        """Simulate the capture helper's live-model fallback."""

        return model, None, False

    monkeypatch.setattr(_capture_state_helpers, "_model_for_validation_replay", cannot_copy)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    runtime = tl.split.prepare(model, torch.ones(32, 4), split_request("after:relu"))
    assert runtime.batch_validation["status"] == "unavailable"
    assert "Cannot copy the model" in runtime.batch_validation["reason"]
    torch.testing.assert_close(runtime.replay(torch.ones(1, 4)), model(torch.ones(1, 4)))
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        runtime.replay(torch.ones(2, 4))


def test_fallback_capture_at_two_has_no_additional_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine B=1 failure retains B=2 without inventing a B=3 probe."""

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Reject singleton input."""

            if x.shape[0] == 1:
                raise ValueError("need multiple rows")
            return torch.relu(x) * 2

    seen: list[int] = []
    original = pipeline.capture_model

    def capture(model: Any, inputs: tuple[Any, ...], spec: Any, **kwargs: Any) -> Any:
        """Record native capture batches."""

        seen.append(int(inputs[0].shape[0]))
        return original(model, inputs, spec, **kwargs)

    monkeypatch.setattr(pipeline, "capture_model", capture)
    model = Model()
    runtime = tl.split.prepare(model, torch.ones(32, 4), split_request("after:relu"))
    assert seen == [1, 2]
    assert runtime.traced_batch_size == 2
    assert runtime.batch_validation["mode"] == "captured_only"
    assert runtime.batch_validation["status"] == "unavailable"
    assert runtime.batch_validation["probe_batch_size"] is None
    torch.testing.assert_close(runtime.replay(torch.ones(2, 4)), model(torch.ones(2, 4)))
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        runtime.replay(torch.ones(8, 4))


@pytest.mark.parametrize("kind", ["shape", "dtype", "container", "integer"])
def test_output_comparison_is_not_broadcasting_allclose(kind: str) -> None:
    """Exact structure/shape/dtype and integer values precede floating tolerance."""

    from torchlens.split.adapters.torch import TorchSplitAdapter

    expected: Any = torch.ones(2, 4)
    actual: Any = expected
    if kind == "shape":
        actual = torch.ones(1, 4)
    elif kind == "dtype":
        actual = expected.double()
    elif kind == "container":
        actual = [expected]
        expected = (expected,)
    else:
        expected = torch.tensor([100000], dtype=torch.int64)
        actual = expected + 1
    assert pipeline._probe_output_mismatch(TorchSplitAdapter(), expected, actual) is not None
