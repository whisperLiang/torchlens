"""Batch-polymorphic split replay and split-training tests.

The batches 1, 2, 3, 8, 32 appear ONLY here.  They are not a production
whitelist: the runtime accepts any positive compatible batch.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan
from torchlens.split.errors import SplitUnsupportedError


class TinyMlp(nn.Module):
    """Small residual-friendly MLP used as the batch-polymorphism fixture."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _params(module: nn.Module) -> list[torch.Tensor]:
    """Return detached parameter clones."""

    return [param.detach().clone() for param in module.parameters()]


def test_single_trace_replays_and_trains_across_runtime_batches() -> None:
    """One canonical capture is reused for replay and training at many batches.

    The runtime batch matrix lives only in this test.  Production code has no
    knowledge of these five values.
    """

    torch.manual_seed(0)
    model = TinyMlp()
    example = torch.randn(2, 4)
    runtime = tl.split.prepare(model, example, split_request("after:relu", trainable=True))

    # One canonical capture at a small batch serves every runtime batch below.
    assert runtime.traced_batch_size in {1, 2}
    graph_id = runtime.graph_identity
    split_id = runtime.split_id
    plan_id = id(runtime.plan)
    capture_id = id(runtime.trace)
    graph_object_id = id(runtime.trace_graph)
    shape_fingerprint = runtime.trace_graph.shape_program.fingerprint
    assert graph_id is not None

    for batch in (1, 2, 3, 8, 32):
        x = torch.randn(batch, 4)
        y = torch.randn(batch, 3)

        with torch.no_grad():
            replayed = runtime.replay(x)
            full = model(x)
        assert replayed.shape == full.shape == (batch, 3)
        torch.testing.assert_close(replayed, full, atol=1e-5, rtol=1e-4)

        boundary = runtime.run_training_prefix(x)
        loss, grads = runtime.train_suffix(boundary, y)
        assert grads
        assert torch.isfinite(loss.detach())
        runtime.backward_prefix(boundary, grads)

        # No recapture, no graph rebuild, no SplitIR rebuild, no replan.
        assert id(runtime.trace) == capture_id
        assert id(runtime.trace_graph) == graph_object_id
        assert id(runtime.plan) == plan_id
        assert runtime.graph_identity == graph_id
        assert runtime.split_ir_identity == graph_id
        assert runtime.split_id == split_id
        assert runtime.trace_graph.shape_program.fingerprint == shape_fingerprint
        assert runtime.traced_batch_size in {1, 2}


def test_split_training_matches_full_model_training() -> None:
    """From identical initial state, split training matches a full-model step."""

    torch.manual_seed(1)
    model = TinyMlp()
    split_model = copy.deepcopy(model)
    x = torch.randn(8, 4)
    y = torch.randn(8, 3)
    runtime = tl.split.prepare(split_model, x, split_request("after:relu", trainable=True))
    boundary = runtime.run_training_prefix(x)
    suffix_opt = torch.optim.SGD(split_model.fc2.parameters(), lr=0.05)
    prefix_opt = torch.optim.SGD(split_model.fc1.parameters(), lr=0.05)
    full_opt = torch.optim.SGD(model.parameters(), lr=0.05)

    full_opt.zero_grad(set_to_none=True)
    full_loss = torch.nn.functional.mse_loss(model(x), y)
    full_loss.backward()
    full_opt.step()
    loss, grads = runtime.train_suffix(boundary, y, optimizer=suffix_opt)
    runtime.backward_prefix(boundary, grads, optimizer=prefix_opt)

    torch.testing.assert_close(loss.detach(), full_loss.detach(), atol=1e-5, rtol=1e-4)
    for left, right in zip(_params(split_model), _params(model), strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)


def test_canonical_capture_prefers_batch_one() -> None:
    """A large example batch is rebatched to canonical B=1, not recaptured at 32."""

    torch.manual_seed(0)
    model = TinyMlp().eval()
    example = torch.randn(32, 4)
    runtime = tl.split.prepare(model, example, split_request("after:relu"))

    assert runtime.traced_batch_size == 1
    replayed = runtime.replay(torch.randn(32, 4))
    assert replayed.shape == (32, 3)


def test_untested_batch_is_not_range_rejected() -> None:
    """A compatible batch that no test used as a range bound is accepted."""

    model = TinyMlp().eval()
    runtime = tl.split.prepare(model, torch.randn(2, 4), split_request("after:relu"))
    x = torch.randn(7, 4)
    with torch.no_grad():
        torch.testing.assert_close(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)


def test_split_points_enumerates_every_compute_boundary() -> None:
    """Every before/after compute cut is discoverable with an explicit reason."""

    model = TinyMlp().eval()
    runtime = tl.split.prepare(model, torch.randn(2, 4), split_request("after:relu"))
    report = runtime.split_points()

    assert report.total == 2 * len(runtime.trace_graph.compute_nodes)
    assert report.total > 0
    assert len(report.supported) + len(report.unsupported) == report.total
    for candidate in report.unsupported:
        assert candidate.unsupported_reason is not None
    reused = runtime.at(report.supported[0].point)
    assert not reused.retains_trace and not runtime.retains_trace
    assert reused.trace_graph is runtime.trace_graph
    assert reused.graph_identity == runtime.graph_identity


def test_every_valid_boundary_replays_across_batches_from_one_capture() -> None:
    """Exhaustive: every enumerated boundary replays, or names its refusal."""

    torch.manual_seed(0)
    model = TinyMlp().eval()
    seed = tl.split.prepare(model, torch.randn(2, 4), split_request("after:relu"))
    report = seed.split_points()
    assert report.total > 0
    assert report.supported, report.reasons()

    for candidate in report.supported:
        runtime = seed.at(candidate.point)
        assert not runtime.retains_trace and not seed.retains_trace
        assert runtime.trace_graph is seed.trace_graph
        for batch in (1, 3, 8):
            x = torch.randn(batch, 4)
            with torch.no_grad():
                torch.testing.assert_close(
                    runtime.replay(x),
                    model(x),
                    atol=1e-5,
                    rtol=1e-4,
                )

    # Nothing is silently skipped: refusals are enumerated with reasons.
    for candidate in report.unsupported:
        assert candidate.unsupported_reasons


def test_residual_and_multi_output_frontiers_cross_every_live_tensor() -> None:
    """A skip connection puts more than one live tensor on the frontier."""

    class Residual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc1(x))
            return self.fc2(h) + h

    model = Residual().eval()
    seed = tl.split.prepare(model, torch.randn(2, 4), split_request("50%"))
    report = seed.split_points()
    multi = [candidate for candidate in report.supported if len(candidate.boundary_value_ids) > 1]
    assert multi, "a residual graph must expose a multi-tensor live frontier"
    for candidate in multi[:3]:
        runtime = seed.at(candidate.point)
        for batch in (1, 3):
            x = torch.randn(batch, 4)
            with torch.no_grad():
                torch.testing.assert_close(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)


def test_boundary_schema_is_batch_neutral() -> None:
    """A boundary schema spells the batch axis symbolically, not as traced B."""

    model = TinyMlp().eval()
    runtime = tl.split.prepare(model, torch.randn(2, 4), split_request("after:relu"))
    symbol = runtime.request.batch_symbol
    schemas = [item for item in runtime.boundary_schema if item.shape is not None]
    assert schemas
    assert any(symbol in tuple(item.shape.as_tuple()) for item in schemas), [
        item.shape.as_tuple() for item in schemas
    ]

    for batch in (1, 8, 32):
        boundary = runtime.run_prefix(torch.randn(batch, 4))
        assert boundary.metadata["runtime_batch_size"] == batch
        # The same semantic schema serves every concrete batch.
        assert set(boundary.spec) == set(runtime.boundary_spec)


def test_cpu_prefix_cuda_suffix_replay_and_training() -> None:
    """PlacementPlan moves prefix and suffix independently when CUDA is present."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for heterogeneous placement.")
    torch.manual_seed(0)
    model = TinyMlp().eval()
    example = torch.randn(2, 4)
    runtime = tl.split.prepare(model, example, split_request("after:relu", trainable=True))
    placed = runtime.with_placement(PlacementPlan.across("cpu", "cuda:0"))
    x = torch.randn(3, 4)
    with torch.no_grad():
        replayed = placed.replay(x)
        full = model(x)
    torch.testing.assert_close(replayed.cpu(), full, atol=1e-4, rtol=1e-3)

    train_model = TinyMlp()
    train_runtime = tl.split.prepare(
        train_model,
        example,
        split_request(
            "after:relu",
            trainable=True,
            placement=PlacementPlan.across("cpu", "cuda:0"),
        ),
    )
    boundary = train_runtime.run_training_prefix(x)
    y = torch.randn(3, 3, device="cuda:0")
    loss, grads = train_runtime.train_suffix(boundary, y)
    assert torch.isfinite(loss.detach().cpu())
    train_runtime.backward_prefix(boundary, grads)


def test_explicit_placement_on_unsupported_backend_fails_closed() -> None:
    """A backend that cannot place state refuses an explicit PlacementPlan."""

    from torchlens.split.adapters._unsupported import UnsupportedSplitAdapter
    from torchlens.split.placement import require_placement_support

    adapter = UnsupportedSplitAdapter("mlx")
    with pytest.raises(SplitUnsupportedError, match="cannot place"):
        require_placement_support(
            adapter,
            PlacementPlan.across("cpu", "cuda:0"),
            split_point="after:relu",
        )
