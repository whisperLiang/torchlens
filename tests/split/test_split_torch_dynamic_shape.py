"""Dynamic-batch split replay tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split.errors import SplitBoundaryError


class DynamicShapeModel(nn.Module):
    """Toy model that uses common shape-changing Torch ops."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.view(batch, -1)
        flat = torch.reshape(flat, (batch, -1))
        flat = torch.flatten(flat, start_dim=1)
        return self.proj(self.relu(flat))


class AttentionLikeDynamicShapeModel(nn.Module):
    """Exercise factory, repeat, permutation, and ambiguous attention reshape rules."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        batch = x.shape[0]
        mask = torch.zeros((batch, 1, x.shape[2]), device=x.device, dtype=x.dtype)
        sequence = torch.cat((x, mask), dim=1).permute(2, 0, 1).contiguous()
        heads = sequence.view(x.shape[2], batch * 2, 2)
        repeated = x.mean(dim=1, keepdim=True).repeat(4, 1, 1)
        return heads, repeated


class CoincidentalBatchSizedReshape(nn.Module):
    """Keep a dimension fixed even when it equals the traced batch size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(2, -1)


class CoincidentalStackAndReduction(nn.Module):
    """Exercise a stack count equal to trace batch and a batch reduction."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.stack((x, x), dim=0), x.sum(dim=0)


class ScalarShapeDependencyModel(nn.Module):
    """Use scalar tensor and Python shape operations as replay dependencies."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = torch.unbind(x.new_tensor([2, 2], dtype=torch.int64))
        reshaped = x.view(x.shape[0], first, second)
        flat = reshaped.flatten(1)
        half = flat.shape[1] // 2
        left, right = torch.split(flat, (half, half), dim=1)
        return left + right


class AffineConcatAndSliceModel(nn.Module):
    """Exercise additive and ceil-div batch expressions from operation semantics."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat((x, x[:1]), dim=0), x[::2]


class InteriorBatchBranchModel(nn.Module):
    """Change a shape at the B=2 sample while preserving operation topology."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = 3 if x.shape[0] == 2 else 2
        return torch.zeros((size, x.shape[1]), device=x.device) + x.sum() * 0


class MutableCaptureStateModel(nn.Module):
    """Mutate ordinary model attributes during capture."""

    def __init__(self) -> None:
        super().__init__()
        self.history = ["initial"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.history.append(int(x.shape[0]))
        self.created_cache = {"batch": int(x.shape[0])}
        return x.reshape(x.shape[0], -1)


def test_batch_symbolic_replay_for_view_reshape_flatten() -> None:
    """A trace at batch 2 replays supported dynamic batches."""

    torch.manual_seed(0)
    model = DynamicShapeModel().eval()
    example = torch.randn(2, 3, 4)
    runtime = tl.split.prepare(
        model,
        example,
        split_request("50%"),
    )

    for batch in (1, 2, 4, 8):
        x = torch.randn(batch, 3, 4)
        assert torch.allclose(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)


def test_dynamic_shape_witness_solves_attention_view_and_preserves_state() -> None:
    """One B=2 sample disambiguates B products without mutating caller state."""

    torch.manual_seed(7)
    rng_state = torch.random.get_rng_state().clone()
    model = AttentionLikeDynamicShapeModel().eval()
    example = torch.randn(2, 3, 8)
    rng_state = torch.random.get_rng_state().clone()
    runtime = tl.split.prepare(
        model,
        example,
        split_request("50%"),
    )

    shape_program = runtime.trace_graph.shape_program
    assert shape_program.unresolved == {}
    assert runtime.traced_batch_size == 1
    assert shape_program.witness_batch_sizes == (2,)
    assert model.forward_calls == 0
    assert torch.equal(torch.random.get_rng_state(), rng_state)

    for batch in (1, 3):
        x = torch.randn(batch, 3, 8)
        actual_heads, actual_repeated = runtime.replay(x)
        expected_heads, expected_repeated = model(x)
        torch.testing.assert_close(actual_heads, expected_heads)
        torch.testing.assert_close(actual_repeated, expected_repeated)


def test_reshape_does_not_guess_from_trace_batch_divisibility() -> None:
    """A coincidental trace dimension stays constant after witness disambiguation."""

    model = CoincidentalBatchSizedReshape().eval()
    runtime = tl.split.prepare(
        model,
        torch.randn(2, 4),
        split_request("50%"),
    )

    assert runtime.trace_graph.shape_program.unresolved == {}
    assert runtime.traced_batch_size == 1
    assert runtime.trace_graph.shape_program.witness_batch_sizes == (2,)
    reshape_node = next(
        node for node in runtime.trace_graph.compute_nodes if node.op_type == "reshape"
    )
    assert runtime.trace_graph.shape_program.proof_sources[reshape_node.canonical_id] == (
        "sampled_shape_witness"
    )
    assert runtime.trace_graph.shape_program.witness_axis_diagnostics[
        reshape_node.canonical_id
    ] == {
        "candidate_axes": (0, 1),
        "accepted_axes": (1,),
        "eliminated_axes": (0,),
    }
    for batch in (1, 3):
        value = torch.randn(batch, 4)
        actual = runtime.replay(value)
        assert actual.shape == (2, batch * 2)
        torch.testing.assert_close(actual, model(value))


def test_rank_change_and_reduction_use_witnesses_without_axis_guessing() -> None:
    """Stack and reduction distinguish fixed trace coincidences from batch axes."""

    model = CoincidentalStackAndReduction().eval()
    runtime = tl.split.prepare(
        model,
        torch.randn(2, 4),
        split_request("50%"),
    )

    shape_program = runtime.trace_graph.shape_program
    assert shape_program.unresolved == {}
    assert runtime.traced_batch_size == 1
    assert shape_program.witness_batch_sizes == (2,)
    stack_node = next(node for node in runtime.trace_graph.compute_nodes if node.op_type == "stack")
    stack_diagnostics = shape_program.witness_axis_diagnostics[stack_node.canonical_id]
    assert stack_diagnostics["accepted_axes"] == (1,)
    assert 0 in stack_diagnostics["eliminated_axes"]
    for batch in (1, 3):
        value = torch.randn(batch, 4)
        actual_stack, actual_sum = runtime.replay(value)
        expected_stack, expected_sum = model(value)
        torch.testing.assert_close(actual_stack, expected_stack)
        torch.testing.assert_close(actual_sum, expected_sum)


def test_scalar_shape_refs_are_frontier_dependencies_at_every_boundary() -> None:
    """Scalar view/split arguments cross boundaries by canonical value identity."""

    model = ScalarShapeDependencyModel().eval()
    trace_input = torch.randn(2, 4)
    seed_runtime = tl.split.prepare(
        model,
        trace_input,
        split_request("50%"),
    )

    for node in seed_runtime.trace_graph.compute_nodes:
        for point in (tl.split.before(node.canonical_id), tl.split.after(node.canonical_id)):
            runtime = seed_runtime.at(point)
            for batch in (1, 3):
                value = torch.randn(batch, 4)
                torch.testing.assert_close(runtime.replay(value), model(value))


def test_concat_and_slice_compile_additive_and_ceildiv_batch_expressions() -> None:
    """Semantic rules represent B+C and ceildiv(B,k) without proportional guessing."""

    model = AffineConcatAndSliceModel().eval()
    runtime = tl.split.prepare(
        model,
        torch.randn(2, 4),
        split_request("50%"),
    )

    shape_program = runtime.trace_graph.shape_program
    assert shape_program.unresolved == {}
    for batch in range(1, 6):
        value = torch.randn(batch, 4)
        actual_concat, actual_slice = runtime.replay(value)
        expected_concat, expected_slice = model(value)
        torch.testing.assert_close(actual_concat, expected_concat)
        torch.testing.assert_close(actual_slice, expected_slice)


@pytest.mark.parametrize("segment", ("prefix", "suffix"))
def test_unresolved_shape_allows_captured_batch_and_refuses_changed_batch(segment: str) -> None:
    """A failed shape probe restricts changed batches across all executing segments."""

    model = InteriorBatchBranchModel().eval()
    seed_runtime = tl.split.prepare(model, torch.randn(2, 4), split_request("50%"))
    shape_program = seed_runtime.trace_graph.shape_program
    assert shape_program is not None
    assert shape_program.unresolved
    unresolved_id = next(iter(shape_program.unresolved))
    point = tl.split.after(unresolved_id) if segment == "prefix" else tl.split.before(unresolved_id)
    runtime = seed_runtime.at(point)

    assert runtime.capability_report.preflight_ok
    assert runtime.capability_report.replay.supported
    diagnostics = runtime.explain_capabilities()["shape_diagnostics"]
    assert diagnostics["unresolved"] == shape_program.unresolved
    assert diagnostics["traced_batch_size"] == runtime.traced_batch_size == 1
    captured_input = torch.randn(runtime.traced_batch_size, 4)
    torch.testing.assert_close(runtime.replay(captured_input), model(captured_input))

    for batch in (2, 3):
        value = torch.randn(batch, 4)
        with pytest.raises(SplitBoundaryError, match="probe did not pass") as exc_info:
            runtime.run_prefix(value)
        assert exc_info.value.context.reason == "batch probe did not pass"

    torch.testing.assert_close(runtime.replay(captured_input), model(captured_input))


def test_capture_transaction_restores_mutable_and_new_plain_attributes() -> None:
    """Preparing a runtime does not retain ordinary model-state mutations."""

    model = MutableCaptureStateModel().eval()
    runtime = tl.split.prepare(
        model,
        torch.randn(2, 4),
        split_request("50%"),
    )

    assert runtime.trace_graph.shape_program.unresolved == {}
    assert model.history == ["initial"]
    assert not hasattr(model, "created_cache")
