"""Value-proof regressions reduced from scheduled real-model failures."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from validation_goldens.test_validation_exemption_hardening import (
    _first_output,
    _install_constant_replay,
)

import torchlens as tl
from torchlens.validation.diagnostics import get_validation_failure


@pytest.mark.parametrize("kind", ["shape_template", "singleton_softmax", "paired_mask"])
@pytest.mark.parametrize("broken", [False, True])
def test_shape_and_index_operands_preserve_value_tripwires(kind: str, broken: bool) -> None:
    """Structural operands pass without excusing corrupted data-value replay."""

    class Model(nn.Module):
        """Reduce LeViT, Informer, and detector operand patterns."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply one shape/selection-dependent operation."""
            if kind == "shape_template":
                return x.reshape_as(x + 1)
            if kind == "singleton_softmax":
                return x.softmax(dim=-1)
            result = x.clone()
            mask = torch.arange(40) < 2
            columns = torch.arange(2)
            result[mask, columns] = x[:2, 0] + 1
            return result

    shape = (2, 3 if broken else 1) if kind == "singleton_softmax" else (40, 3)
    trace = tl.trace(Model(), torch.randn(shape), save_arg_values=True)
    if broken:
        _install_constant_replay(
            trace,
            {
                "shape_template": "reshape_as",
                "singleton_softmax": "softmax",
                "paired_mask": "__setitem__",
            }[kind],
        )
    assert (
        trace.validate_forward_pass([_first_output(trace)], validate_metadata=False) is not broken
    )


def test_zero_affine_batch_norm_has_value_irrelevant_input_and_stats() -> None:
    """Zero-initialized residual normalization still replays and validates."""
    model = nn.BatchNorm2d(3).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    trace = tl.trace(model, torch.randn(2, 3, 4, 4), save_arg_values=True)

    assert trace.validate_forward_pass([_first_output(trace)])
    assert (
        trace.validation_replay_status.exempted_reason_counts["multiplicative_zero_annihilator"]
        >= 1
    )


@pytest.mark.parametrize("constant_values", [False, True])
@pytest.mark.parametrize("proof", ["constant_values", "zero_query", "zero_key"])
def test_attention_operand_irrelevance_requires_its_saved_proof(
    constant_values: bool, proof: str
) -> None:
    """Freezing one attention operand requires its independent saved-value proof."""

    class Attention(nn.Module):
        """Keep Q/K/V as three independent parents."""

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Compute one unmasked attention call without dropout."""
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q, k, v = (torch.randn(1, 2, 3, 4) for _ in range(3))
    if constant_values and proof == "constant_values":
        v = v[..., :1, :].expand_as(v).clone()
    elif constant_values and proof == "zero_query":
        q.zero_()
    elif constant_values and proof == "zero_key":
        k.zero_()
    trace = tl.trace(Attention(), (q, k, v), save_arg_values=True)
    op = next(op for op in trace.layer_list if op.func_name == "scaled_dot_product_attention")
    original = op.func
    saved_q, saved_k = op.saved_args[0].clone(), op.saved_args[1].clone()

    def frozen_operand(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Drop only Q or K, leaving both other value dependencies live."""
        return original(q, saved_k, v) if proof == "zero_query" else original(saved_q, k, v)

    op.func = frozen_operand
    assert (
        trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)
        is constant_values
    )


@pytest.mark.parametrize("kind", ["attention", "masked_scatter", "scatter_reduce", "index"])
@pytest.mark.parametrize("broken", [False, True])
def test_operand_irrelevance_requires_the_saved_value_proof(kind: str, broken: bool) -> None:
    """Real constant operands pass; frozen value-sensitive replays still fail."""

    class Model(nn.Module):
        """Exercise the same operand boundaries as the real-model regressions."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply one operation with a proved or value-sensitive operand."""
            if kind == "attention":
                value = x + 2 if broken else torch.zeros_like(x)
                return torch.nn.functional.scaled_dot_product_attention(x, x + 1, value)
            if kind == "masked_scatter":
                mask = torch.ones_like(x, dtype=torch.bool)
                if broken:
                    mask[0] = False
                return x.masked_scatter(mask, x + 2)
            if kind == "scatter_reduce":
                index = torch.arange(x.shape[0]).reshape(-1, 1).expand_as(x)
                return x.scatter_reduce(0, index, x + 2, reduce="sum", include_self=broken)
            result = x.clone()
            index = torch.zeros(1, dtype=torch.long)
            result[index] = x[:1] + 2
            return result

    shape = (1, 1, 3, 4) if kind == "attention" else (3 if broken else 1, 2)
    trace = tl.trace(Model(), torch.randn(shape), save_arg_values=True)
    if broken:
        _install_constant_replay(
            trace,
            {
                "attention": "scaled_dot_product_attention",
                "masked_scatter": "masked_scatter",
                "scatter_reduce": "scatter_reduce",
                "index": "__setitem__",
            }[kind],
        )
    assert (
        trace.validate_forward_pass([_first_output(trace)], validate_metadata=False) is not broken
    )


@pytest.mark.parametrize("broken", [False, True])
def test_boolean_or_requires_an_independent_true_cooperand(broken: bool) -> None:
    """An all-true other operand absorbs OR; an influential frozen OR fails."""

    class Model(nn.Module):
        """Keep the absorbing value outside the perturbed parent."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Combine the input with a constant boolean operand."""
            other = torch.zeros_like(x) if broken else torch.ones_like(x)
            return x | other

    trace = tl.trace(Model(), torch.tensor([False, True]), save_arg_values=True)
    if broken:
        _install_constant_replay(trace, "__or__")
    assert (
        trace.validate_forward_pass([_first_output(trace)], validate_metadata=False) is not broken
    )


def test_influential_batch_norm_parent_cannot_use_zero_affine_exemption() -> None:
    """A constant replay of ordinary BatchNorm must still fail perturbation."""
    trace = tl.trace(nn.BatchNorm2d(3).eval(), torch.randn(2, 3, 4, 4), save_arg_values=True)
    _install_constant_replay(trace, "batch_norm")

    assert not trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)
    assert get_validation_failure(trace).check == "perturbation"


@pytest.mark.parametrize("empty", [True, False])
def test_empty_broadcast_product_has_no_values_to_perturb(empty: bool) -> None:
    """An empty proposal tensor stays empty when its scalar scale changes."""

    class EmptyProduct(nn.Module):
        """Keep the value-sensitive scalar and the empty tensor in the graph."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Multiply an empty slice by a scalar derived from the input."""
            return x[: 0 if empty else 1] * x.sum()

    trace = tl.trace(EmptyProduct(), torch.randn(2, 3), save_arg_values=True)
    if not empty:
        _install_constant_replay(trace, "__mul__")
    assert trace.validate_forward_pass([_first_output(trace)]) is empty


@pytest.mark.parametrize("empty", [True, False])
def test_empty_rois_select_no_feature_values(empty: bool) -> None:
    """An empty ROI list cannot transmit feature values into its empty result."""
    torchvision = pytest.importorskip("torchvision")

    class EmptyRois(nn.Module):
        """Apply the native ROI operator to a real, empty coordinate buffer."""

        def __init__(self) -> None:
            """Create valid empty packed ROI coordinates."""
            super().__init__()
            rois = torch.empty(0, 5) if empty else torch.tensor([[0.0, 0.0, 0.0, 4.0, 4.0]])
            self.register_buffer("rois", rois)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pool no regions from the input feature map."""
            return torchvision.ops.roi_align(x, self.rois, (2, 2), sampling_ratio=2)

    trace = tl.trace(EmptyRois(), torch.randn(1, 3, 8, 8), save_arg_values=True)
    if not empty:
        _install_constant_replay(trace, "roi_align")
    assert trace.validate_forward_pass([_first_output(trace)]) is empty


def test_broadcast_index_full_overwrite_is_proved_by_destination_positions() -> None:
    """Repeated column indices across batches can still address every cell once."""

    class BroadcastWrite(nn.Module):
        """Use Informer's per-batch permutation assignment."""

        def forward(self, x: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
            """Overwrite all rows using distinct broadcasted coordinate pairs."""
            out = x.clone()
            rows = torch.arange(2).unsqueeze(1)
            columns = torch.tensor([[2, 0, 1], [1, 2, 0]])
            out[rows, columns] = replacement
            return out

    x, replacement = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    trace = tl.trace(BroadcastWrite(), [x, replacement], save_arg_values=True)
    assert trace.validate_forward_pass([_first_output(trace)])


@pytest.mark.parametrize(("upper", "broken"), [(6, False), (6, True), (5, False)])
def test_saturated_clamp_requires_a_boundary_crossing_perturbation(
    upper: int, broken: bool
) -> None:
    """Cross a real saturation boundary; a frozen replay must still fail."""

    class SaturatedClamp(nn.Module):
        """Clamp well beyond the input's local perturbation range."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Map values below five to the lower saturation boundary."""
            return torch.clamp(x, min=5, max=upper)

    trace = tl.trace(SaturatedClamp(), torch.zeros(2, 3), save_arg_values=True)
    if broken:
        _install_constant_replay(trace, "clamp")
    assert trace.validate_forward_pass([_first_output(trace)]) is not broken


@pytest.mark.parametrize("kind", ["addcmul", "einsum", "and", "conv"])
@pytest.mark.parametrize("broken", [False, True])
def test_zero_cooperand_annihilates_fused_and_contraction_inputs(kind: str, broken: bool) -> None:
    """Zero LayerScale weights and boolean masks are exact value proofs."""

    class ZeroCooperand(nn.Module):
        """Keep the all-zero cooperand separate from the perturbed input."""

        def __init__(self) -> None:
            """Create the captured, non-perturbed zero buffer."""
            super().__init__()
            shape = {"einsum": (3, 4), "conv": (1, 3, 4)}.get(kind, (2, 3))
            self.register_buffer("zero", torch.zeros(shape))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the selected operation with the zero cooperand."""
            if kind == "addcmul":
                return torch.addcmul(self.zero + 1, self.zero, x)
            if kind == "einsum":
                return torch.einsum("ij,jk->ik", x, self.zero)
            if kind == "conv":
                return nn.functional.conv1d(self.zero, x.unsqueeze(-1))
            return x.bool() & self.zero.bool()

    trace = tl.trace(ZeroCooperand(), torch.randn(2, 3), save_arg_values=True)
    if broken:
        _install_constant_replay(trace, {"and": "__and__", "conv": "conv1d"}.get(kind, kind))
    assert trace.validate_forward_pass([_first_output(trace)]) is not broken
    if not broken:
        assert "multiplicative_zero_annihilator" in (
            trace.validation_replay_status.exempted_reason_counts
        )


@pytest.mark.parametrize("kind", ["mul", "and"])
def test_repeated_zero_parent_is_not_a_nonperturbed_annihilator(kind: str) -> None:
    """A broken replay of x*x or x&x must fail even when saved x is zero."""

    class RepeatedParent(nn.Module):
        """Use the same parent in both operand positions."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the selected product to the same input twice."""
            return x * x if kind == "mul" else x & x

    dtype = torch.float32 if kind == "mul" else torch.bool
    trace = tl.trace(RepeatedParent(), torch.zeros(2, 3, dtype=dtype), save_arg_values=True)
    _install_constant_replay(trace, "__mul__" if kind == "mul" else "__and__")
    assert not trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)
    assert get_validation_failure(trace).check == "perturbation"
