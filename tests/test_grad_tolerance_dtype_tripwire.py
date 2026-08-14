"""Dtype-aware gradient-validation tolerances (tripwire strengthening).

The legacy module constants ``PARAM_GRAD_VALIDATION_*`` /
``LAYER_GRAD_VALIDATION_*`` are the fp32 rows of an error model that was
silently applied to every dtype: fp64 gradients were checked ~4.5e11 of their
own ULPs loose (masking corruption far above fp64 round-off), while fp16
gradients were checked two orders BELOW their own eps (false-failing every
non-bitwise agreement).  ``param_grad_tolerances_for_dtype`` /
``layer_grad_tolerances_for_dtype`` derive the dtype-correct row; the fp32
row is bit-identical to the legacy constants.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.utils.tensor_utils import (
    LAYER_GRAD_VALIDATION_ATOL,
    LAYER_GRAD_VALIDATION_RTOL,
    PARAM_GRAD_VALIDATION_ATOL,
    PARAM_GRAD_VALIDATION_RTOL,
    layer_grad_tolerances_for_dtype,
    param_grad_tolerances_for_dtype,
)

pytestmark = pytest.mark.smoke


def test_fp32_rows_are_exactly_the_legacy_constants() -> None:
    """The fp32 row must not drift from the exported legacy pair."""

    assert param_grad_tolerances_for_dtype(torch.float32) == (
        PARAM_GRAD_VALIDATION_RTOL,
        PARAM_GRAD_VALIDATION_ATOL,
    )
    assert layer_grad_tolerances_for_dtype(torch.float32) == (
        LAYER_GRAD_VALIDATION_RTOL,
        LAYER_GRAD_VALIDATION_ATOL,
    )


def test_fp64_rows_are_tighter_than_fp32_by_the_eps_ratio() -> None:
    """fp64 gradients earn the fp32 ULP budget in fp64 ULPs, not fp32 decimals."""

    eps_ratio = float(torch.finfo(torch.float64).eps) / float(torch.finfo(torch.float32).eps)
    rtol, atol = param_grad_tolerances_for_dtype(torch.float64)
    assert rtol == pytest.approx(PARAM_GRAD_VALIDATION_RTOL * eps_ratio)
    assert atol == pytest.approx(rtol / 10.0)
    layer_rtol, layer_atol = layer_grad_tolerances_for_dtype(torch.float64)
    assert layer_rtol == pytest.approx(LAYER_GRAD_VALIDATION_RTOL * eps_ratio)
    assert layer_atol == pytest.approx(layer_rtol / 10.0)


def test_fp64_corruption_masked_by_legacy_constants_now_fails() -> None:
    """The exact corruption class the fp32 decimals blessed must now FAIL.

    An all-zero fp64 gradient buffer whose true values sit below the legacy
    1e-5 atol reads EQUAL under the legacy pair -- zero detection power --
    and must read UNEQUAL under the fp64 row.
    """

    true_grads = torch.full((16,), 1e-6, dtype=torch.float64)
    zeroed = torch.zeros_like(true_grads)
    assert torch.allclose(
        zeroed,
        true_grads,
        rtol=PARAM_GRAD_VALIDATION_RTOL,
        atol=PARAM_GRAD_VALIDATION_ATOL,
    ), "precondition: the legacy pair masks this corruption"
    rtol, atol = param_grad_tolerances_for_dtype(torch.float64)
    assert not torch.allclose(zeroed, true_grads, rtol=rtol, atol=atol)


def test_fp16_one_ulp_agreement_passes_and_corruption_fails() -> None:
    """fp16 rows admit storage rounding while catching sign flips and zeroing."""

    rtol, atol = param_grad_tolerances_for_dtype(torch.float16)
    eps16 = float(torch.finfo(torch.float16).eps)
    assert rtol >= eps16, "a tolerance below one fp16 ULP false-fails every replay"
    assert rtol < 1.0, "the row must still catch sign flips and zeroed buffers"
    grads = torch.ones((8,), dtype=torch.float16)
    one_ulp = torch.nextafter(grads, torch.tensor(2.0, dtype=torch.float16))
    assert torch.allclose(one_ulp, grads, rtol=rtol, atol=atol)
    assert not torch.allclose(-grads, grads, rtol=rtol, atol=atol)
    assert not torch.allclose(torch.zeros_like(grads), grads, rtol=rtol, atol=atol)


def test_bf16_rows_scale_with_bf16_eps() -> None:
    """bf16 rows follow bf16's own eps, and still catch total corruption."""

    rtol, atol = layer_grad_tolerances_for_dtype(torch.bfloat16)
    eps = float(torch.finfo(torch.bfloat16).eps)
    assert rtol >= eps
    assert rtol < 1.0
    grads = torch.full((8,), 1e-2, dtype=torch.bfloat16)
    assert not torch.allclose(torch.zeros_like(grads), grads, rtol=rtol, atol=atol)


def test_layer_rows_are_tighter_than_param_rows_per_dtype() -> None:
    """Elementwise (layer) comparisons keep their 10x-tighter budget per dtype."""

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        param_rtol, _ = param_grad_tolerances_for_dtype(dtype)
        layer_rtol, _ = layer_grad_tolerances_for_dtype(dtype)
        assert layer_rtol < param_rtol


def test_complex_dtypes_use_component_precision() -> None:
    """Complex dtypes derive from their component real dtype's finfo."""

    assert param_grad_tolerances_for_dtype(torch.complex64) == param_grad_tolerances_for_dtype(
        torch.float32
    )
    assert param_grad_tolerances_for_dtype(torch.complex128) == param_grad_tolerances_for_dtype(
        torch.float64
    )


def test_non_float_dtype_stays_strict() -> None:
    """A misrouted non-float dtype gets the strictest row, never a loose one."""

    rtol, _ = param_grad_tolerances_for_dtype(torch.int64)
    fp64_rtol, _ = param_grad_tolerances_for_dtype(torch.float64)
    assert rtol <= fp64_rtol
