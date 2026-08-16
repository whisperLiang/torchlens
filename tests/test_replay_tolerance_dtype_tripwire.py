"""r7 R74 F1 (opus b9 MED-HIGH, PROVEN): pin the forward-replay tolerance band.

The band behind every per-op replay verdict (`_tolerances_for_dtype`,
consumed by ``validation/core.py`` with ``allow_tolerance=True``) appeared in
ZERO test files, while the sibling GRAD band is pinned hard
(``tests/test_grad_tolerance_dtype_tripwire.py``). The comparator's own entry
self-test is a PRESENCE check, not a TIGHTNESS check: opus binary-searched the
fp32 row and measured that an rtol of 0.3333 -- 5,461x looser than the shipped
6.1e-5 -- still passes the self-test and blesses 30% corruption of every
replayed activation. This file is the tightness pin: exact derived rows, a
ceiling no future headroom change can walk past the self-test's 1/3 blind
spot, and a corruption-fails assertion per dtype family.

The shipped band is correct; this is the tripwire that keeps it so.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.utils.tensor_utils import (
    _ACCUMULATING_REPLAY_ULP_HEADROOM,
    _DTYPE_FLOAT_TOLERANCES,
    _LOW_PRECISION_REPLAY_ULP_HEADROOM,
    _REPLAY_ULP_HEADROOM,
    _tolerances_for_dtype,
    derive_float_tolerances,
)

pytestmark = pytest.mark.smoke

#: The four precomputed rows and the corruption factor each must reject
#: (chosen far above the row's rtol so the assertion is about the BAND, not
#: about float noise: 1% >> 512*eps32, 10% >> 4*eps16, 30% >> 4*eps_bf16).
_ROW_CASES = (
    (torch.float32, _ACCUMULATING_REPLAY_ULP_HEADROOM, 1.01),
    (torch.float64, _ACCUMULATING_REPLAY_ULP_HEADROOM, 1.01),
    (torch.float16, _LOW_PRECISION_REPLAY_ULP_HEADROOM, 1.10),
    (torch.bfloat16, _LOW_PRECISION_REPLAY_ULP_HEADROOM, 1.30),
)


def test_replay_rows_equal_their_derivation_exactly() -> None:
    """Each precomputed row is exactly its finfo derivation -- no drifted literal."""

    for dtype, headroom, _ in _ROW_CASES:
        expected = derive_float_tolerances(dtype, headroom)
        assert _tolerances_for_dtype(dtype) == expected, dtype
        # The mutable process-global cache row matches a fresh derivation:
        # a poisoned or leaked row (the cache is written on first use) is a
        # verdict threshold, so drift here is corruption, not staleness.
        assert _DTYPE_FLOAT_TOLERANCES[dtype] == expected, dtype
    assert _REPLAY_ULP_HEADROOM[torch.float64] == _REPLAY_ULP_HEADROOM[torch.float32]


def test_replay_rtol_stays_far_below_the_self_test_blind_spot() -> None:
    """No row's rtol may approach 1/3 -- the comparator self-test's blind spot.

    The entry self-test's only relative sentinel is a 1/3 gap (0.5 vs 0.75),
    so any band under 1/3 passes it. This ceiling (1/8) keeps every row an
    order of magnitude inside that, so a future headroom bump must
    consciously edit this test rather than silently walk the band out.
    """

    for dtype, _, _ in _ROW_CASES:
        rtol, atol = _tolerances_for_dtype(dtype)
        eps = float(torch.finfo(dtype).eps)
        assert eps <= rtol < 1.0 / 8.0, (dtype, rtol)
        # atol forgives jitter only at the bottom of the representable range:
        # it must sit below the smallest NORMAL value, never blessing total
        # corruption of small normals (the pre-derivation decimal floors did).
        assert 0.0 < atol < float(torch.finfo(dtype).tiny), (dtype, atol)


def test_replay_band_rejects_corruption_per_dtype_family() -> None:
    """Uniform relative corruption well above each row's rtol must FAIL the band."""

    for dtype, _, corruption in _ROW_CASES:
        rtol, atol = _tolerances_for_dtype(dtype)
        base = torch.linspace(0.1, 4.0, 64).to(dtype)
        corrupted = (base.to(torch.float64) * corruption).to(dtype)
        assert not torch.allclose(base, corrupted, rtol=rtol, atol=atol), (
            f"{dtype}: {corruption:.2f}x corruption passed the replay band "
            f"(rtol={rtol}, atol={atol})"
        )
        # And the band is not vacuously tight: the value itself passes.
        assert torch.allclose(base, base.clone(), rtol=rtol, atol=atol)


def test_derived_out_of_table_rows_follow_the_eps_class() -> None:
    """Out-of-table float/complex dtypes derive their own row by eps class.

    complex64's component eps equals fp32's -> accumulating headroom;
    complex32 (component eps ~9.8e-4, storage-rounding class) must take the
    few-ULP budget -- deriving it at 512 ULP produced rtol 0.5, a row that
    would bless 40% corruption (the docstring's own incident).
    """

    rtol64, _ = _tolerances_for_dtype(torch.complex64)
    assert rtol64 == pytest.approx(
        _ACCUMULATING_REPLAY_ULP_HEADROOM * float(torch.finfo(torch.complex64).eps)
    )
    if hasattr(torch, "complex32"):
        rtol32, _ = _tolerances_for_dtype(torch.complex32)
        assert rtol32 == pytest.approx(
            _LOW_PRECISION_REPLAY_ULP_HEADROOM * float(torch.finfo(torch.complex32).eps)
        )
        assert rtol32 < 1.0 / 8.0
