"""Pins on the FORWARD-replay tolerance band (r7 b9-opus R74r6-F1).

The gradient bands have had hard per-dtype pins since the dtype-tolerance
sprint (``test_grad_tolerance_dtype_tripwire.py``); the forward-replay band --
``_tolerances_for_dtype``, the judge behind every per-op replay verdict --
appeared in ZERO test files. Nothing bounded it: the comparator self-test's
loosest sentinel was a 1/3 relative gap, so a band loosened 5,461x from the
shipped fp32 row still ran green and blessed 30% corruption of every replayed
activation. These are the mirror pins: exact derived rows, an rtol ceiling no
future headroom change can walk out of, and a corruption-fails assertion per
dtype family.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.utils.tensor_utils import (
    _ACCUMULATING_REPLAY_ULP_HEADROOM,
    _DTYPE_FLOAT_TOLERANCES,
    _LOW_PRECISION_REPLAY_ULP_HEADROOM,
    _tolerances_for_dtype,
    derive_float_tolerances,
    tensor_nanequal,
)

pytestmark = pytest.mark.smoke

#: No replay row may ever reach this: bf16 (the loosest shipped row) is
#: 4 * eps = 3.125e-2, and anything approaching O(1) blesses gross corruption
#: (the accidental 512-ULP derivation for a storage-rounding dtype produced
#: rtol 0.5 -- exactly the class this ceiling refuses).
_RTOL_CEILING = 0.05


def test_rows_are_exactly_the_derived_ulp_model() -> None:
    """Each shipped row equals its documented headroom-class derivation."""

    for dtype, headroom in (
        (torch.float32, _ACCUMULATING_REPLAY_ULP_HEADROOM),
        (torch.float64, _ACCUMULATING_REPLAY_ULP_HEADROOM),
        (torch.float16, _LOW_PRECISION_REPLAY_ULP_HEADROOM),
        (torch.bfloat16, _LOW_PRECISION_REPLAY_ULP_HEADROOM),
    ):
        assert _tolerances_for_dtype(dtype) == derive_float_tolerances(dtype, headroom), dtype


def test_fp32_row_is_the_exported_legacy_pair() -> None:
    """The facade constants and the fp32 replay row must not drift apart."""

    from torchlens.utils.tensor_utils import (
        MAX_FLOATING_POINT_TOLERANCE,
        REL_FLOATING_POINT_TOLERANCE,
    )

    assert _tolerances_for_dtype(torch.float32) == (
        REL_FLOATING_POINT_TOLERANCE,
        MAX_FLOATING_POINT_TOLERANCE,
    )


def test_fp64_row_is_tighter_than_fp32_by_the_eps_ratio() -> None:
    """fp64 replays earn the fp32 ULP budget in fp64 ULPs, not fp32 decimals."""

    eps_ratio = float(torch.finfo(torch.float64).eps) / float(torch.finfo(torch.float32).eps)
    rtol32, _ = _tolerances_for_dtype(torch.float32)
    rtol64, _ = _tolerances_for_dtype(torch.float64)
    assert rtol64 == pytest.approx(rtol32 * eps_ratio)


def test_every_row_respects_the_eps_floor_and_the_rtol_ceiling() -> None:
    """One-ULP agreement must pass; no row may drift toward O(1) looseness."""

    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        rtol, atol = _tolerances_for_dtype(dtype)
        eps = float(torch.finfo(dtype).eps)
        assert rtol >= eps, f"{dtype}: below one ULP false-fails every replay"
        assert rtol < _RTOL_CEILING, f"{dtype}: rtol {rtol} walked toward blessing corruption"
        assert atol < eps, f"{dtype}: atol {atol} blesses corruption of small normal values"


def test_derivation_cache_holds_the_derived_rows() -> None:
    """The process-global cache rows equal the pure derivation (poison pin)."""

    for dtype, headroom in (
        (torch.float32, _ACCUMULATING_REPLAY_ULP_HEADROOM),
        (torch.float64, _ACCUMULATING_REPLAY_ULP_HEADROOM),
        (torch.float16, _LOW_PRECISION_REPLAY_ULP_HEADROOM),
        (torch.bfloat16, _LOW_PRECISION_REPLAY_ULP_HEADROOM),
    ):
        assert _DTYPE_FLOAT_TOLERANCES[dtype] == derive_float_tolerances(dtype, headroom)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_ten_percent_corruption_fails_in_every_dtype_family(dtype: torch.dtype) -> None:
    """10% relative corruption must read UNEQUAL at every dtype's band.

    The loosest shipped row is bf16 at 4 ULP = 3.125e-2, so 10% clears every
    band with margin; a future loosening that blesses this class is exactly
    the corruption the comparator exists to catch.
    """

    values = torch.tensor([1.0, -2.0, 0.5, 8.0], dtype=dtype)
    corrupted = values * 1.10
    assert not tensor_nanequal(values, corrupted, allow_tolerance=True)


def test_zeroed_small_values_fail_fp32_and_fp64() -> None:
    """The atol term must never bless total corruption of small values.

    The former decimal atol floors (1e-5 fp32) read ``zeros == full(1e-6)``
    EQUAL -- zero detection power below the floor. The subnormal-quantum atol
    model must read it UNEQUAL.
    """

    for dtype in (torch.float32, torch.float64):
        true_values = torch.full((8,), 1e-6, dtype=dtype)
        zeroed = torch.zeros_like(true_values)
        assert not tensor_nanequal(zeroed, true_values, allow_tolerance=True), dtype


def test_comparator_self_test_bounds_the_effective_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.33-rtol poisoned band must FAIL the comparator self-test.

    Red-capable (the R74r6-F1 measurement): pre-fix the self-test's loosest
    sentinel gap was 1/3 relative, so ``rtol = 0.33`` -- ~5,400x looser than
    the shipped fp32 row, blessing 30% corruption -- still passed (measured
    at the hunt-6 pin). The two-sided band sentinels now reject it, and an
    absurdly TIGHT band fails the accept-side sentinel too.
    """

    from torchlens.utils import tensor_utils as tensor_utils_mod
    from torchlens.validation.core import _comparator_self_test

    _comparator_self_test()  # healthy band passes

    monkeypatch.setitem(tensor_utils_mod._DTYPE_FLOAT_TOLERANCES, torch.float32, (0.33, 1e-5))
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        _comparator_self_test()

    monkeypatch.setitem(tensor_utils_mod._DTYPE_FLOAT_TOLERANCES, torch.float32, (1e-9, 0.0))
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        _comparator_self_test()
