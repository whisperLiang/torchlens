"""Targeted regression tests for hardening group r18g (tensor_utils + rng).

Each test pins a specific correctness / validation-tripwire finding reconciled
against main c2f27a9d. Tripwire-strengthening tests are accompanied (in the
fixer report) by a mutation proof: reverting the fix makes the paired assertion
fail. These tests are deterministic and CPU-only.
"""

import pytest
import torch

from torchlens.utils.tensor_utils import tensor_nanequal


# ---------------------------------------------------------------------------
# H2 -- tensor_nanequal must distinguish a NaN from the finite sentinel value.
# ---------------------------------------------------------------------------
_SENTINEL = 0.7234691827346


def test_h2_nan_vs_finite_sentinel_not_equal():
    """A NaN must never read EQUAL to a real finite value (esp. the sentinel)."""
    nan_side = torch.tensor([float("nan"), 1.0])
    sentinel_side = torch.tensor([_SENTINEL, 1.0])
    assert tensor_nanequal(nan_side, sentinel_side) is False
    assert tensor_nanequal(sentinel_side, nan_side) is False  # symmetric


def test_h2_matching_nan_pattern_still_equal():
    """Genuinely-equal tensors (same NaN pattern + values) still compare equal."""
    a = torch.tensor([float("nan"), 2.0, float("nan")])
    b = torch.tensor([float("nan"), 2.0, float("nan")])
    assert tensor_nanequal(a, b) is True


def test_h2_complex_nan_component_distinguished():
    """Complex: a NaN in a component must not read equal to a finite twin."""
    a = torch.tensor([complex(float("nan"), 1.0)])
    b = torch.tensor([complex(_SENTINEL, 1.0)])
    assert tensor_nanequal(a, b) is False


# ---------------------------------------------------------------------------
# H4 -- float allclose tolerance must NOT apply to integer/bool dtypes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_h4_integer_dtypes_are_exact_under_tolerance(dtype):
    """Distinct integers must never read equal, even with allow_tolerance=True.

    Values are large enough that the default float rtol (1e-4) would otherwise
    absorb a diff of 1 (1e-4 * 1_000_001 ~= 100 >> 1) -- the exact false-equal
    the pre-fix code produced.
    """
    a = torch.tensor([1_000_000], dtype=dtype)
    b = torch.tensor([1_000_001], dtype=dtype)
    assert tensor_nanequal(a, b, allow_tolerance=True) is False


def test_h4_small_integers_also_exact():
    """uint8/int16 exact-equality holds too (belt-and-suspenders)."""
    a = torch.tensor([10, 20], dtype=torch.int16)
    b = torch.tensor([10, 21], dtype=torch.int16)
    assert tensor_nanequal(a, b, allow_tolerance=True) is False


def test_h4_bool_dtype_exact_under_tolerance():
    a = torch.tensor([True, False])
    b = torch.tensor([True, True])
    assert tensor_nanequal(a, b, allow_tolerance=True) is False


def test_h4_float_tolerance_still_allowed():
    """Floating-point tolerance path is preserved for genuine float jitter."""
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([1.0 + 1e-7, 2.0 - 1e-7], dtype=torch.float32)
    assert tensor_nanequal(a, b, allow_tolerance=True) is True
    assert tensor_nanequal(a, b, allow_tolerance=False) is False
