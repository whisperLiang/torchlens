"""Targeted regression tests for hardening group r18g (tensor_utils + rng).

Each test pins a specific correctness / validation-tripwire finding reconciled
against main c2f27a9d. Tripwire-strengthening tests are accompanied (in the
fixer report) by a mutation proof: reverting the fix makes the paired assertion
fail. These tests are deterministic and CPU-only.
"""

import pytest
import torch

from torchlens.utils.tensor_utils import _copy_tensor_payload, tensor_nanequal


class _UncloneableTensor(torch.Tensor):
    """Tensor whose every clone strategy fails, exercising the last-resort path.

    ``detach`` returns ``self`` and ``.data`` preserves the subclass, so
    ``x.detach().clone()`` and ``x.data.cpu().clone()`` both route back through
    the overridden ``clone`` and raise -- reaching ``_copy_tensor_payload``'s
    final fallback.
    """

    def clone(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("clone deliberately disabled")

    def detach(self, *args, **kwargs):  # type: ignore[override]
        return self


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


# ---------------------------------------------------------------------------
# H3 -- _copy_tensor_payload must fail loud, never fabricate a zeros payload.
# ---------------------------------------------------------------------------
def test_h3_clone_failure_raises_not_fabricates():
    """A tensor that cannot be cloned must raise, not return fabricated zeros."""
    x = torch.tensor([3.0, 4.0], dtype=torch.float64).as_subclass(_UncloneableTensor)
    with pytest.raises(RuntimeError, match="could not copy a tensor payload"):
        _copy_tensor_payload(x, detach_tensor=True, save_mode="copy")


def test_h3_normal_tensor_still_copies():
    """The fail-loud path must not disturb ordinary copyable tensors."""
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)
    out = _copy_tensor_payload(x, detach_tensor=True, save_mode="copy")
    assert out.dtype == torch.float64
    assert torch.equal(out, x)
