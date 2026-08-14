"""Tripwire-strengthening tests for the MLX and Paddle replay oracles.

Two masked-corruption classes are locked out here:

* The MLX oracle used ``atol == rtol`` per dtype band, so every element whose
  magnitude sat below the band's decimal tolerance could be replaced by zero
  (or sign-flipped) and still read EQUAL.  The absolute term must only absorb
  jitter at the bottom of the representable range, never bless corruption of
  small normal values.
* The Paddle oracle applied one fp32-shaped decimal pair (rtol 1e-5 /
  atol 1e-6) to EVERY float dtype: fp64 corruption thousands of times larger
  than fp64 round-off read as agreement, while a legitimate 1-ULP fp16
  replay difference false-FAILED.

These tests exercise the pure-numpy comparison cores directly, so they run
without the optional backend runtimes installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchlens.backends.mlx.validation import _payloads_close as mlx_payloads_close
from torchlens.backends.paddle.validation import _arrays_close as paddle_arrays_close

pytestmark = pytest.mark.smoke


class TestMLXPayloadsClose:
    """MLX replay oracle: absolute term must not bless sub-band corruption."""

    def test_all_zero_replay_of_low_magnitude_fp32_fails(self) -> None:
        """An all-zero replay of activations below the old atol must FAIL."""

        saved = np.full((8,), 4e-6, dtype=np.float32)
        assert not mlx_payloads_close(np.zeros_like(saved), saved)

    def test_sign_flipped_replay_of_low_magnitude_fp32_fails(self) -> None:
        """A sign-flipped replay of low-magnitude activations must FAIL."""

        saved = np.array([4e-6, -3e-6, 2e-6, -1e-6], dtype=np.float32)
        assert not mlx_payloads_close(-saved, saved)

    def test_all_zero_replay_of_low_magnitude_fp16_fails(self) -> None:
        """The fp16 band's absolute term must not swallow zeroed elements."""

        saved = np.full((8,), 4e-4, dtype=np.float16)
        assert not mlx_payloads_close(np.zeros_like(saved), saved)

    def test_identical_payloads_pass(self) -> None:
        """Exact agreement stays a PASS."""

        saved = np.array([4e-6, -3e-6, 0.0, 1.5], dtype=np.float32)
        assert mlx_payloads_close(saved.copy(), saved)

    def test_relative_jitter_within_band_passes(self) -> None:
        """Normal-range replays within the relative band stay a PASS."""

        saved = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        replay = saved * (1.0 + 5e-6)
        assert mlx_payloads_close(replay.astype(np.float32), saved)

    def test_fp16_one_ulp_rounding_passes(self) -> None:
        """A one-ULP fp16 storage-rounding difference is agreement, not corruption."""

        saved = np.ones((4,), dtype=np.float16)
        replay = np.nextafter(saved, np.float16(2.0), dtype=np.float16)
        assert mlx_payloads_close(replay, saved)

    def test_nan_pattern_agreement_passes_nan_vs_number_fails(self) -> None:
        """equal_nan doctrine: identical NaN patterns agree, NaN-vs-number fails."""

        saved = np.array([np.nan, 1.0], dtype=np.float32)
        assert mlx_payloads_close(saved.copy(), saved)
        assert not mlx_payloads_close(np.array([0.0, 1.0], dtype=np.float32), saved)


class TestPaddleArraysClose:
    """Paddle replay oracle: tolerances must be dtype-aware in both directions."""

    def test_fp64_small_value_corruption_fails(self) -> None:
        """fp64 elements below the old fp32 atol must not be zeroable."""

        saved = np.full((8,), 5e-7, dtype=np.float64)
        assert not paddle_arrays_close(np.zeros_like(saved), saved)

    def test_fp64_relative_corruption_fails(self) -> None:
        """fp64 relative corruption ~4e9 ULP must FAIL (old rtol blessed it)."""

        saved = np.ones((8,), dtype=np.float64)
        assert not paddle_arrays_close(saved * (1.0 + 1e-6), saved)

    def test_fp64_reorder_scale_noise_passes(self) -> None:
        """fp64 differences at fp64 round-off scale stay a PASS."""

        saved = np.ones((8,), dtype=np.float64)
        replay = saved * (1.0 + 4.0 * float(np.finfo(np.float64).eps))
        assert paddle_arrays_close(replay, saved)

    def test_fp32_small_value_corruption_fails(self) -> None:
        """fp32 elements below the old 1e-6 atol must not be zeroable."""

        saved = np.full((8,), 5e-7, dtype=np.float32)
        assert not paddle_arrays_close(np.zeros_like(saved), saved)

    def test_fp32_relative_band_unchanged(self) -> None:
        """The fp32 relative band stays at the legacy 1e-5 strictness."""

        saved = np.ones((8,), dtype=np.float32)
        assert paddle_arrays_close(saved * (1.0 + 5e-6), saved)
        assert not paddle_arrays_close(saved * (1.0 + 1e-4), saved)

    def test_fp16_one_ulp_rounding_passes(self) -> None:
        """A one-ULP fp16 storage difference is a legitimate replay, not a FAIL.

        The old dtype-blind pair (rtol 1e-5, two orders below fp16's own eps)
        false-failed every non-bitwise fp16 replay; dtype-aware scaling admits
        storage rounding while still catching real corruption below.
        """

        saved = np.ones((4,), dtype=np.float16)
        replay = np.nextafter(saved, np.float16(2.0), dtype=np.float16)
        assert paddle_arrays_close(replay, saved)

    def test_fp16_zeroed_replay_fails(self) -> None:
        """Zeroed fp16 activations still FAIL under the dtype-aware band."""

        saved = np.full((8,), 1e-2, dtype=np.float16)
        assert not paddle_arrays_close(np.zeros_like(saved), saved)

    def test_shape_and_dtype_mismatch_fail(self) -> None:
        """Shape or dtype disagreement is corruption, never coerced."""

        saved = np.ones((4,), dtype=np.float32)
        assert not paddle_arrays_close(np.ones((5,), dtype=np.float32), saved)
        assert not paddle_arrays_close(np.ones((4,), dtype=np.float64), saved)

    def test_integer_and_bool_stay_exact(self) -> None:
        """Integer (incl. bf16-as-uint16 transport) and bool payloads compare exactly."""

        ints = np.arange(4, dtype=np.uint16)
        assert paddle_arrays_close(ints.copy(), ints)
        assert not paddle_arrays_close(ints + 1, ints)
        bools = np.array([True, False])
        assert paddle_arrays_close(bools.copy(), bools)
        assert not paddle_arrays_close(~bools, bools)

    def test_nan_pattern_agreement_passes_nan_vs_number_fails(self) -> None:
        """equal_nan doctrine holds for the dtype-aware bands."""

        saved = np.array([np.nan, 1.0], dtype=np.float64)
        assert paddle_arrays_close(saved.copy(), saved)
        assert not paddle_arrays_close(np.array([0.0, 1.0], dtype=np.float64), saved)


@pytest.mark.backend_paddle
class TestPaddlePayloadsCloseLive:
    """The paddle-tensor wrapper must route through the dtype-aware core."""

    def test_fp64_corruption_fails_on_live_tensors(self) -> None:
        """fp64 corruption below the old decimal pair FAILS on real tensors."""

        paddle = pytest.importorskip("paddle")
        from torchlens.backends.paddle.validation import _payloads_close

        saved = paddle.full((8,), 5e-7, dtype="float64")
        zeros = paddle.zeros((8,), dtype="float64")
        assert not _payloads_close(zeros, saved)
        assert _payloads_close(saved.clone(), saved)
