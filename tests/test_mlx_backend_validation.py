"""MLX live replay-validation oracle tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.backend_mlx

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402
from torchlens.validation.status import ValidationReplayStatus  # noqa: E402


class _TwoLayerMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 3)
        self.l2 = nn.Linear(3, 2)

    def __call__(self, x: mx.array) -> mx.array:
        return self.l2(nn.relu(self.l1(x)))


def _healthy_trace() -> tl.Trace:
    return tl.trace(_TwoLayerMLP(), mx.ones((1, 4)), backend="mlx")


def test_mlx_validation_healthy_two_layer_mlp_passes() -> None:
    """Validate a healthy MLX MLP with per-op replay and perturbation."""

    trace = _healthy_trace()
    assert MLXBackend().validate_trace(trace) is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert status.replayed_node_count >= 1


def test_mlx_validation_registered_spec_dispatches() -> None:
    """The registered spec routes to the real oracle, not the old refusal."""

    from torchlens.backends import get_backend_spec

    spec = get_backend_spec("mlx")
    assert spec.capabilities.validation_replay is True
    assert spec.validate_trace(_healthy_trace()) is True


def test_mlx_validation_fails_corrupted_saved_output() -> None:
    """Fail validation when a saved MLX op output payload is corrupted."""

    trace = _healthy_trace()
    victim = next(op for op in trace.layer_list if op.out is not None and op.uses_params)
    with pytest.warns(UserWarning):
        victim.out = victim.out + 1.0

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_fails_dropped_capture_material() -> None:
    """Fail (never vacuously pass) when replay material is missing on a live trace."""

    trace = _healthy_trace()
    trace._mlx_op_captures = []

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_loaded_payload_stripped_trace_is_unavailable() -> None:
    """Return unavailable status for loaded traces stripped of replay material."""

    trace = _healthy_trace()
    trace._loaded_from_bundle = True
    trace._mlx_op_captures = ()

    result = MLXBackend().validate_trace(trace)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unavailable"
    assert result.passed is False
