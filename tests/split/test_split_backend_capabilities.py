"""Split backend adapter capability gates."""

from __future__ import annotations

import pytest

from torchlens.backends import get_backend_spec
from torchlens.split.adapters import resolve_split_adapter
from torchlens.split.errors import SplitUnsupportedError


def test_mlx_adapter_does_not_fallback() -> None:
    """MLX still gates replay explicitly when its native runtime is unavailable."""

    adapter = resolve_split_adapter(get_backend_spec("mlx"))

    assert adapter.name == "mlx"
    assert adapter.supports_replay is False
    assert adapter.supports_training is False
    assert adapter.supports_boundary_cache is False
    with pytest.raises(SplitUnsupportedError):
        adapter.build_segments(None, None, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["jax", "tf"])
def test_native_ir_non_torch_adapters_advertise_replay_capabilities(backend: str) -> None:
    """JAX and TensorFlow expose native split replay/cache/training/dynamic batch."""

    adapter = resolve_split_adapter(get_backend_spec(backend))

    assert adapter.name in {backend, "tf"}
    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_dynamic_batch is True


def test_paddle_adapter_advertises_replay_capabilities() -> None:
    """Paddle has generated-eager replay/cache/training/dynamic batch."""

    adapter = resolve_split_adapter("paddle")

    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_dynamic_batch is True


def test_tinygrad_adapter_advertises_replay_capabilities() -> None:
    """tinygrad exposes UOp replay/cache/training/dynamic-batch support."""

    adapter = resolve_split_adapter("tinygrad")

    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_dynamic_batch is True


def test_torch_adapter_advertises_v1_capabilities() -> None:
    """Torch keeps full replay/training/cache/dynamic-batch capabilities."""

    adapter = resolve_split_adapter("torch")

    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_dynamic_batch is True
