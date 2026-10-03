"""Split backend adapter capability gates."""

from __future__ import annotations

import pytest

from torchlens.backends import get_backend_spec
from torchlens.split.adapters import resolve_split_adapter


def test_mlx_adapter_advertises_replay_training_and_placement_capabilities() -> None:
    """MLX exposes eager replay, functional training and native stream placement."""

    adapter = resolve_split_adapter(get_backend_spec("mlx"))

    assert adapter.name == "mlx"
    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_state_placement is True


@pytest.mark.parametrize("backend", ["jax", "tf"])
def test_native_ir_non_torch_adapters_advertise_replay_capabilities(backend: str) -> None:
    """JAX and TensorFlow expose native split replay/cache/training/dynamic batch."""

    adapter = resolve_split_adapter(get_backend_spec(backend))

    assert adapter.name in {backend, "tf"}
    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_state_placement is False


def test_paddle_adapter_advertises_replay_capabilities() -> None:
    """Paddle has generated-eager replay/cache/training/dynamic batch."""

    adapter = resolve_split_adapter("paddle")

    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_state_placement is False


def test_tinygrad_adapter_advertises_replay_capabilities() -> None:
    """tinygrad exposes UOp replay/cache/training/dynamic-batch support."""

    adapter = resolve_split_adapter("tinygrad")

    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_state_placement is False


def test_torch_adapter_advertises_v2_capabilities() -> None:
    """Torch keeps replay/training/cache and adds segment state placement."""

    adapter = resolve_split_adapter("torch")

    assert adapter.supports_replay is True
    assert adapter.supports_training is True
    assert adapter.supports_boundary_cache is True
    assert adapter.supports_state_placement is True


def test_no_adapter_declares_a_dynamic_batch_capability() -> None:
    """Batch polymorphism is unconditional: no adapter gates a batch range."""

    for name in ("torch", "jax", "tf", "paddle", "tinygrad", "mlx"):
        adapter = resolve_split_adapter(name)
        assert not hasattr(adapter, "supports_dynamic_batch")
