"""Dataclass input and probe snapshots must not re-run model constructors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch
from v2_helpers import split_request

import torchlens as tl
from torchlens.split.adapters.torch import TorchSplitAdapter
from torchlens.split.batching import rebatch_inputs
from torchlens.split.pipeline import _snapshot_probe_output


def test_frozen_custom_init_dataclass_rebatches_without_constructor_calls() -> None:
    """Clone all fields, including init=False state, without touching originals."""

    calls: list[int] = []

    @dataclass(frozen=True, slots=True, init=False)
    class Input:
        """A tensor container whose constructor does not accept field names."""

        values: torch.Tensor
        cached: torch.Tensor = field(init=False)

        def __init__(self, size: int) -> None:
            """Construct state once, independently of replay cloning."""

            calls.append(size)
            object.__setattr__(self, "values", torch.arange(size * 3).reshape(size, 3))
            object.__setattr__(self, "cached", torch.tensor([7]))

    source = Input(2)
    adapter = TorchSplitAdapter()
    (cloned,), _ = rebatch_inputs(
        (source,), None, axes={"/args/0/values": 0}, batch_size=1, adapter=adapter
    )
    snapshot = _snapshot_probe_output(adapter, source)
    assert calls == [2]
    assert cloned is not source and snapshot is not source
    torch.testing.assert_close(cloned.values, source.values[:1])
    torch.testing.assert_close(snapshot.values, source.values)
    for copied in (cloned, snapshot):
        assert copied.values.data_ptr() != source.values.data_ptr()
        assert copied.cached.data_ptr() != source.cached.data_ptr()
        torch.testing.assert_close(copied.cached, source.cached)


def test_equinox_custom_init_parameter_input_replays_larger_batch() -> None:
    """Equinox's frozen dataclass module remains a valid JAX input pytree."""

    eqx = pytest.importorskip("equinox")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    calls: list[int] = []

    class Weights(eqx.Module):
        """Parameters initialized from a size, not from keyword field values."""

        weight: Any

        def __init__(self, size: int) -> None:
            """Initialize a deterministic dense parameter once."""

            calls.append(size)
            self.weight = jnp.arange(size * 2, dtype=jnp.float32).reshape(size, 2) / 10

    def model(params: Weights, x: Any) -> Any:
        """Treat the custom module as an explicit parameter pytree."""

        return jax.nn.relu(x @ params.weight)

    with jax.default_device(jax.devices("cpu")[0]):
        params = Weights(3)
        original = params.weight
        runtime = tl.split.prepare(
            model, (params, jnp.ones((2, 3))), split_request("50%", backend="jax")
        )
        assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
        for batch in (1, 2, 3):
            x = jnp.arange(batch * 3, dtype=jnp.float32).reshape(batch, 3) - 2
            np.testing.assert_allclose(runtime.replay(params, x), model(params, x), atol=1e-6)
        assert calls == [3]
        assert params.weight is original
