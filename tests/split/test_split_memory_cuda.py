"""Opt-in allocated-memory regressions for detached prefixes and suffix microbatches.

Run with ``TORCHLENS_CUDA_MEMORY_TESTS=1 pytest -s`` to print measured bytes.
The comparisons use one runtime, warm both paths, clear optimizer gradients,
and release results before resetting CUDA's allocated-memory peak. Cached
allocator reservations are deliberately not counted or flushed.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import ReplayBoundary, SplitRuntime

pytestmark = [
    pytest.mark.slow,
    pytest.mark.serial,
    pytest.mark.optional,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable."),
    pytest.mark.skipif(
        os.environ.get("TORCHLENS_CUDA_MEMORY_TESTS") != "1",
        reason="TORCHLENS_CUDA_MEMORY_TESTS=1 enables CUDA peak-memory comparisons.",
    ),
]


class ActivationHeavySplit(nn.Module):
    """Small state with spatial activations dominating each segment's memory."""

    def __init__(self) -> None:
        """Build separate prefix and suffix parameter owners."""

        super().__init__()
        self.prefix = nn.Sequential(
            *[layer for _ in range(8) for layer in (nn.Conv2d(8, 8, 1), nn.Tanh())]
        )
        self.bridge = nn.ReLU()
        self.suffix = nn.Sequential(
            *[layer for _ in range(12) for layer in (nn.Conv2d(8, 8, 1), nn.Tanh())]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify pooled suffix activations.

        Parameters
        ----------
        x:
            Batch of eight-channel spatial inputs.
        """

        hidden = self.suffix(self.bridge(self.prefix(x)))
        return self.head(self.pool(hidden).flatten(1))


def _reset_peak() -> int:
    """Collect released Python objects and start an allocated-memory interval."""

    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    return baseline


def _suffix_peak(
    runtime: SplitRuntime,
    boundary: ReplayBoundary,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    microbatch_size: int | None,
) -> dict[str, int | float]:
    """Measure one step without returning references to its graph or gradients.

    Parameters
    ----------
    runtime, boundary, targets:
        Prepared runtime, detached logical boundary, and classification labels.
    optimizer:
        Zero-learning-rate optimizer; the same parameters serve every measurement.
    microbatch_size:
        Optional suffix execution batch size.
    """

    optimizer.zero_grad(set_to_none=True)
    baseline = _reset_peak()
    loss, gradients = runtime.train_suffix(
        boundary, targets, optimizer=optimizer, microbatch_size=microbatch_size
    )
    torch.cuda.synchronize()
    assert gradients
    return {
        "baseline_bytes": baseline,
        "peak_bytes": torch.cuda.max_memory_allocated(),
        "loss": loss.detach().item(),
    }


def _prefix_peak(runtime: SplitRuntime, inputs: torch.Tensor, *, connected: bool) -> dict[str, int]:
    """Measure a prefix while its returned boundary remains alive.

    Parameters
    ----------
    runtime, inputs:
        Prepared runtime and one logical batch.
    connected:
        Whether to retain the prefix graph for its later backward operation.
    """

    baseline = _reset_peak()
    boundary = runtime.run_training_prefix(inputs) if connected else runtime.run_prefix(inputs)
    torch.cuda.synchronize()
    assert any(value.grad_fn is not None for value in boundary.tensors.values()) == connected
    return {"baseline_bytes": baseline, "peak_bytes": torch.cuda.max_memory_allocated()}


def test_cuda_microbatch_and_detached_prefix_reduce_peak_allocated_memory(
    record_property: Callable[[str, object], None],
) -> None:
    """Repeated steady-state measurements show both activation-memory reductions.

    Parameters
    ----------
    record_property:
        Pytest recorder used to retain exact measurements in JUnit reports.
    """

    torch.manual_seed(13)
    model = ActivationHeavySplit().cuda().eval()
    inputs = torch.randn(12, 8, 128, 128, device="cuda")
    targets = torch.randint(0, 4, (12,), device="cuda")
    runtime = tl.split.prepare(model, inputs[:1], split_request("after:bridge", trainable=True))
    graph_identity = runtime.graph_identity
    split_id = runtime.split_id
    boundary = runtime.run_prefix(inputs)
    optimizer = torch.optim.SGD([*model.suffix.parameters(), *model.head.parameters()], lr=0)

    # Warm both batch shapes and CUDA library workspaces before comparing.
    for microbatch_size in (None, 3):
        _suffix_peak(runtime, boundary, targets, optimizer, microbatch_size)
    suffix_measurements = []
    for _ in range(2):
        full = _suffix_peak(runtime, boundary, targets, optimizer, None)
        micro = _suffix_peak(runtime, boundary, targets, optimizer, 3)
        assert micro["peak_bytes"] < full["peak_bytes"], (full, micro)
        assert micro["loss"] == pytest.approx(full["loss"], rel=1e-5, abs=1e-6)
        suffix_measurements.append({"full": full, "microbatch": micro})

    optimizer.zero_grad(set_to_none=True)
    del boundary
    for connected in (False, True):
        _prefix_peak(runtime, inputs, connected=connected)
    prefix_measurements = []
    for _ in range(2):
        detached = _prefix_peak(runtime, inputs, connected=False)
        connected = _prefix_peak(runtime, inputs, connected=True)
        assert detached["peak_bytes"] < connected["peak_bytes"], (detached, connected)
        prefix_measurements.append({"detached": detached, "connected": connected})

    assert runtime.graph_identity == graph_identity
    assert runtime.split_id == split_id
    measurements = {
        "model": "ActivationHeavySplit",
        "device": torch.cuda.get_device_name(),
        "logical_batch": 12,
        "microbatch_size": 3,
        "input_shape": list(inputs.shape),
        "split_point": "after:bridge",
        "split_node": runtime.plan.target_node_id,
        "suffix": suffix_measurements,
        "prefix": prefix_measurements,
    }
    encoded = json.dumps(measurements, sort_keys=True)
    record_property("cuda_memory_measurements", encoded)
    print(f"TorchLens split CUDA memory: {encoded}")
