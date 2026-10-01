"""Measure explicit split replay and analysis-only candidate overhead.

Run with ``python -W ignore benchmarks/split_runtime_overhead.py`` from this checkout.
To compare with an older revision, copy this script outside the repository and run
it with ``PYTHONPATH`` pointing at each checkout. Results are median wall times
in milliseconds. Older revisions without ``analyze`` report null for the two
separate selected-point stages while still measuring ``at``.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

import torchlens as tl
from torchlens.split import SplitFeatures, SplitRequest, after


def _median_ms(action: Callable[[], Any], repeats: int = 12) -> float:
    """Return median elapsed milliseconds for repeated calls.

    Parameters
    ----------
    action
        Operation to time after warmup.
    repeats
        Number of measured invocations.
    """

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        samples.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(samples), 3)


class _WidthModel(nn.Module):
    """Two affine layers with a short prefix and a parameterized suffix."""

    def __init__(self, width: int) -> None:
        """Build the benchmark model."""

        super().__init__()
        self.head = nn.Linear(width, width)
        self.relu = nn.ReLU()
        self.tail = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the model."""

        return self.tail(self.relu(self.head(x)))


class _DeepModel(nn.Module):
    """A small model with 32 before/after candidate points."""

    def __init__(self) -> None:
        """Build repeated affine and activation operations."""

        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(8)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all layers in order."""

        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def main() -> None:
    """Print comparable replay, candidate, and selected-cut measurements."""

    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    for width in (16, 1024):
        model = _WidthModel(width).eval()
        x = torch.randn(1, width)
        runtime = tl.split.prepare(
            model,
            x,
            SplitRequest(point=after("relu"), features=SplitFeatures(retain_trace=False)),
        )
        boundary = runtime.run_prefix(x)
        runtime.run_suffix(boundary)
        rows.append(
            {
                "model": f"width_{width}",
                "prefix_ms": _median_ms(lambda runtime=runtime, x=x: runtime.run_prefix(x)),
                "suffix_ms": _median_ms(
                    lambda runtime=runtime, boundary=boundary: runtime.run_suffix(boundary)
                ),
                "end_to_end_ms": _median_ms(
                    lambda runtime=runtime, x=x: runtime.run_suffix(runtime.run_prefix(x))
                ),
            }
        )

    model = _DeepModel().eval()
    x = torch.randn(1, 8)
    runtime = tl.split.prepare(
        model,
        x,
        SplitRequest(point=after("layers.3"), features=SplitFeatures(retain_trace=False)),
    )
    with patch.object(
        runtime.adapter, "build_segments", wraps=runtime.adapter.build_segments
    ) as build:
        candidate_ms = _median_ms(lambda: runtime.split_points(diagnose=True), repeats=3)
        builds_during_candidates = build.call_count
    report = runtime.split_points(diagnose=True)
    selected = report.supported[0].point
    analysis_ms = None
    materialize_ms = None
    if hasattr(runtime, "analyze"):
        analysis_ms = _median_ms(lambda: runtime.analyze(selected), repeats=3)
        analysis = runtime.analyze(selected)
        materialize_ms = _median_ms(lambda: runtime.materialize(analysis), repeats=3)
    with patch.object(
        runtime.adapter, "build_segments", wraps=runtime.adapter.build_segments
    ) as build:
        selected_at_ms = _median_ms(lambda: runtime.at(selected), repeats=3)
        builds_during_at = build.call_count
    rows.append(
        {
            "model": "deep_8",
            "candidate_ms": candidate_ms,
            "candidate_count": report.total,
            "supported_count": len(report.supported),
            "builds_during_candidates": builds_during_candidates,
            "selected_analyze_ms": analysis_ms,
            "selected_materialize_ms": materialize_ms,
            "selected_at_ms": selected_at_ms,
            "builds_during_at": builds_during_at,
        }
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
