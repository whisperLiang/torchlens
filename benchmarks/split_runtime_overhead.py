"""Measure explicit split replay and analysis-only candidate overhead.

Run with ``python -W ignore benchmarks/split_runtime_overhead.py`` from this checkout.
To compare with an older revision, copy this script outside the repository and run
it with ``PYTHONPATH`` pointing at each checkout. Results are median wall times
in milliseconds, with P95 and individual samples. Use --source-root for comparison.
The measured checkout must support analysis, materialization and analysis-cache control.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from math import ceil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import nn


def _measure(
    action: Callable[[], Any],
    repeats: int = 20,
    *,
    before: Callable[[], Any] | None = None,
    warmup: int = 0,
) -> dict[str, Any]:
    """Return median, P95 and samples; optional setup is outside each timed span.

    Parameters
    ----------
    action
        Operation to time after explicit warmup by the caller.
    repeats
        Number of measured invocations.
    before
        Optional untimed setup such as clearing an analysis cache.
    warmup
        Untimed invocations with the same setup; cold-cache probes stay cold.
    """

    for _ in range(warmup):
        if before is not None:
            before()
        action()
    samples = []
    for _ in range(repeats):
        if before is not None:
            before()
        start = time.perf_counter()
        action()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[ceil(0.95 * len(samples)) - 1],
        "samples_ms": samples,
    }


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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup nonnegative")
    sys.path.insert(0, str(args.source_root.resolve()))

    import torchlens as tl
    from torchlens.split import SplitFeatures, SplitRequest, after

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    measure = partial(_measure, repeats=args.repeats, warmup=args.warmup)
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
        with torch.no_grad():
            torch.testing.assert_close(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)
        rows.append(
            {
                "model": f"width_{width}",
                "prefix": measure(lambda runtime=runtime, x=x: runtime.run_prefix(x)),
                "suffix": measure(
                    lambda runtime=runtime, boundary=boundary: runtime.run_suffix(boundary)
                ),
                "end_to_end": measure(
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
        candidate_ms = measure(lambda: runtime.split_points(diagnose=True))
        builds_during_candidates = build.call_count
    report = runtime.split_points(diagnose=True)
    selected = report.supported[0].point
    analysis_ms = measure(lambda: runtime.analyze(selected))
    analysis = runtime.analyze(selected)
    materialize_ms = measure(lambda: runtime.materialize(analysis))
    with patch.object(
        runtime.adapter, "build_segments", wraps=runtime.adapter.build_segments
    ) as build:
        selected_at_ms = measure(lambda: runtime.at(selected))
        builds_during_at = build.call_count
    cold_analysis = measure(lambda: runtime.analyze(selected), before=runtime.clear_analysis_cache)
    runtime.analyze(selected)
    cached_analysis = measure(lambda: runtime.analyze(selected))
    rows.append(
        {
            "model": "deep_8",
            "analysis_cache_miss": cold_analysis,
            "analysis_cache_hit": cached_analysis,
            "candidate": candidate_ms,
            "candidate_count": report.total,
            "supported_count": len(report.supported),
            "builds_during_candidates": builds_during_candidates,
            "selected_analyze": analysis_ms,
            "selected_materialize": materialize_ms,
            "selected_at": selected_at_ms,
            "builds_during_at": builds_during_at,
        }
    )
    print(
        json.dumps(
            {
                "source": str(tl.__file__),
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "repeats": args.repeats,
                "warmup": args.warmup,
                "seed": args.seed,
                "build_counts_include_warmup": True,
                "rows": rows,
            }
        )
    )


if __name__ == "__main__":
    main()
