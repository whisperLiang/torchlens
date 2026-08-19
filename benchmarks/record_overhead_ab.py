"""A/B overhead measurement for ``tl.record`` on real vision models.

Methodology follows the promoted P2 A/B harness
(``tests/test_perf_capture_ab.py``): torch intra-op threads pinned to 1,
median-of-N wall clock with one warmup call, load context recorded. This
script extends the small-capture rows (tiny CNN/MLP) with real-model rows —
torchvision ResNet-50 and ViT-B/16 at ``(1, 3, 224, 224)`` — and measures the
sparse predicate recorder ``tl.record`` alongside full ``tl.trace`` capture:

- ``native``: plain ``model(x)`` under ``torch.no_grad()``.
- ``record``: ``tl.record(model, x, save=<predicate>)`` (sparse event stream).
- ``trace_pred``: ``tl.trace(model, x, save=<predicate>)`` (full structure,
  selective payload retention).
- ``trace_default``: ``tl.trace(model, x)`` (full structure, default saves).

Run it as a script; it prints one JSON payload:

    python benchmarks/record_overhead_ab.py [--n 8] [--models resnet50,vit_b_16]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torchlens as tl  # noqa: E402


def measure_median_ms(fn: Callable[[], Any], n: int = 8, warmup: int = 1) -> dict[str, float]:
    """Return median/min/p75 wall-clock milliseconds over ``n`` timed calls."""

    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "min_ms": times[0],
        "p75_ms": times[(3 * len(times)) // 4],
        "n": float(n),
    }


def load_context() -> dict[str, Any]:
    """Record the box/load context measurements were taken under."""

    context: dict[str, Any] = {
        "nproc": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }
    try:
        context["load_average_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        context["load_average_1m"] = None
    return context


class _single_torch_thread:
    """Pin torch intra-op threads to 1 for the measurement, then restore."""

    def __enter__(self) -> None:
        self._saved = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc: object) -> None:
        torch.set_num_threads(self._saved)


def _build_model(name: str) -> tuple[nn.Module, torch.Tensor, Any]:
    """Return (model, input, save predicate) for one benchmark row."""

    import torchvision.models as tvm

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    if name == "resnet50":
        return tvm.resnet50(weights=None).eval(), x, tl.func("relu")
    if name == "vit_b_16":
        return tvm.vit_b_16(weights=None).eval(), x, tl.func("layer_norm")
    raise SystemExit(f"unknown model row: {name}")


def real_model_rows(names: list[str], n: int) -> dict[str, dict[str, float]]:
    """Measure native vs record vs trace tiers per real-model row."""

    rows: dict[str, dict[str, float]] = {}
    with _single_torch_thread():
        for name in names:
            model, x, predicate = _build_model(name)
            with torch.no_grad():
                native = measure_median_ms(lambda m=model, inp=x: m(inp), n=n)
            record = measure_median_ms(
                lambda m=model, inp=x, p=predicate: tl.record(m, inp, save=p), n=n
            )
            trace_pred = measure_median_ms(
                lambda m=model, inp=x, p=predicate: tl.trace(m, inp, save=p), n=n
            )
            trace_default = measure_median_ms(lambda m=model, inp=x: tl.trace(m, inp), n=n)
            rows[name] = {
                "native_ms": native["median_ms"],
                "record_ms": record["median_ms"],
                "trace_pred_ms": trace_pred["median_ms"],
                "trace_default_ms": trace_default["median_ms"],
                "record_ratio": record["median_ms"] / native["median_ms"],
                "trace_pred_ratio": trace_pred["median_ms"] / native["median_ms"],
                "trace_default_ratio": trace_default["median_ms"] / native["median_ms"],
                "n": float(n),
            }
    return rows


def main() -> None:
    """Measurement pass: print all rows as one JSON payload."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--models", default="resnet50,vit_b_16")
    args = parser.parse_args()

    payload: dict[str, Any] = {"context": load_context()}
    payload["real_models"] = real_model_rows(args.models.split(","), n=args.n)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
