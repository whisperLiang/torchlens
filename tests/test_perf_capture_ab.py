"""Quiet-box A/B capture-overhead and fast-tier harness with re-baselined gates.

This is the durable home of the ad-hoc ``/tmp/perf_ab.py`` A/B harness (D14
ruling: the ~20-30% small-capture fixed overhead is ACCEPTED for the feature
sprint; the scheduled post-features perf pass owns optimizing it, and the
harness is promoted here so the measurement methodology survives). Gates key
to measurements taken at the sprint merge-base, recorded in
``tests/perf_baselines/capture_ab_baseline.json``, with the D15 blocking
ceiling: a re-measured gate metric may not exceed its baseline by more than
10%. The 2% strict advisory mode arrives with the post-features perf pass.

Two workloads:

1. **Small-capture A/B** — native forward vs ``tl.trace`` on the tiny
   CNN/MLP reference pair (verbatim from the original harness). The gated
   metric is the dimensionless overhead ratio ``trace_ms / native_ms``,
   which self-normalizes across boxes far better than raw milliseconds;
   the raw medians are recorded alongside for provenance.
2. **Fast-tier decode pair** — native vs wrapped-episode (``tl.trace`` per
   decode step) vs guarded-fast (``trace.run(inputs=..., fast=True)`` per
   step) on a fixed-window greedy decode loop over a HuggingFace causal LM.
   The METAPLAN P2 floor (r3 decision rule) is absolute: guarded-fast must
   beat wrapped-episode by >= 2x wall clock on the primary CPU row. Both
   tiers share the same explicit functional save predicate
   (``tl.func("layer_norm")``), the fast tier's supported scoped-collection
   spelling. ``use_cache=False`` keeps the output structure fixed across
   steps (a growing ``past_key_values`` cache would trip the fast tier's
   output-structure guard, and the fixed-shape sliding window is the
   reference decode workload anyway).

Marked ``rare``: this file must never run in smoke or the default tier
(``addopts = -m 'not rare'`` deselects it); perf measurement needs a quiet
box and the decode rows take tens of seconds. Select it explicitly:

    pytest tests/test_perf_capture_ab.py -m rare

or run it as a script for a measurement/report pass (prints all rows plus
load context; ``--write-baseline`` refreshes the committed baseline — only
do that at a declared re-baselining point, never to absorb a regression):

    python tests/test_perf_capture_ab.py [--decode-model gpt2] [--write-baseline]
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

try:
    import torchlens as tl
except ModuleNotFoundError:  # script mode: put the repo root on sys.path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torchlens as tl

pytestmark = pytest.mark.rare

#: D15 ruling: blocking merge gate — measured metric <= baseline * (1 + 10%).
GATE_CEILING_FRACTION = 0.10

#: METAPLAN P2 r3 decision rule: guarded-fast must beat wrapped-episode by
#: >= this factor (wall clock) on the primary CPU decode row, or the default
#: tier reverts to wrapped-episode and guarded-fast ships opt-in.
FAST_TIER_FLOOR = 2.0

BASELINE_PATH = Path(__file__).parent / "perf_baselines" / "capture_ab_baseline.json"

#: Default decode reference model (gpt2-class row). The mid-LLM row is the
#: same harness pointed at a larger checkpoint via ``--decode-model`` /
#: ``TL_PERF_AB_DECODE_MODEL``; only the gpt2-class row is asserted in-test
#: so the test stays runnable from the common HF cache.
DECODE_MODEL_DEFAULT = "gpt2"
DECODE_WINDOW = 64
DECODE_STEPS = 12


def _mk_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


def _mk_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


SMALL_CAPTURE_CASES: tuple[tuple[str, Callable[[], nn.Module], tuple[int, ...]], ...] = (
    ("cnn", _mk_cnn, (2, 3, 32, 32)),
    ("mlp", _mk_mlp, (8, 128)),
)


def measure_median_ms(fn: Callable[[], Any], n: int = 12, warmup: int = 1) -> dict[str, float]:
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


def capture_overhead_rows(n: int = 12) -> dict[str, dict[str, float]]:
    """Measure native forward vs ``tl.trace`` on the small-capture pair."""

    rows: dict[str, dict[str, float]] = {}
    with _single_torch_thread():
        torch.manual_seed(0)
        for name, build, shape in SMALL_CAPTURE_CASES:
            model = build().eval()
            x = torch.randn(*shape)
            with torch.no_grad():
                native = measure_median_ms(lambda: model(x), n=n)
            trace = measure_median_ms(lambda: tl.trace(model, x), n=n)
            rows[name] = {
                "native_ms": native["median_ms"],
                "trace_ms": trace["median_ms"],
                "ratio": trace["median_ms"] / native["median_ms"],
                "n": float(n),
            }
    return rows


def decode_pair_rows(
    model_name: str = DECODE_MODEL_DEFAULT,
    steps: int = DECODE_STEPS,
    window: int = DECODE_WINDOW,
) -> dict[str, float]:
    """Measure native vs wrapped-episode vs guarded-fast on a decode loop.

    Fixed-window greedy decode: the ``(1, window)`` id sequence is rolled
    forward natively once (append argmax token, drop the oldest), and then
    every tier is timed over that identical shape-stable input sequence, so
    the three rows compare pure per-step forward cost with no
    output-extraction asymmetry.
    """

    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).eval()
    model.config.use_cache = False
    prompt = "The quick brown fox jumps over the lazy dog. " * window
    ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :window]
    predicate = tl.func("layer_norm")

    def run_tier(step: Callable[[torch.Tensor], Any], windows: list[torch.Tensor]) -> float:
        times: list[float] = []
        for current in windows:
            start = time.perf_counter()
            step(current)
            times.append((time.perf_counter() - start) * 1000.0)
        return statistics.median(times)

    with _single_torch_thread():
        # Roll the greedy fixed-window decode forward natively once.
        windows = [ids]
        with torch.no_grad():
            for _ in range(steps - 1):
                logits = model(windows[-1]).logits
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                windows.append(torch.cat([windows[-1][:, 1:], nxt], dim=1))

        # Warm both tiers once: wrapper install, module prep, fast binders.
        warm = tl.trace(model, ids, save=predicate)
        warm.run(inputs=ids, fast=True)

        def native_step(current: torch.Tensor) -> Any:
            with torch.no_grad():
                return model(current)

        def wrapped_step(current: torch.Tensor) -> Any:
            return tl.trace(model, current, save=predicate)

        def fast_step(current: torch.Tensor) -> Any:
            return warm.run(inputs=current, fast=True)

        native_ms = run_tier(native_step, windows)
        wrapped_ms = run_tier(wrapped_step, windows)
        fast_ms = run_tier(fast_step, windows)

    return {
        "native_ms": native_ms,
        "wrapped_ms": wrapped_ms,
        "fast_ms": fast_ms,
        "fast_vs_wrapped": wrapped_ms / fast_ms,
        "wrapped_vs_native": wrapped_ms / native_ms,
        "fast_vs_native": fast_ms / native_ms,
        "steps": float(steps),
        "window": float(window),
    }


def _load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        pytest.skip(f"no committed baseline at {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text())


def test_small_capture_overhead_within_gate() -> None:
    """D15 gate: overhead ratio within 10% of the merge-base baseline."""

    baseline = _load_baseline()
    rows = capture_overhead_rows()
    failures: list[str] = []
    for name, row in rows.items():
        base_ratio = baseline["small_capture"][name]["ratio"]
        ceiling = base_ratio * (1.0 + GATE_CEILING_FRACTION)
        if row["ratio"] > ceiling:
            failures.append(
                f"{name}: ratio {row['ratio']:.2f} > gate {ceiling:.2f} "
                f"(baseline {base_ratio:.2f} @ {baseline['sha'][:8]})"
            )
    assert not failures, "; ".join(failures)


def test_fast_tier_floor_decode_reference() -> None:
    """METAPLAN P2 floor: guarded-fast >= 2x wrapped on the CPU decode row."""

    row = decode_pair_rows()
    assert row["fast_vs_wrapped"] >= FAST_TIER_FLOOR, (
        f"guarded-fast {row['fast_ms']:.0f}ms vs wrapped {row['wrapped_ms']:.0f}ms "
        f"= {row['fast_vs_wrapped']:.2f}x < {FAST_TIER_FLOOR}x floor: the default "
        f"tier reverts to wrapped-episode and guarded-fast ships opt-in."
    )


def main() -> None:
    """Measurement/report pass: print all rows, optionally refresh baseline."""

    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-model", default=os.environ.get("TL_PERF_AB_DECODE_MODEL"))
    parser.add_argument("--steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--skip-small", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = {"context": load_context()}
    try:
        payload["sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        payload["sha"] = "unknown"

    if not args.skip_small:
        payload["small_capture"] = capture_overhead_rows(n=args.n)
    if args.decode_model:
        payload["decode_pair"] = {
            args.decode_model: decode_pair_rows(args.decode_model, steps=args.steps)
        }
    print(json.dumps(payload, indent=2))

    if args.write_baseline:
        if "small_capture" not in payload:
            raise SystemExit("--write-baseline requires the small-capture rows")
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"baseline written: {BASELINE_PATH}")


if __name__ == "__main__":
    main()
