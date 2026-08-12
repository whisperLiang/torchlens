"""P5 per-scenario capture CPU A/B: legacy vs decomposed producer.

STAGED for the P5 campaign (do not run as part of fast gates — ~minutes).
Paired ``time.process_time`` CPU comparison per parity scenario, interleaved
A/B/A/B to cancel drift, reporting per-scenario ratios and a pooled verdict.
cProfile is deliberately NOT used (blind to allocator/import/C-level and
overstates generator/hook cost 3-6x — speedening lesson).

Usage (from the worktree root, gates venv on PATH):

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests \
        python -m producer_parity.bench_p5_capture_ab [--reps 15] [--scenario NAME]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any

_PRODUCER_ENV = "TORCHLENS_CAPTURE_PRODUCER"


def _run_once(scenario: Any, producer: str) -> float:
    os.environ[_PRODUCER_ENV] = producer
    try:
        model, inputs = scenario.build()
        start = time.process_time()
        captured = scenario.capture(model, inputs)
        elapsed = time.process_time() - start
        del captured
        return elapsed
    finally:
        os.environ.pop(_PRODUCER_ENV, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=15, help="paired reps per scenario")
    parser.add_argument("--scenario", default=None, help="single scenario name")
    args = parser.parse_args()

    from producer_parity._models import SCENARIOS, scenario_by_name

    scenarios = [scenario_by_name(args.scenario)] if args.scenario else list(SCENARIOS)
    worst_ratio = 0.0
    print(f"{'scenario':<28} {'legacy(ms)':>12} {'decomp(ms)':>12} {'ratio':>7}")
    for scenario in scenarios:
        # One warm-up per leg (wrapping, imports, allocator pools).
        _run_once(scenario, "legacy")
        _run_once(scenario, "decomposed")
        legacy_times: list[float] = []
        decomposed_times: list[float] = []
        for _ in range(args.reps):
            legacy_times.append(_run_once(scenario, "legacy"))
            decomposed_times.append(_run_once(scenario, "decomposed"))
        legacy_median = statistics.median(legacy_times)
        decomposed_median = statistics.median(decomposed_times)
        ratio = decomposed_median / legacy_median if legacy_median else float("inf")
        worst_ratio = max(worst_ratio, ratio)
        print(
            f"{scenario.name:<28} {legacy_median * 1000:>12.2f} "
            f"{decomposed_median * 1000:>12.2f} {ratio:>7.3f}"
        )
    print(f"\nworst decomposed/legacy median ratio: {worst_ratio:.3f}")
    # Budget verdict per DoR section 8: the decomposed producer must not
    # regress capture CPU beyond noise; investigate anything over ~5%.
    return 0 if worst_ratio <= 1.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
