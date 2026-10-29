"""Independent-process repartition benchmark with uninstrumented latency samples.

Stage shares and Python metadata memory use separate runs, so profiling does not
inflate the reported wall times. Example: python benchmarks/split_repartition.py --output result.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from math import ceil
from pathlib import Path
from typing import Any
from unittest.mock import patch


def distribution(values: list[float]) -> dict[str, Any]:
    """Report every sample and a nearest-rank P95 without interpolation."""
    return {
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[ceil(0.95 * len(values)) - 1],
        "n": len(values),
        "samples_ms": values,
    }


def main() -> None:
    """Measure analysis, materialization, rotating recuts and post-training reuse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix-counts", type=int, nargs=3, default=[18, 34, 51])
    args = parser.parse_args()
    if min(args.repeats, args.batch_size) < 1:
        parser.error("repeats and batch-size must be positive")
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    from torchvision.models import resnet18

    import torchlens.split as split
    from torchlens.split import (
        GraphPartition,
        SplitFeatures,
        api,
        planner,
        runtime as runtime_module,
    )

    # The comparison path uses only GraphPartition on both revisions. All
    # reference-version handling is confined to this measurement driver.
    context = getattr(split, "context", runtime_module)
    from torchlens.split.adapters.torch import TorchSplitAdapter, _GeneratedSegmentBase
    from torchlens.split.runtime import SplitRuntime
    from torchlens.split.state import SegmentState

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        report = {"status": "not_run", "reason": "CUDA unavailable"}
    else:
        model = resnet18(weights=None).eval().to(args.device)
        x = torch.randn(args.batch_size, 3, 32, 32, device=args.device)

        def sync() -> None:
            if args.device == "cuda":
                torch.cuda.synchronize()

        def measure(
            action: Callable[[], Any], before: Callable[[], Any] | None = None
        ) -> dict[str, Any]:
            values = []
            for _ in range(args.repeats):
                if before:
                    before()
                sync()
                start = time.perf_counter()
                action()
                sync()
                values.append(1000 * (time.perf_counter() - start))
            return distribution(values)

        started = time.perf_counter()
        features = SplitFeatures(training=True, retain_trace=False)
        capture_phases: dict[str, float] = {}

        def capture_timer(name: str, action: Callable[..., Any]) -> Callable[..., Any]:
            """Record identical capture stages on both compared revisions."""

            def timed(*values: Any, **kwargs: Any) -> Any:
                phase_start = time.perf_counter()
                result = action(*values, **kwargs)
                capture_phases[name] = 1000 * (time.perf_counter() - phase_start)
                return result

            return timed

        with (
            patch.object(
                api,
                "capture_canonical_model",
                capture_timer("canonical_capture", api.capture_canonical_model),
            ),
            patch.object(
                api,
                "normalize_to_split_ir",
                capture_timer("normalization_and_batch_probe", api.normalize_to_split_ir),
            ),
        ):
            if hasattr(split, "prepare_graph"):
                graph = split.prepare_graph(model, x, features=features)
            else:
                graph = split.prepare(
                    model, x, split.SplitRequest(GraphPartition(frozenset()), features=features)
                )
        capture_ms = 1000 * (time.perf_counter() - started)
        nodes = graph.trace_graph.compute_nodes
        partitions = [
            GraphPartition(frozenset(n.canonical_id for n in nodes[:count]))
            for count in args.prefix_counts
        ]
        for partition in partitions:
            graph.analyze(partition)
        if len(set(partitions)) != 3:
            raise AssertionError("benchmark requires three distinct partitions")
        runtime = graph.materialize(graph.analyze(partitions[0]))

        def recut(partition: Any) -> Any:
            nonlocal runtime
            method = getattr(runtime, "repartition", None)
            runtime = (
                method(partition)
                if method is not None
                else runtime.materialize(runtime.analyze(partition))
            )
            return runtime

        def rotate() -> Any:
            nonlocal iteration
            partition = partitions[iteration % 3]
            iteration += 1
            return recut(partition)

        iteration = 0
        for partition in partitions:
            recut(partition)
        timings = {
            "cold_analysis": measure(
                lambda: runtime.analyze(partitions[0]), runtime.clear_analysis_cache
            ),
            "warm_analysis": measure(lambda: runtime.analyze(partitions[0])),
        }
        analysis = runtime.analyze(partitions[0])
        timings["cold_materialization"] = measure(
            lambda: runtime.materialize(analysis),
            getattr(graph.trace_graph, "_segment_blueprints", {}).clear,
        )
        timings["warm_materialization"] = measure(lambda: runtime.materialize(analysis))
        timings["same_partition_repartition"] = measure(lambda: recut(partitions[0]))
        timings["rotating_repartition"] = measure(rotate)
        timings["cold_repartition"] = measure(rotate, runtime.clear_analysis_cache)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_times, post_train = [], []
        for _ in range(min(args.repeats, 10)):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            boundary = runtime.run_training_prefix(x)
            _, grads = runtime.train_suffix(
                boundary, torch.zeros(args.batch_size, dtype=torch.long, device=args.device)
            )
            runtime.backward_prefix(boundary, grads)
            optimizer.step()
            sync()
            train_times.append(1000 * (time.perf_counter() - start))
            start = time.perf_counter()
            rotate()
            sync()
            post_train.append(1000 * (time.perf_counter() - start))
        timings["training_step"] = distribution(train_times)
        timings["post_training_repartition"] = distribution(post_train)
        # Exclusive phase accounting in a SEPARATE instrumented pass.
        phases: dict[str, list[float]] = defaultdict(list)
        stack: list[list[float]] = []

        def instrument(name: str, action: Callable[..., Any]) -> Callable[..., Any]:
            def timed(*a: Any, **kw: Any) -> Any:
                start = time.perf_counter()
                frame = [0.0]
                stack.append(frame)
                try:
                    return action(*a, **kw)
                finally:
                    elapsed = time.perf_counter() - start
                    stack.pop()
                    if stack:
                        stack[-1][0] += elapsed
                    phases[name].append(1000 * (elapsed - frame[0]))

            return timed

        targets = [
            (planner, "_validate_replay_dependencies", "validation"),
            (planner, "_partition_sets", "validation"),
            (planner, "_validate_partition_effects", "effects"),
            (planner, "_frontier_node_ids", "frontier"),
            (planner, "_boundary_metadata", "frontier"),
            (context, "lower_split_program", "lowering"),
            (context, "analyze_split_capabilities", "capability"),
            (TorchSplitAdapter, "build_segments", "segment_materialization"),
            (SplitRuntime, "_segment_state_snapshots", "state_inspection"),
            (_GeneratedSegmentBase, "bound_state_values", "state_binding"),
            (SplitRuntime, "_inherit_segment_state", "state_inheritance"),
            (SegmentState, "_replicate", "device_transfer"),
        ]
        with ExitStack() as scope:
            for owner, method, name in targets:
                scope.enter_context(
                    patch.object(owner, method, instrument(name, getattr(owner, method)))
                )
            for _ in range(args.repeats):
                runtime.clear_analysis_cache()
                instrument("overhead", rotate)()
        phase_totals = {name: sum(values) for name, values in phases.items()}
        total = sum(phase_totals.values())
        phase_report = {
            name: {
                **distribution(values),
                "exclusive_total_ms": phase_totals[name],
                "instrumented_share_percent": 100 * phase_totals[name] / total,
            }
            for name, values in phases.items()
        }
        phase_report.setdefault("device_transfer", {"status": "not_run", "n": 0})
        gc.collect()
        tracemalloc.start()
        for _ in range(args.repeats):
            rotate()
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        host_memory = None
        try:
            import psutil

            info = psutil.Process().memory_info()
            host_memory = {
                "retained_process_rss_bytes": info.rss,
                "peak_process_working_set_bytes": getattr(info, "peak_wset", None),
            }
        except ImportError:
            pass
        cache_info = runtime.analysis_cache_info()
        report = {
            "status": "measured",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": args.device,
            "model": "torchvision.resnet18(weights=None).eval()",
            "seed": args.seed,
            "batch": args.batch_size,
            "threads": 1,
            "capture_ms": capture_ms,
            "capture_phases_ms": capture_phases,
            "capture_scope": "capture_and_normalization"
            if hasattr(split, "prepare_graph")
            else "capture_normalization_and_empty_partition_runtime",
            "source": split.__file__,
            "timings": timings,
            "phases_separate_instrumented_run": phase_report,
            "cache": {
                **cache_info,
                "hit_rate": cache_info["hits"] / max(1, cache_info["hits"] + cache_info["misses"]),
            },
            "memory_separate_run": {
                "python_metadata_peak_bytes": peak,
                "python_metadata_retained_bytes": retained,
                "host_memory": host_memory,
                "cuda_peak_allocated": torch.cuda.max_memory_allocated()
                if args.device == "cuda"
                else None,
            },
            "device_transfer": "not_run: CPU-only"
            if not torch.cuda.is_available()
            else "not_run: same-device benchmark",
            "partition_ids": [p.as_boundary() for p in partitions],
            "prefix_counts": args.prefix_counts,
        }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
