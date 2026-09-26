"""Measure RF-DETR split preparation/replay in one fresh CPU or CUDA process.

Run with ``python benchmarks/split_memory.py --device cuda``. Use
``--source-root`` to compare an independent checkout with the same environment
and the current ``SplitRuntime.retains_trace`` API.
RSS includes imported libraries and host allocator caches; CUDA allocated and
reserved bytes are reported separately. No model downloads are allowed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any


def _rss_bytes() -> int:
    """Read this Linux process's current resident memory."""

    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("This benchmark requires Linux /proc/self/status.")


def _tensor_leaves(value: Any, torch: Any) -> list[Any]:
    """Flatten model outputs for numerical comparison and storage accounting."""

    if isinstance(value, torch.Tensor):
        return [value]
    children = value.values() if isinstance(value, dict) else value
    if isinstance(value, (dict, list, tuple)):
        return [leaf for item in children for leaf in _tensor_leaves(item, torch)]
    return []


def _storage_bytes(values: list[Any]) -> int:
    """Count each physical tensor storage once, including aliased views."""

    storages = {}
    for value in values:
        storage = value.untyped_storage()
        storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def _latency_samples(
    workloads: dict[str, Any], *, torch: Any, cuda: bool, runs: int
) -> dict[str, Any]:
    """Time warmed workloads in interleaved rounds with device synchronization.

    Parameters
    ----------
    workloads
        Named zero-argument execution paths.
    torch
        Active PyTorch module.
    cuda
        Whether GPU work must be synchronized around each sample.
    runs
        Number of timed samples per path.

    Returns
    -------
    dict
        Per-path samples and median/p5/p95 wall-clock milliseconds.
    """

    def sync() -> None:
        """Drain pending GPU kernels before reading the host clock."""

        if cuda:
            torch.cuda.synchronize()

    for _ in range(3):
        for workload in workloads.values():
            workload()
    sync()
    gc.collect()
    samples: dict[str, list[float]] = {name: [] for name in workloads}
    gc.disable()
    try:
        for _ in range(runs):
            for name, workload in workloads.items():
                sync()
                started = time.perf_counter()
                output = workload()
                sync()
                samples[name].append((time.perf_counter() - started) * 1000)
                del output
    finally:
        gc.enable()
    result: dict[str, Any] = {}
    for name, values in samples.items():
        ordered = sorted(values)
        result[name] = {
            "median_ms": statistics.median(values),
            "p5_ms": ordered[int(0.05 * (runs - 1))],
            "p95_ms": ordered[int(0.95 * (runs - 1))],
            "samples_ms": values,
        }
    return result


def main() -> None:
    """Capture once, measure representative replays, and compare native output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-grad", action="store_true")
    parser.add_argument("--retain-trace", action="store_true")
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=0,
        help="Measure warmed forward/prefix/suffix/replay latency at the 50%% split point.",
    )
    parser.add_argument(
        "--compare-replay-fingerprint",
        action="store_true",
        help="Also time replay with the prior one-fingerprint path.",
    )
    parser.add_argument(
        "--compare-trusted-boundary",
        action="store_true",
        help="Also time public prefix/suffix calls with explicit state checks disabled.",
    )
    args = parser.parse_args()
    if args.latency_runs < 0:
        parser.error("--latency-runs must be nonnegative")
    if args.compare_replay_fingerprint and not args.latency_runs:
        parser.error("--compare-replay-fingerprint requires --latency-runs")
    if args.compare_trusted_boundary and not args.latency_runs:
        parser.error("--compare-trusted-boundary requires --latency-runs")
    sys.path.insert(0, str(args.source_root.resolve()))
    weights_dir = Path(
        os.environ.setdefault(
            "RF_HOME", str(Path.home() / ".cache" / "torchlens" / "models" / "rfdetr")
        )
    )
    if not (weights_dir / "rf-detr-nano.pth").is_file():
        raise FileNotFoundError("Cache rf-detr-nano.pth before running this benchmark.")

    import torch
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import NestedTensor

    import torchlens as tl

    torch.set_num_threads(1)
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.init()

    def snapshot(phase: str, **extra: Any) -> None:
        """Print byte-valued measurements after synchronizing this process."""

        if cuda:
            torch.cuda.synchronize()
        row = {
            "phase": phase,
            "device": args.device,
            "capture_grad": args.capture_grad,
            "retain_trace_requested": args.retain_trace,
            "rss_bytes": _rss_bytes(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            **extra,
        }
        if cuda:
            row.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
        print(json.dumps(row), flush=True)

    class TensorModel(torch.nn.Module):
        """Match the RF-DETR real-model test's tensor-only input ABI."""

        def __init__(self, core: torch.nn.Module) -> None:
            """Keep the official detection core unchanged."""

            super().__init__()
            self.core = core

        def forward(self, images: torch.Tensor) -> Any:
            """Construct the all-valid mask used by the split tests."""

            mask = torch.zeros(
                (images.shape[0], images.shape[2], images.shape[3]),
                device=images.device,
                dtype=torch.bool,
            )
            return self.core(NestedTensor(images, mask))

    snapshot("imported", torch_version=torch.__version__, source=str(tl.__file__))
    detector = RFDETRNano()
    model = TensorModel(detector.model.model).to(args.device).eval()
    inputs = torch.zeros(2, 3, 384, 384, device=args.device)
    gc.collect()
    snapshot("model", parameter_bytes=_storage_bytes(list(model.parameters())))
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with nullcontext() if args.capture_grad else torch.no_grad():
        features = (
            tl.split.SplitFeatures(retain_trace=True)
            if args.retain_trace
            else tl.split.SplitFeatures()
        )
        runtime = tl.split.prepare(
            model,
            inputs,
            tl.split.SplitRequest(point=tl.split.percent(50), backend="torch", features=features),
        )
    gc.collect()
    retained_trace = runtime.retains_trace
    snapshot(
        "prepared",
        elapsed_seconds=time.perf_counter() - started,
        compute_nodes=len(runtime.trace_graph.compute_nodes),
        batch_validation=runtime.batch_validation,
        retains_trace=retained_trace,
        output_storage_bytes=_storage_bytes(
            [op.out for op in runtime.trace if isinstance(op.out, torch.Tensor)]
            if retained_trace
            else []
        ),
    )
    replay_input = torch.zeros(3, 3, 384, 384, device=args.device)
    with torch.inference_mode():
        expected = model(replay_input)
        points = (
            tl.split.before(runtime.trace_graph.compute_nodes[0].canonical_id),
            tl.split.percent(50),
            tl.split.after(runtime.trace_graph.compute_nodes[-1].canonical_id),
        )
        for point in points:
            replay = runtime.at(point)
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            actual = replay.replay(replay_input)
            for left, right in zip(
                _tensor_leaves(actual, torch), _tensor_leaves(expected, torch), strict=True
            ):
                torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-3)
            snapshot(
                "replay", point=point.as_boundary(), elapsed_seconds=time.perf_counter() - started
            )
            del actual, replay
        if args.latency_runs:
            latency_input = inputs
            expected_latency = model(latency_input)
            latency_boundary = runtime.run_prefix(latency_input)
            actual_latency = runtime.run_suffix(latency_boundary)
            expected_leaves = _tensor_leaves(expected_latency, torch)
            actual_leaves = _tensor_leaves(actual_latency, torch)
            if len(expected_leaves) != len(actual_leaves):
                raise AssertionError("Split output tensor count differs from native forward.")
            for left, right in zip(actual_leaves, expected_leaves, strict=True):
                torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-3)
            workloads = {
                "forward": partial(model, latency_input),
                "prefix": partial(runtime.run_prefix, latency_input),
                "suffix": partial(runtime.run_suffix, latency_boundary),
                "replay": partial(runtime.replay, latency_input),
            }
            if args.compare_trusted_boundary:
                trusted_boundary = runtime.run_prefix(latency_input, check_state=False)
                trusted_output = runtime.run_suffix(trusted_boundary, check_state=False)
                for left, right in zip(
                    _tensor_leaves(trusted_output, torch), expected_leaves, strict=True
                ):
                    torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-3)
                workloads["prefix_trusted"] = partial(
                    runtime.run_prefix, latency_input, check_state=False
                )
                workloads["suffix_trusted"] = partial(
                    runtime.run_suffix, trusted_boundary, check_state=False
                )
            if args.compare_replay_fingerprint:

                def replay_with_state_fingerprint(target_runtime: Any = runtime) -> Any:
                    """Run the previous replay path with one state hash."""

                    boundary = target_runtime.run_prefix(latency_input)
                    target_runtime.validate_boundary(boundary, validate_state=False)
                    boundary = target_runtime._transport_boundary(
                        boundary, target_runtime.placement.suffix
                    )
                    return target_runtime.segments.suffix(boundary)

                workloads["replay_with_state_fingerprint"] = replay_with_state_fingerprint
            timings = _latency_samples(workloads, torch=torch, cuda=cuda, runs=args.latency_runs)
            snapshot(
                "latency",
                batch_size=int(latency_input.shape[0]),
                split_point=runtime.request.boundary,
                output_tensor_leaves=len(expected_leaves),
                max_abs_error=max(
                    float((left - right).abs().max())
                    for left, right in zip(actual_leaves, expected_leaves, strict=True)
                ),
                runs=args.latency_runs,
                timings=timings,
            )
    del expected, replay_input
    if retained_trace:
        runtime.trace.cleanup()
    del runtime
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    snapshot("cleaned")


if __name__ == "__main__":
    main()
