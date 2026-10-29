"""Compare diagnostic/compact training and compact inference in fresh processes.

Run each mode in a separate process, for example::

    python benchmarks/split_training_compact.py --device cuda --mode diagnostic-training
    python benchmarks/split_training_compact.py --device cuda --mode compact-training
    python benchmarks/split_training_compact.py --device cuda --mode compact-inference

Use the same script with ``--source-root`` pointing to old/new checkouts. An old
checkout refusing compact training reports ``unsupported`` rather than substituting
diagnostic training. No downloads, optimizer steps or model weights are required.
CUDA allocated and reserved memory are distinct; RSS includes allocator caches.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import weakref
from math import ceil
from pathlib import Path
from typing import Any
from unittest.mock import patch


def _rss_bytes() -> int | None:
    """Return current host RSS when a platform provider is available."""

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    return None


def main() -> None:
    """Measure preparation stages, retained storage and warmed training/replay."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--mode",
        choices=("diagnostic-training", "compact-training", "compact-inference"),
        default="compact-training",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--recuts", type=int, default=32)
    args = parser.parse_args()
    if (
        min(args.width, args.depth, args.batch_size, args.repeats) < 1
        or min(args.warmup, args.recuts) < 0
    ):
        parser.error("width, depth, batch-size and repeats must be positive; warmup nonnegative")
    sys.path.insert(0, str(args.source_root.resolve()))

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens.split import api
    from torchlens.split.errors import SplitUnsupportedError

    torch.set_num_threads(1)
    torch.manual_seed(42)
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; this is not a passing measurement")
    training = args.mode != "compact-inference"

    class Model(nn.Module):
        """Repeated affine layers with a shape-preserving residual connection."""

        def __init__(self) -> None:
            """Construct the requested model without downloading state."""
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(args.width, args.width) for _ in range(args.depth)]
            )

        def forward(self, x: Any) -> Any:
            """Evaluate the residual MLP."""
            residual = x
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x + residual

    def sync() -> None:
        """Synchronize device work before measuring wall-clock time."""
        if cuda:
            torch.cuda.synchronize()

    def memory() -> dict[str, Any]:
        """Read retained and peak allocation counters independently."""
        sync()
        result: dict[str, Any] = {"rss_bytes": _rss_bytes()}
        if cuda:
            result.update(
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        return result

    model = Model().to(args.device).train(training)
    inputs = torch.randn(args.batch_size, args.width, device=args.device, requires_grad=training)
    targets = torch.randn_like(inputs)
    request = tl.split.SplitRequest(
        point=tl.split.after(f"layers.{max(args.depth // 2 - 1, 0)}"),
        features=tl.split.SplitFeatures(
            training=training, retain_trace=args.mode == "diagnostic-training"
        ),
    )
    report: dict[str, Any] = {
        "mode": args.mode,
        "device": args.device,
        "torch_version": torch.__version__,
        "source": str(tl.__file__),
        "batch_size": args.batch_size,
        "width": args.width,
        "depth": args.depth,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "before_prepare": memory(),
    }
    capture_refs: list[Any] = []

    def instrument(name: str, action: Any) -> Any:
        """Wrap a preparation stage without retaining its returned objects."""

        def measured(*values: Any, **kwargs: Any) -> Any:
            """Record stage wall time and memory at completion."""
            sync()
            started = time.perf_counter()
            result = action(*values, **kwargs)
            sync()
            report[name] = {"ms": (time.perf_counter() - started) * 1000, **memory()}
            if name == "capture":
                capture_refs.append(weakref.ref(result[0]))
            return result

        return measured

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    sync()
    started = time.perf_counter()
    try:
        with (
            patch.object(
                api, "capture_canonical_model", instrument("capture", api.capture_canonical_model)
            ),
            patch.object(
                api, "normalize_to_split_ir", instrument("normalize", api.normalize_to_split_ir)
            ),
            patch.object(
                api, "execute_split_runtime", instrument("materialize", api.execute_split_runtime)
            ),
        ):
            runtime = tl.split.prepare(model, inputs, request)
    except SplitUnsupportedError as error:
        report.update(status="unsupported", error=str(error))
        print(json.dumps(report))
        return
    sync()
    report["prepare_ms"] = (time.perf_counter() - started) * 1000
    gc.collect()
    report.update(
        status="ok",
        retains_trace=runtime.retains_trace,
        capture_alive=any(reference() is not None for reference in capture_refs),
        after_prepare=memory(),
    )

    def step() -> None:
        """Execute forward/backward, explicitly without an optimizer step."""
        model.zero_grad(set_to_none=True)
        inputs.grad = None
        if training:
            boundary = runtime.run_training_prefix(inputs)
            _, gradients = runtime.train_suffix(boundary, targets)
            runtime.backward_prefix(boundary, gradients)
        else:
            with torch.inference_mode():
                runtime.replay(inputs)

    with torch.no_grad():
        expected = model(inputs)
        actual = runtime.replay(inputs)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        report["forward_max_abs_error"] = float((actual - expected).abs().max())
        del actual, expected
    if training:
        model.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        expected_grads = [parameter.grad.clone() for parameter in model.parameters()]
        expected_input_grad = inputs.grad.clone()
        step()
        for parameter, gradient in zip(model.parameters(), expected_grads, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(inputs.grad, expected_input_grad, atol=1e-5, rtol=1e-4)
        report["parameter_gradient_max_abs_error"] = max(
            float((parameter.grad - gradient).abs().max())
            for parameter, gradient in zip(model.parameters(), expected_grads, strict=True)
        )
        del expected_grads, expected_input_grad
    for _ in range(args.warmup):
        step()
    gc.collect()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        sync()
        started = time.perf_counter()
        step()
        sync()
        samples.append((time.perf_counter() - started) * 1000)
    report["step"] = {
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[ceil(0.95 * len(samples)) - 1],
        "samples_ms": samples,
        **memory(),
    }
    model.zero_grad(set_to_none=True)
    inputs.grad = None
    gc.collect()
    report["after_steps"] = memory()
    recut_samples = []
    recut_refs = []
    recut_snapshots = []
    nodes = runtime.trace_graph.compute_nodes
    for index in range(args.recuts):
        point = tl.split.after(nodes[index % len(nodes)].canonical_id)
        sync()
        started = time.perf_counter()
        recut = runtime.at(point)
        sync()
        recut_samples.append((time.perf_counter() - started) * 1000)
        recut_refs.append(weakref.ref(recut))
        del recut
        if (index + 1) % 16 == 0 or index + 1 == args.recuts:
            gc.collect()
            recut_snapshots.append({"count": index + 1, **memory()})
    gc.collect()
    report["recuts"] = {
        "count": args.recuts,
        "median_ms": statistics.median(recut_samples) if recut_samples else None,
        "p95_ms": sorted(recut_samples)[ceil(0.95 * len(recut_samples)) - 1]
        if recut_samples
        else None,
        "samples_ms": recut_samples,
        "retained_child_runtimes": sum(reference() is not None for reference in recut_refs),
        "memory_snapshots": recut_snapshots,
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
