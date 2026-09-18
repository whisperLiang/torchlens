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
import sys
import time
from contextlib import nullcontext
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


def main() -> None:
    """Capture once, measure representative replays, and compare native output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-grad", action="store_true")
    parser.add_argument("--retain-trace", action="store_true")
    args = parser.parse_args()
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
