"""Measure cold and warm public optimizer migration at fixed DAG cuts."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any, TypeVar
from unittest.mock import patch

from torchlens.split import (
    GraphPartition,
    OptimizerStateSnapshot,
    SplitFeatures,
    SplitRuntime,
    prepare_graph,
)
from torchlens.split.optimizers import _migration, get_optimizer_adapter

T = TypeVar("T")
_MODELS = ("mlp", "resnet18", "densenet121")


def _tensorflow_workload(name: str) -> tuple[Any, Any, Any, Any]:
    """Make an offline FP32 Keras workload with real model weights and gradients."""

    import tensorflow as tf

    tf.keras.utils.set_random_seed(7)
    with tf.device("/CPU:0"):
        if name == "mlp":
            model = tf.keras.Sequential(
                [
                    tf.keras.Input((16,)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(10),
                ]
            )
            inputs = tf.ones((2, 16))
        else:
            model = tf.keras.applications.DenseNet121(
                input_shape=(32, 32, 3), weights=None, classes=10, classifier_activation=None
            )
            inputs = tf.ones((2, 32, 32, 3))
        return model, inputs, tf.zeros((2, 10)), tf.keras.optimizers.Adam(0.001)


def _torch_workload() -> tuple[Any, Any, Any, Any]:
    """Make a CPU ResNet18 workload with locally initialized weights."""

    import torch
    from torchvision.models import resnet18

    torch.manual_seed(7)
    model = resnet18(weights=None, num_classes=10).eval()
    inputs = torch.ones(2, 3, 64, 64)
    return model, inputs, torch.zeros(2, 10), torch.optim.Adam(model.parameters(), lr=0.001)


def _fixed_cut(name: str) -> tuple[SplitRuntime, GraphPartition, dict[str, Any]]:
    """Capture once, train one native step, and select the fixed midpoint cut."""

    backend = "torch" if name == "resnet18" else "tf"
    model, inputs, target, optimizer = (
        _torch_workload() if backend == "torch" else _tensorflow_workload(name)
    )
    graph = prepare_graph(
        model,
        inputs,
        backend=backend,
        features=SplitFeatures(training=True, retain_trace=None if backend == "tf" else False),
    )
    dag = graph.compute_dag()
    source = dag.to_partition(())
    target_cut = dag.to_partition(call.call_id for call in dag.nodes[: len(dag.nodes) // 2])
    runtime = graph.materialize(graph.analyze(source))
    runtime.train_suffix(runtime.run_training_prefix(inputs), target, optimizer=optimizer)
    return runtime, target_cut, {"suffix": optimizer}


def _timed(samples: dict[str, float], key: str, action: Callable[..., T]) -> Callable[..., T]:
    """Accumulate wall time for a native phase that may run once per owner."""

    def run(*args: Any, **kwargs: Any) -> T:
        """Invoke the measured operation on the same entry point as normal migration."""

        start = perf_counter_ns()
        value = action(*args, **kwargs)
        samples[key] = samples.get(key, 0.0) + (perf_counter_ns() - start) / 1e6
        return value

    return run


def _snapshot_bytes(snapshot: OptimizerStateSnapshot) -> int:
    """Count unique CPU tensor payload bytes, including shared global state once."""

    arrays = {}
    for state in snapshot.optimizers.values():
        for slot in (
            *state.global_state,
            *(slot for slots in state.slots.values() for slot in slots),
        ):
            arrays[id(slot.payload)] = slot.payload.nbytes
    return sum(arrays.values())


def _measure(
    runtime: SplitRuntime, cut: GraphPartition, optimizers: Mapping[str, Any]
) -> tuple[dict[str, float], int, int]:
    """Measure public analysis, export, recut and restore with native phase timings."""

    samples: dict[str, float] = {}
    _timed(samples, "parameter_ownership_ms", runtime.parameter_ownership)()
    start = perf_counter_ns()
    plan = _timed(samples, "optimizer_plan_ms", runtime.plan_optimizer_state_migration)(
        cut, optimizers=optimizers
    )
    if not plan.supported:
        raise RuntimeError(f"Benchmark migration refused: {plan.unsupported_reasons!r}")
    snapshot = _timed(samples, "snapshot_export_ms", runtime.export_optimizer_state)(
        optimizers=optimizers, plan=plan
    )
    recut = _timed(samples, "repartition_ms", runtime.repartition)(cut)
    adapter = get_optimizer_adapter(runtime.adapter.name, optimizers["suffix"])
    with ExitStack() as stack:
        for obj, function, key in (
            (_migration, "_validate_header", "snapshot_validation_ms"),
            (_migration, "_validate_snapshot", "snapshot_validation_ms"),
            (adapter, "build_optimizer", "optimizer_construction_ms"),
            (adapter, "restore_state", "slot_restore_ms"),
        ):
            stack.enter_context(
                patch.object(obj, function, _timed(samples, key, getattr(obj, function)))
            )
        result = _timed(samples, "total_restore_ms", recut.restore_optimizer_state)(snapshot)
    samples["total_migration_ms"] = (perf_counter_ns() - start) / 1e6
    return samples, _snapshot_bytes(snapshot), result.restored_slots


def _benchmark(name: str, warmup: int, repeats: int) -> dict[str, Any]:
    """Return separate first-run and steady-state medians in milliseconds."""

    runtime, cut, optimizers = _fixed_cut(name)
    cold, payload_bytes, slots = _measure(runtime, cut, optimizers)
    for _ in range(warmup):
        _measure(runtime, cut, optimizers)
    hot = [_measure(runtime, cut, optimizers)[0] for _ in range(repeats)]
    return {
        "backend": runtime.adapter.name,
        "calls": len(runtime.compute_dag().nodes),
        "snapshot_payload_bytes": payload_bytes,
        "restored_slots": slots,
        "cold_ms": cold,
        "warm_median_ms": {key: median(row[key] for row in hot) for key in cold},
        "warm_samples_ms": hot,
    }


def main() -> None:
    """Write measured benchmark results for the requested model or the full matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*_MODELS, "all"), default="all")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    models = _MODELS if args.model == "all" else (args.model,)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": version("torch"),
        "device": "cpu",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "models": {},
    }
    if any(model != "resnet18" for model in models):
        try:
            report["tensorflow"] = version("tensorflow")
        except PackageNotFoundError:
            report["tensorflow"] = version("tensorflow-cpu")
        report["keras"] = version("keras")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in models:
        report["models"][name] = _benchmark(name, args.warmup, args.repeats)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"{name}: {report['models'][name]['warm_median_ms']['total_migration_ms']:.2f} ms warm",
            flush=True,
        )


if __name__ == "__main__":
    main()
