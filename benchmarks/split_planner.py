"""Measure synthetic DAG planning with ordered native writes, without executing tensors.

Example: ``python benchmarks/split_planner.py --writes 250 --repeats 25``.
Run this same script with ``--source-root`` pointing at each compared checkout.
The workload is a normalized chain of add/ReLU-write pairs with a midpoint cut.
It measures uncached ``plan_split`` only: no capture, lowering, segment creation,
replay, training or CUDA work. It is not a model-execution speedup measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchlens.split.graph import SplitTraceGraph, SplitTraceNode


def _node(
    node_id: str,
    index: int,
    *,
    parent: str | None = None,
    target: Any = None,
    arguments: tuple[Any, ...] = (),
    is_input: bool = False,
    is_output: bool = False,
) -> SplitTraceNode:
    """Construct one normalized synthetic value with explicit replay dependencies."""

    from torchlens.intervention.types import CapturedArgTemplate
    from torchlens.split.graph import SplitTraceNode

    return SplitTraceNode(
        label=node_id,
        raw_label=node_id,
        canonical_id=node_id,
        backend="torch",
        raw_index=index,
        op_type=node_id.split("_", 1)[0],
        target=target,
        func_call_id=index,
        args_template=CapturedArgTemplate(args=arguments),
        kwargs_template=None,
        parents=() if parent is None else (parent,),
        children=(),
        output_ref=None,
        module_path=None,
        output_shape=(2, 4),
        symbolic_output_shape=None,
        dtype="torch.float32",
        requires_grad=True,
        output_container_path=(),
        output_container_spec=None,
        is_input=is_input,
        is_output=is_output,
        is_buffer=False,
        is_buffer_only_source=False,
        is_param_source=False,
        param_refs=(),
        replay_source_policy="constant",
        op=None,
    )


def _graph(writes: int) -> SplitTraceGraph:
    """Build a write-heavy linear DAG without capture or tensor allocation."""

    import torch

    from torchlens.intervention.types import LiteralValue
    from torchlens.split.graph import ReplayValueRef, SplitTraceGraph

    nodes = [_node("input", 0, is_input=True)]
    parent = "input"
    for index in range(writes):
        for name, target, arguments in (
            ("add", torch.add, (ReplayValueRef(parent), LiteralValue(1))),
            ("relu", torch.relu_, (ReplayValueRef(f"add_{index}"),)),
        ):
            node_id = f"{name}_{index}"
            nodes.append(
                _node(node_id, len(nodes), parent=parent, target=target, arguments=arguments)
            )
            parent = node_id
    nodes.append(_node("output", len(nodes), parent=parent, is_output=True))
    return SplitTraceGraph("torch", tuple(nodes), ("input",), ("output",), "synthetic-writes", 2)


def main() -> None:
    """Report cold planning separately from warmed graph metadata, in milliseconds."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--writes", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.writes < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("--writes/--repeats must be positive and --warmup nonnegative")
    sys.path.insert(0, str(args.source_root.resolve()))

    import torch

    import torchlens as tl
    from torchlens.split import GraphPartition, SplitRequest
    from torchlens.split.planner import plan_split

    graph = _graph(args.writes)
    prefix_count = 2 * (args.writes // 2)
    request = SplitRequest(
        GraphPartition(frozenset(node.canonical_id for node in graph.nodes[1 : prefix_count + 1]))
    )
    samples = []
    for _ in range(1 + args.warmup + args.repeats):
        start = time.perf_counter()
        plan = plan_split(graph, request)
        samples.append((time.perf_counter() - start) * 1000)
        assert len(plan.ordered_mutation_node_ids) == args.writes
        assert len(plan.boundary_node_ids) == 1
    steady = samples[1 + args.warmup :]
    print(
        json.dumps(
            {
                "scope": "synthetic uncached plan_split; no tensor execution",
                "source_root": str(args.source_root.resolve()),
                "torchlens_module": tl.__file__,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "compute_nodes": 2 * args.writes,
                "writes": args.writes,
                "prefix_compute_nodes": prefix_count,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "cold_ms": samples[0],
                "median_ms": statistics.median(steady),
                "p95_ms": sorted(steady)[ceil(0.95 * len(steady)) - 1],
                "samples_ms": steady,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
