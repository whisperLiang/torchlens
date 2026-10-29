"""Strict isolated real-model validation of sequential and noncontiguous partitions.

Run: python benchmarks/split_graph_partition.py --model resnet18 --training --output result.json
No fallback between partition kinds. RF-DETR requires a trusted local checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch


def tensor_leaves(value: Any) -> list[Any]:
    """Flatten native tensor structures without dropping integer or nonfinite leaves."""
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in tensor_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensor_leaves(item)]
    return []


def finite_close(actual: Any, expected: Any, *, atol: float = 1e-4) -> float:
    """Reject nonfinite values, then compare structure, dtype, shape and numeric values."""
    import torch

    for value in (actual, expected):
        if any(not bool(torch.isfinite(tensor).all()) for tensor in tensor_leaves(value)):
            raise AssertionError("nonfinite values cannot attest equivalence")
    if (actual is None) != (expected is None):
        raise AssertionError(
            f"gradient presence mismatch: {type(actual).__name__}, {type(expected).__name__}"
        )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-3, check_device=False)
    return max(
        (
            float((a.detach().cpu() - b.detach().cpu()).abs().max())
            for a, b in zip(tensor_leaves(actual), tensor_leaves(expected), strict=True)
            if a.numel() and a.dtype != torch.bool
        ),
        default=0.0,
    )


def scalar_loss(output: Any, targets: Any = None) -> Any:
    """Use every floating output branch in one differentiable scalar objective."""
    tensors = [t for t in tensor_leaves(output) if t.is_floating_point() and t.requires_grad]
    if not tensors:
        raise AssertionError("no differentiable output")
    return sum(t.square().mean() for t in tensors)


def noncontiguous(graph: Any, selected: frozenset[str]) -> bool:
    """Verify nonempty sides and an inversion against the ORIGINAL compute order."""
    ordered = [node.canonical_id for node in graph.compute_nodes]
    return (
        bool(selected)
        and len(selected) < len(ordered)
        and selected != frozenset(ordered[: len(selected)])
    )


def candidates(graph: Any, kind: str) -> list[Any]:
    """Generate atomic ancestor closures or topological prefixes without exponential search."""
    from torchlens.split import GraphPartition

    compute = graph.compute_nodes
    compute_ids = frozenset(n.canonical_id for n in compute)
    call_by_id = graph.replay_call_by_output_id
    indexes = sorted(range(len(compute)), key=lambda i: (abs(i - len(compute) // 2), i))
    result, seen = [], set()
    for index in indexes:
        selected = set()
        pending = (
            [compute[index].canonical_id]
            if kind == "noncontiguous_dag"
            else [n.canonical_id for n in compute[: index + 1]]
        )
        while pending:
            node_id = pending.pop()
            if node_id in selected:
                continue
            selected.add(node_id)
            pending.extend(
                graph.node_id_by_alias.get(parent, parent)
                for parent in graph.node_by_id[node_id].parents
            )
            call = call_by_id.get(node_id)
            if call is not None:
                pending.extend(call.output_node_ids)
        ids = frozenset(selected & compute_ids)
        if not ids or len(ids) == len(compute) or ids in seen:
            continue
        seen.add(ids)
        if noncontiguous(graph, ids) != (kind == "noncontiguous_dag"):
            continue
        result.append(GraphPartition(ids))
    return result


def load_model(name: str, weights: Path | None) -> tuple[Any, tuple[int, ...], dict[str, Any]]:
    """Load official architectures with the original benchmark configurations."""
    import torch
    from torch import nn

    if name == "resnet18":
        from torchvision.models import resnet18

        return (
            resnet18(weights=None, num_classes=10),
            (3, 64, 64),
            {"weights": "random initialization", "num_classes": 10},
        )
    if name == "transformer":
        from transformers import DistilBertConfig, DistilBertModel

        config = DistilBertConfig(
            vocab_size=128,
            n_layers=2,
            dim=32,
            hidden_dim=64,
            n_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
        )

        class EmbeddingInputs(nn.Module):
            """Exercise the official attention stack with differentiable embedding inputs."""

            def __init__(self) -> None:
                super().__init__()
                self.core = DistilBertModel(config)

            def forward(self, x: Any) -> Any:
                return self.core(inputs_embeds=x, output_hidden_states=True, return_dict=True)

        return (
            EmbeddingInputs(),
            (12, 32),
            {
                "weights": "random initialization",
                "config": config.to_dict(),
                "input": "inputs_embeds; includes nested hidden_states",
            },
        )
    if name == "yolov8":
        from ultralytics import YOLO

        return (
            YOLO("yolov8n.yaml").model,
            (3, 160, 160),
            {"weights": "random initialization", "architecture": "yolov8n.yaml"},
        )
    if weights is None or not weights.is_file():
        raise FileNotFoundError(
            "RF-DETR requires --weights with an existing trusted local checkpoint"
        )
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import NestedTensor

    class Images(nn.Module):
        """Supply a valid spatial mask to the unchanged official detection core."""

        def __init__(self) -> None:
            super().__init__()
            self.core = RFDETRNano(pretrain_weights=str(weights)).model.model

        def forward(self, x: Any) -> Any:
            return self.core(
                NestedTensor(
                    x,
                    torch.zeros(
                        (x.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=torch.bool
                    ),
                )
            )

    return Images(), (3, 384, 384), {"weights": str(weights.resolve())}


def training_step(
    runtime: Any, reference: Any, x: Any, actual_opt: Any, expected_opt: Any
) -> dict[str, Any]:
    """Compare independent native and split AD graphs, including connected frontier gradients."""
    import torch
    from torch.overrides import TorchFunctionMode

    from torchlens import _state

    actual_opt.zero_grad(set_to_none=True)
    expected_opt.zero_grad(set_to_none=True)
    ref_x = x.detach().clone().requires_grad_(True)
    observed_gradients: dict[str, Any] = {}
    attached: set[str] = set()
    graph = runtime.trace_graph
    boundary_ids = {value: key for key, value in runtime.plan.boundary_bindings.items()}
    originals, _ = _state.wrap_epoch_ledgers()
    calls_by_target: dict[Any, list[Any]] = {}
    for call in graph.replay_calls:
        target = next(
            (
                graph.node_by_id[n].target
                for n in call.output_node_ids
                if graph.node_by_id[n].target is not None
            ),
            None,
        )
        if target is not None:
            calls_by_target.setdefault(target, []).append(call)
    targets_by_type: dict[str, set[Any]] = {}
    for node in graph.compute_nodes:
        if node.target is not None:
            targets_by_type.setdefault(node.op_type, set()).add(node.target)
    counts: dict[Any, int] = {}

    def attach(tensor: Any, key: str) -> None:
        if tensor.requires_grad:
            attached.add(key)

            def save_gradient(gradient: Any) -> None:
                observed_gradients[key] = gradient.detach().clone()

            tensor.register_hook(save_gradient)

    for node_id, key in boundary_ids.items():
        if graph.node_by_id[node_id].is_input:
            attach(ref_x, key)

    class ObserveFrontier(TorchFunctionMode):
        """Observe native outputs by exact captured target and occurrence, without replay."""

        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            result = func(*args, **(kwargs or {}))
            target = originals.get(id(func), func)
            calls = calls_by_target.get(target)
            if calls is None:
                # TorchFunctionMode sees some functional facades while capture
                # records their native builtin. Align only an unambiguous target
                # family, preserving native occurrence order and output arity.
                family = str(getattr(func, "__name__", "")).replace("_", "")
                targets = targets_by_type.get(family, set())
                if len(targets) != 1:
                    return result
                target = next(iter(targets))
                calls = calls_by_target[target]
            index = counts.get(target, 0)
            counts[target] = index + 1
            if index >= len(calls):
                raise AssertionError(f"native reference call count changed: {func}")
            call = calls[index]
            outputs = tensor_leaves(result)
            if len(outputs) != len(call.output_node_ids):
                raise AssertionError("native reference output count changed")
            for tensor, node_id in zip(outputs, call.output_node_ids, strict=True):
                if node_id in boundary_ids:
                    attach(tensor, boundary_ids[node_id])
            return result

    torch.manual_seed(317)
    with ObserveFrontier():
        expected_output = reference(ref_x)
    expected_loss = scalar_loss(expected_output)
    expected_loss.backward()
    torch.manual_seed(317)
    boundary = runtime.run_training_prefix(x)
    for tensor in boundary.tensors.values():
        if tensor.requires_grad:
            tensor.retain_grad()
    actual_loss, gradients = runtime.train_suffix(boundary, None, loss_fn=scalar_loss)
    runtime.backward_prefix(boundary, gradients)
    loss_error = finite_close(actual_loss, expected_loss)
    input_error = finite_close(x.grad, ref_x.grad)
    gradient_error = max(
        (
            finite_close(a.grad, b.grad)
            for a, b in zip(runtime.model.parameters(), reference.parameters(), strict=True)
        ),
        default=0.0,
    )
    required = {key for key, tensor in boundary.tensors.items() if tensor.requires_grad}
    if not required <= attached:
        raise AssertionError(f"reference frontier hook coverage missing: {required - attached}")
    frontier_error = max(
        (finite_close(boundary.tensors[key].grad, observed_gradients.get(key)) for key in required),
        default=0.0,
    )
    actual_opt.step()
    expected_opt.step()
    update_error = max(
        (
            finite_close(a, b)
            for a, b in zip(runtime.model.parameters(), reference.parameters(), strict=True)
        ),
        default=0.0,
    )
    optimizer_error = max(
        (
            finite_close(actual_opt.state[a], expected_opt.state[b])
            for a, b in zip(runtime.model.parameters(), reference.parameters(), strict=True)
        ),
        default=0.0,
    )
    buffer_error = max(
        (
            finite_close(a, b)
            for a, b in zip(runtime.model.buffers(), reference.buffers(), strict=True)
        ),
        default=0.0,
    )
    return {
        "loss": float(actual_loss.detach()),
        "loss_error": loss_error,
        "input_gradient_error": input_error,
        "parameter_gradient_error": gradient_error,
        "connected_boundary_gradient_error": frontier_error,
        "parameter_update_error": update_error,
        "optimizer_state_error": optimizer_error,
        "buffer_error": buffer_error,
    }


def validate_kind(
    context: Any,
    reference: Any,
    initial: dict[str, Any],
    dimensions: tuple[int, ...],
    kind: str,
    args: Any,
) -> dict[str, Any]:
    """Independently report one partition kind; never substitute sequential success for DAG success."""
    import torch

    from torchlens.split.errors import SplitError

    row = {
        "partition_kind": kind,
        "partition_valid": False,
        "noncontiguous_verified": False,
        "forward_passed": False,
        "backward_passed": False,
        "optimizer_update_passed": False,
        "dynamic_batch_passed": False,
        "recut_passed": False,
        "cross_device_tested": False,
        "attempts": [],
        "batches": [],
        "training_steps": [],
    }
    accepted = []
    for partition in candidates(context.trace_graph, kind):
        try:
            analysis = context.analyze(partition)
            if not any(n in partition.prefix_node_ids for n in analysis.plan.boundary_node_ids):
                raise AssertionError("frontier lacks a true compute dependency")
            context.materialize(analysis)
            accepted.append(analysis)
            row["attempts"].append({"partition": partition.as_boundary(), "status": "accepted"})
            if len(accepted) == 3:
                break
        except SplitError as error:
            row["attempts"].append(
                {
                    "partition": partition.as_boundary(),
                    "status": "unsupported",
                    "reason": getattr(getattr(error, "context", None), "reason", error.code),
                    "error": str(error),
                }
            )
    if not accepted:
        return {
            **row,
            "status": "unsupported",
            "reason": "no legal partition of this kind",
            "candidate_count": len(row["attempts"]),
        }
    row.update(
        partition_valid=True,
        noncontiguous_verified=kind == "noncontiguous_dag",
        distinct_partitions=len(accepted),
        partition_identities=[a.request.boundary for a in accepted],
    )
    context.model.load_state_dict(initial)
    reference.load_state_dict(initial)
    runtime = context.materialize(accepted[0])
    for batch in args.batches:
        x = torch.randn(batch, *dimensions, device=args.device)
        with torch.no_grad():
            torch.manual_seed(317)
            expected = reference(x)
            torch.manual_seed(317)
            actual = runtime.replay(x)
            error = finite_close(actual, expected)
            finite_close(runtime.replay(x), expected)
        profile = context.graph_profile(batch_size=batch)
        layout = runtime.boundary_layout(batch_size=batch)
        row["batches"].append(
            {
                "batch": batch,
                "status": "passed",
                "forward_error": error,
                "forward_flops": profile.forward_flops,
                "backward_flops_estimate": profile.backward_flops,
                "boundary_bytes": layout.forward_payload_bytes,
            }
        )
    row["forward_passed"] = True
    row["dynamic_batch_passed"] = {1, 2, 4, 8} <= set(args.batches)
    if args.training:
        optimizers = {
            "sgd": lambda p: torch.optim.SGD(p, lr=0.01),
            "momentum": lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
            "adam": lambda p: torch.optim.Adam(p, lr=0.001),
        }
        for name in args.optimizers:
            context.model.load_state_dict(initial)
            reference.load_state_dict(initial)
            runtime = context.materialize(accepted[0])
            actual_opt, expected_opt = (
                optimizers[name](context.model.parameters()),
                optimizers[name](reference.parameters()),
            )
            sequence = (0, 1, 2, 0) if len(accepted) == 3 else (0,) * 4
            for index, batch in enumerate(args.batches):
                analysis = accepted[sequence[index % 4]]
                start = time.perf_counter()
                migration = runtime.plan_state_migration(analysis.request.partition)
                migration_ms = 1000 * (time.perf_counter() - start)
                start = time.perf_counter()
                runtime = runtime.repartition(analysis.request.partition)
                recut_ms = 1000 * (time.perf_counter() - start)
                if runtime.trace_graph is not context.trace_graph:
                    raise AssertionError("recut replaced the captured graph")
                x = torch.randn(batch, *dimensions, device=args.device, requires_grad=True)
                start = time.perf_counter()
                errors = training_step(runtime, reference, x, actual_opt, expected_opt)
                row["training_steps"].append(
                    {
                        "optimizer": name,
                        "batch": batch,
                        "partition": runtime.request.boundary,
                        "prefix_nodes": len(runtime.request.partition.prefix_node_ids),
                        "suffix_nodes": len(context.trace_graph.compute_nodes)
                        - len(runtime.request.partition.prefix_node_ids),
                        "boundary_bytes": runtime.boundary_layout(
                            batch_size=batch, training=True
                        ).forward_payload_bytes,
                        "migration_plan_ms": migration_ms,
                        "migration_identity_preserved": all(
                            e.identity_preserved is not False for e in migration.entries
                        ),
                        "repartition_ms": recut_ms,
                        "training_ms": 1000 * (time.perf_counter() - start),
                        **errors,
                    }
                )
                gc.collect()
                try:
                    import psutil

                    row["training_steps"][-1]["rss_after_gc_bytes"] = (
                        psutil.Process().memory_info().rss
                    )
                except ImportError:
                    row["training_steps"][-1]["rss_after_gc_bytes"] = None
        row.update(
            backward_passed=True,
            optimizer_update_passed=True,
            recut_passed=len(accepted) == 3 and len(args.batches) >= 4,
        )
    else:
        row["training_status"] = "not_run"
    row["cross_device_status"] = (
        "not_run: CPU-only environment"
        if not torch.cuda.is_available()
        else "not_run: use explicit cross-device matrix"
    )
    row["status"] = (
        "passed"
        if row["forward_passed"]
        and (not args.training or (row["backward_passed"] and row["recut_passed"]))
        else "not_validated"
    )
    return row


def main() -> None:
    """Run one model in this independent process and save auditable structured results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("resnet18", "transformer", "yolov8", "rfdetr"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--optimizers",
        choices=("sgd", "momentum", "adam"),
        nargs="+",
        default=["sgd", "momentum", "adam"],
    )
    parser.add_argument(
        "--partition-kind", choices=("both", "sequential", "noncontiguous_dag"), default="both"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch

    from torchlens.split import SplitFeatures, pipeline, prepare_graph

    torch.set_num_threads(1)
    torch.manual_seed(42)
    report = {
        "model": args.model,
        "device": args.device,
        "torch": torch.__version__,
        "seed": 42,
        "model_mode": "eval including AD probes",
        "cross_device_tested": False,
        "results": [],
        "tolerance": {"atol": 1e-4, "rtol": 1e-3},
    }
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model, dimensions, configuration = load_model(args.model, args.weights)
        model.to(args.device).eval()
        reference = deepcopy(model)
        initial = deepcopy(model.state_dict())
        capture = pipeline.capture_model
        counts = {"canonical": 0, "shape_witness": 0}

        def counted(*a: Any, **kw: Any) -> Any:
            counts["shape_witness" if kw.get("shape_witness") else "canonical"] += 1
            return capture(*a, **kw)

        with patch.object(pipeline, "capture_model", side_effect=counted):
            context = prepare_graph(
                model,
                torch.randn(1, *dimensions, device=args.device),
                features=SplitFeatures(training=args.training, retain_trace=False),
            )
            captured_counts = dict(counts)
            for kind in (
                ("sequential", "noncontiguous_dag")
                if args.partition_kind == "both"
                else (args.partition_kind,)
            ):
                try:
                    result = validate_kind(context, reference, initial, dimensions, kind, args)
                except Exception as error:
                    result = {
                        "partition_kind": kind,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(limit=4),
                    }
                report["results"].append(result)
            if counts != captured_counts:
                raise AssertionError("execution or recuts recaptured the model")
        report.update(
            configuration=configuration,
            capture_counts=counts,
            recut_capture_count=0,
            graph_compute_nodes=len(context.trace_graph.compute_nodes),
            batch_validation=context.batch_validation,
        )
        report["status"] = (
            "passed" if all(r["status"] == "passed" for r in report["results"]) else "not_validated"
        )
    except (ImportError, FileNotFoundError) as error:
        report.update(status="skipped", reason=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    if report["status"] == "failed" or any(r["status"] == "failed" for r in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
