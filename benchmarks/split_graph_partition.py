"""Strict isolated real-model validation of sequential and noncontiguous partitions.

Run: python benchmarks/split_graph_partition.py --model resnet18 --training --output result.json
No fallback between partition kinds. RF-DETR requires a trusted local checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from unittest.mock import patch


def result_status(rows: list[dict[str, Any]]) -> str:
    """Settle requested validations without promoting missing evidence to success."""
    statuses = {row["status"] for row in rows}
    if "failed" in statuses:
        return "failed"
    if len(statuses) == 1:
        return statuses.pop()
    return "not_validated"


def synchronize() -> None:
    """Finish work on every participating CUDA device at measurement boundaries."""
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def memory_snapshot() -> dict[str, Any]:
    """Separate process RSS from live and reserved CUDA storage."""
    import torch

    try:
        import psutil

        rss = psutil.Process().memory_info().rss
    except ImportError:
        rss = None
    return {
        "rss_bytes": rss,
        "cuda": {
            str(i): {
                "allocated_bytes": torch.cuda.memory_allocated(i),
                "reserved_bytes": torch.cuda.memory_reserved(i),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
            }
            for i in range(torch.cuda.device_count())
        },
    }


def effective_parameters(runtime: Any) -> dict[str, Any]:
    """Read execution bindings, including placed replicas, through stable state IDs."""
    result = {}
    for state_id, owners in runtime.parameter_bindings().items():
        if len({id(value) for value in owners.values()}) != 1:
            raise AssertionError(f"ambiguous effective parameter: {state_id}")
        result[state_id.removeprefix("parameter:")] = next(iter(owners.values()))
    return result


def make_optimizer(name: str, parameters: Any) -> Any:
    """Construct the same native optimizer for independent reference and split state."""
    import torch

    parameters = list({id(value): value for value in parameters if value.requires_grad}.values())
    if name == "adam":
        return torch.optim.Adam(parameters, lr=0.001)
    return torch.optim.SGD(parameters, lr=0.01, momentum=0.9 if name == "momentum" else 0)


def rebind_optimizer(name: str, previous: Any, old: dict[str, Any], new: dict[str, Any]) -> Any:
    """Explicitly reassociate caller-owned history after runtime state migration."""
    import torch

    optimizer = make_optimizer(name, new.values())
    if old.keys() != new.keys():
        raise AssertionError("repartition changed the effective parameter inventory")
    for key, parameter in new.items():
        for field, value in previous.state.get(old[key], {}).items():
            optimizer.state[parameter][field] = (
                value.detach().to(value.device if field == "step" else parameter.device).clone()
                if isinstance(value, torch.Tensor)
                else deepcopy(value)
            )
    return optimizer


def buffer_error(runtime: Any, reference: Any) -> float:
    """Check current execution buffers, including BatchNorm counters and running statistics."""
    from torchlens.split.ownership import _inventory

    rows, bindings = _inventory(runtime, runtime._segment_state_snapshots())
    expected = dict(reference.named_buffers())
    for row in rows:
        if row.kind == "buffer":
            for alias in row.aliases:
                if alias not in expected:
                    value = reference
                    for component in alias.split("."):
                        value = getattr(value, component)
                    expected[alias] = value
    return max(
        (
            finite_close(value, expected[alias])
            for row in rows
            if row.kind == "buffer"
            for alias in row.aliases
            for value in bindings[row.state_id].values()
        ),
        default=0.0,
    )


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


def load_model(
    name: str, weights: Path | None, *, model_mode: str = "eval"
) -> tuple[Any, tuple[int, ...], dict[str, Any]]:
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
            dropout=0.1 if model_mode == "train" else 0.0,
            attention_dropout=0.1 if model_mode == "train" else 0.0,
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

    detector = RFDETRNano(pretrain_weights=str(weights))

    class Images(nn.Module):
        """Supply a valid spatial mask to the unchanged official detection core."""

        def __init__(self) -> None:
            super().__init__()
            self.core = detector.model.model

        def forward(self, x: Any) -> Any:
            return self.core(
                NestedTensor(
                    x,
                    torch.zeros(
                        (x.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=torch.bool
                    ),
                )
            )

    return (
        Images(),
        (3, 384, 384),
        {"weights": str(weights.resolve()), "model_config": detector.model_config.model_dump()},
    )


def training_objective(model: Any, args: Any, configuration: dict[str, Any]) -> Any:
    """Use official detection losses with annotated boxes for actual train-mode probes."""
    if args.model_mode != "train":
        return scalar_loss
    if args.model == "yolov8":
        from ultralytics.cfg import get_cfg

        model.args = get_cfg()
        criteria: dict[str, Any] = {}

        def yolo_loss(output: Any, targets: Any) -> Any:
            device = tensor_leaves(output)[0].device
            batch = {key: value.to(device) for key, value in targets.items()}
            key = str(device)
            if key not in criteria:
                # The official criterion keeps device-local anchors/strides.
                # Give each comparison device its own native loss object.
                criteria[key] = deepcopy(model).to(device).init_criterion()
            return criteria[key](output, batch)[0].sum()

        return yolo_loss
    if args.model == "rfdetr":
        from rfdetr.config import RFDETRNanoConfig, TrainConfig
        from rfdetr.models.lwdetr import build_criterion_from_config

        criterion, _ = build_criterion_from_config(
            RFDETRNanoConfig(**{**configuration["model_config"], "device": args.device}),
            TrainConfig(dataset_dir="."),
        )

        def detr_loss(output: Any, targets: Any) -> Any:
            device = tensor_leaves(output)[0].device
            criterion.to(device)
            annotations = [
                {key: value.to(device) for key, value in item.items()} for item in targets
            ]
            losses = criterion(output, annotations)
            return sum(
                value * criterion.weight_dict[key]
                for key, value in losses.items()
                if key in criterion.weight_dict
            )

        return detr_loss
    return scalar_loss


def training_targets(name: str, x: Any) -> Any:
    """Supply one valid labelled normalized box per synthetic detection image."""
    import torch

    batch = x.shape[0]
    if name == "yolov8":
        return {
            "img": x,
            "batch_idx": torch.arange(batch),
            "cls": torch.zeros(batch, 1),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]).repeat(batch, 1),
        }
    if name == "rfdetr":
        return [
            {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]])}
            for _ in range(batch)
        ]
    return None


def native_family(name: str) -> str:
    """Unify native observation facades with captured semantic operation families."""
    normalized = name.replace("_", "").lower()
    return {
        "iadd": "add",
        "isub": "sub",
        "imul": "mul",
        "itruediv": "div",
        "truediv": "div",
        "truedivide": "div",
        "ifloordiv": "floordivide",
        "floordiv": "floordivide",
        "splitwithsizes": "split",
    }.get(normalized, normalized)


def training_step(
    runtime: Any,
    reference: Any,
    x: Any,
    actual_opt: Any,
    expected_opt: Any,
    *,
    loss_fn: Any = scalar_loss,
    targets: Any = None,
) -> dict[str, Any]:
    """Compare independent native and split AD graphs, including connected frontier gradients."""
    import torch
    from torch.overrides import TorchFunctionMode

    from torchlens.split.adapters.torch import _slice_output_by_path
    from torchlens.split.graph import ReplayCall

    actual_opt.zero_grad(set_to_none=True)
    expected_opt.zero_grad(set_to_none=True)
    ref_x = x.detach().to(next(reference.parameters()).device).clone().requires_grad_(True)
    observed_gradients: dict[str, Any] = {}
    attached: set[str] = set()
    graph = runtime.trace_graph
    boundary_ids = {value: key for key, value in runtime.plan.boundary_bindings.items()}
    calls_by_family: dict[str, list[Any]] = {}
    # Native observation also sees state writes. The partition's compute-call
    # inventory excludes buffer nodes; using it here shifts occurrence indexes
    # for BatchNorm counters and partially retained multi-output calls.
    native_groups: dict[Any, list[Any]] = {}
    for node in graph.nodes:
        if not node.is_input and not node.is_output:
            key = node.func_call_id if node.func_call_id is not None else node.canonical_id
            native_groups.setdefault(key, []).append(node)
    for key, members in native_groups.items():
        call = ReplayCall(str(key), members[0].func_call_id, tuple(n.canonical_id for n in members))
        leader = next((node for node in members if node.target is not None), None)
        if leader is not None:
            calls_by_family.setdefault(native_family(leader.op_type), []).append((leader, call))
    counts: dict[str, int] = {}

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
        """Observe native outputs in semantic call order, without executing split recipes."""

        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            family = native_family(str(getattr(func, "__name__", "")))
            calls = calls_by_family.get(family)
            if calls is None:
                return func(*args, **(kwargs or {}))
            index = counts.get(family, 0)
            counts[family] = index + 1
            if index >= len(calls):
                raise AssertionError(f"native reference call count changed: {func}")
            leader, call = calls[index]
            from torchlens.utils.rng import execute_with_restored_rng_autocast

            op = leader.op
            result = execute_with_restored_rng_autocast(
                func,
                args,
                kwargs or {},
                rng_states=getattr(op, "func_rng_states", None),
                autocast_state=getattr(op, "func_autocast_state", None),
            )
            if not boundary_ids.keys() & set(call.output_node_ids):
                return result
            for node_id in call.output_node_ids:
                if node_id in boundary_ids:
                    tensor = _slice_output_by_path(
                        result, graph.node_by_id[node_id].output_container_path
                    )
                    if not isinstance(tensor, torch.Tensor):
                        raise AssertionError(f"native frontier path is not a tensor: {node_id}")
                    attach(tensor, boundary_ids[node_id])
            return result

    torch.manual_seed(317)
    with ObserveFrontier():
        expected_output = reference(ref_x)
    expected_loss = loss_fn(expected_output, targets)
    expected_loss.backward()
    torch.manual_seed(317)
    synchronize()
    start = time.perf_counter()
    boundary = runtime.run_training_prefix(x)
    for tensor in boundary.tensors.values():
        if tensor.requires_grad:
            tensor.retain_grad()
    timing: dict[str, float] = {}

    def checked_loss(output: Any, targets: Any) -> Any:
        synchronize()
        timing["forward_ms"] = 1000 * (time.perf_counter() - start)
        timing["output_error"] = finite_close(output, expected_output)
        timing["backward_start"] = time.perf_counter()
        return loss_fn(output, targets)

    actual_loss, gradients = runtime.train_suffix(boundary, targets, loss_fn=checked_loss)
    runtime.backward_prefix(boundary, gradients)
    synchronize()
    backward_ms = 1000 * (time.perf_counter() - timing["backward_start"])
    loss_error = finite_close(actual_loss, expected_loss)
    input_error = finite_close(x.grad, ref_x.grad)
    actual_parameters = effective_parameters(runtime)
    expected_parameters = dict(reference.named_parameters())
    bound = {id(p) for group in actual_opt.param_groups for p in group["params"]}
    if bound != {id(p) for p in actual_parameters.values() if p.requires_grad}:
        raise AssertionError("optimizer is bound to stale execution parameters")
    for name, parameter in expected_parameters.items():
        if name not in actual_parameters and parameter.grad is not None:
            raise AssertionError(f"missing execution parameter gradient: {name}")
    gradient_error = max(
        (
            finite_close(p.grad, expected_parameters[name].grad)
            for name, p in actual_parameters.items()
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
            for a, b in ((p, expected_parameters[n]) for n, p in actual_parameters.items())
        ),
        default=0.0,
    )
    optimizer_error = max(
        (
            finite_close(actual_opt.state[a], expected_opt.state[b])
            for a, b in ((p, expected_parameters[n]) for n, p in actual_parameters.items())
        ),
        default=0.0,
    )
    return {
        "forward_error": timing["output_error"],
        "forward_ms": timing["forward_ms"],
        "loss_and_backward_ms": backward_ms,
        "loss": float(actual_loss.detach()),
        "loss_error": loss_error,
        "input_gradient_error": input_error,
        "parameter_gradient_error": gradient_error,
        "connected_boundary_gradient_error": frontier_error,
        "parameter_update_error": update_error,
        "optimizer_state_error": optimizer_error,
        "buffer_error": buffer_error(runtime, reference),
    }


def validate_kind(
    context: Any,
    reference: Any,
    initial: dict[str, Any],
    dimensions: tuple[int, ...],
    kind: str,
    args: Any,
) -> dict[str, Any]:
    """Retain completed checks and candidate refusals even when a later step fails."""
    from torchlens.split.errors import SplitError

    row: dict[str, Any] = {}
    try:
        return _validate_kind(context, reference, initial, dimensions, kind, args, row)
    except Exception as error:
        status = "unsupported" if isinstance(error, SplitError) else "failed"
        row["checks"] = {
            key: status if value == "not_validated" else value
            for key, value in row.get("checks", {}).items()
        }
        row.update(
            partition_kind=kind,
            status=status,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(limit=6),
            reason=getattr(getattr(error, "context", None), "reason", None),
        )
        return row


def _validate_kind(
    context: Any,
    reference: Any,
    initial: dict[str, Any],
    dimensions: tuple[int, ...],
    kind: str,
    args: Any,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Independently report one partition kind; never substitute sequential success for DAG success."""
    import torch

    from torchlens.split.errors import SplitError

    row.update(
        {
            "partition_kind": kind,
            "partition_valid": False,
            "noncontiguous_verified": False,
            "forward_passed": False,
            "backward_passed": False,
            "optimizer_update_passed": False,
            "dynamic_batch_passed": False,
            "recut_passed": False,
            "cross_device_tested": False,
            "checks": dict.fromkeys(
                (
                    "forward",
                    "backward",
                    "optimizer",
                    "input_gradient",
                    "parameter_gradient",
                    "boundary_gradient",
                    "buffers",
                    "rng_dropout",
                    "dynamic_batch",
                    "repartition",
                    "cross_device",
                    "train_mode",
                ),
                "not_run",
            ),
            "attempts": [],
            "batches": [],
            "training_steps": [],
        }
    )
    accepted = []
    heterogeneous = context.adapter.normalize_device(
        args.prefix_device
    ) != context.adapter.normalize_device(args.suffix_device)
    analysis_times, materialization_times = [], []
    for partition in candidates(context.trace_graph, kind):
        try:
            synchronize()
            start = time.perf_counter()
            analysis = context.analyze(partition)
            synchronize()
            analysis_times.append(1000 * (time.perf_counter() - start))
            if not any(n in partition.prefix_node_ids for n in analysis.plan.boundary_node_ids):
                raise AssertionError("frontier lacks a true compute dependency")
            start = time.perf_counter()
            context.materialize(analysis)
            synchronize()
            materialization_times.append(1000 * (time.perf_counter() - start))
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
            "checks": {
                key: "unsupported" if key in {"forward", "train_mode"} else value
                for key, value in row["checks"].items()
            },
        }
    row.update(
        partition_valid=True,
        noncontiguous_verified=kind == "noncontiguous_dag",
        distinct_partitions=len(accepted),
        partition_identities=[a.request.boundary for a in accepted],
        analysis_ms=analysis_times,
        materialization_ms=materialization_times,
    )
    context.model.load_state_dict(initial)
    reference.load_state_dict(initial)
    runtime = context.materialize(accepted[0])
    for batch in args.batches:
        x = torch.randn(
            batch, *dimensions, device=args.prefix_device, dtype=getattr(torch, args.dtype)
        )
        with torch.no_grad():
            torch.manual_seed(317)
            # Train-mode forward is checked once inside the matched AD step:
            # redundant forwards would mutate BatchNorm and consume dropout RNG.
            if args.model_mode == "train":
                continue
            expected = reference(x.to(args.device))
            torch.manual_seed(317)
            synchronize()
            start = time.perf_counter()
            actual = runtime.replay(x)
            synchronize()
            forward_ms = 1000 * (time.perf_counter() - start)
            error = finite_close(actual, expected)
            finite_close(runtime.replay(x), expected)
        start = time.perf_counter()
        profile = context.graph_profile(batch_size=batch)
        profile_ms = 1000 * (time.perf_counter() - start)
        layout = runtime.boundary_layout(batch_size=batch)
        row["batches"].append(
            {
                "batch": batch,
                "status": "passed",
                "forward_error": error,
                "forward_ms": forward_ms,
                "graph_profile_ms": profile_ms,
                "forward_flops": profile.forward_flops,
                "backward_flops_estimate": profile.backward_flops,
                "boundary_bytes": layout.forward_payload_bytes,
            }
        )
    row["forward_passed"] = args.model_mode == "eval"
    if args.model_mode == "eval":
        row["checks"]["forward"] = "passed"
    row["dynamic_batch_passed"] = {1, 2, 4, 8} <= set(args.batches)
    if args.training:
        for name in args.optimizers:
            context.model.load_state_dict(initial)
            reference.load_state_dict(initial)
            runtime = context.materialize(accepted[0])
            parameters = effective_parameters(runtime)
            actual_opt, expected_opt = (
                make_optimizer(name, parameters.values()),
                make_optimizer(name, reference.parameters()),
            )
            sequence = (0, 1, 2, 0) if len(accepted) == 3 else (0,) * 4
            for index, batch in enumerate(args.batches):
                row["checks"].update(
                    dict.fromkeys(
                        (
                            "forward",
                            "backward",
                            "optimizer",
                            "input_gradient",
                            "parameter_gradient",
                            "boundary_gradient",
                            "buffers",
                            "repartition",
                        ),
                        "not_validated",
                    )
                )
                if args.model_mode == "train":
                    row["checks"]["train_mode"] = "not_validated"
                if heterogeneous:
                    row["cross_device_tested"] = True
                    row["checks"]["cross_device"] = "not_validated"
                analysis = accepted[sequence[index % 4]]
                synchronize()
                start = time.perf_counter()
                migration = runtime.plan_state_migration(analysis.request.partition)
                synchronize()
                migration_ms = 1000 * (time.perf_counter() - start)
                if not migration.supported:
                    raise AssertionError(
                        f"optimizer migration unsupported: {migration.unsupported_reasons}"
                    )
                start = time.perf_counter()
                runtime = runtime.repartition(analysis.request.partition)
                synchronize()
                recut_ms = 1000 * (time.perf_counter() - start)
                current = effective_parameters(runtime)
                actual_opt = rebind_optimizer(name, actual_opt, parameters, current)
                parameters = current
                # Inherit current buffers rather than rebinding capture-time values.
                buffer_error(runtime, reference)
                if runtime.trace_graph is not context.trace_graph:
                    raise AssertionError("recut replaced the captured graph")
                x = torch.randn(
                    batch,
                    *dimensions,
                    device=args.prefix_device,
                    dtype=getattr(torch, args.dtype),
                    requires_grad=True,
                )
                start = time.perf_counter()
                errors = training_step(
                    runtime,
                    reference,
                    x,
                    actual_opt,
                    expected_opt,
                    loss_fn=args.loss_fn,
                    targets=training_targets(args.model, x) if args.model_mode == "train" else None,
                )
                row["forward_passed"] = True
                synchronize()
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
                        "memory": memory_snapshot(),
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
        row["checks"].update(
            dict.fromkeys(
                (
                    "forward",
                    "backward",
                    "optimizer",
                    "input_gradient",
                    "parameter_gradient",
                    "boundary_gradient",
                    "buffers",
                ),
                "passed",
            )
        )
        row["checks"]["repartition"] = "passed" if row["recut_passed"] else "not_validated"
        if args.model_mode == "train":
            row["checks"]["train_mode"] = "passed"
            if args.dropout_active:
                row["checks"]["rng_dropout"] = "passed"
    else:
        row["training_status"] = "not_run"
    row["checks"]["forward"] = "passed"
    row["checks"]["dynamic_batch"] = "passed" if row["dynamic_batch_passed"] else "not_validated"
    row["cross_device_tested"] = heterogeneous
    if row["cross_device_tested"]:
        row["checks"]["cross_device"] = "passed"
    row["memory"] = memory_snapshot()
    row["status"] = (
        "passed"
        if row["forward_passed"]
        and row["dynamic_batch_passed"]
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
    parser.add_argument("--prefix-device", help="Prefix placement, e.g. cpu or cuda:0")
    parser.add_argument("--suffix-device", help="Suffix placement, e.g. cpu or cuda:1")
    parser.add_argument("--model-mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Only passed exits successfully")
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
    if not args.batches or min(args.batches) < 1:
        parser.error("batches must be positive")
    args.training = args.training or args.model_mode == "train"
    args.prefix_device = args.prefix_device or args.device
    args.suffix_device = args.suffix_device or args.device
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch

    from torchlens.split import PlacementPlan, SplitFeatures, pipeline, prepare_graph
    from torchlens.split.errors import SplitError

    torch.set_num_threads(1)
    torch.manual_seed(42)
    report = {
        "model": args.model,
        "device": args.device,
        "placement": {"prefix": args.prefix_device, "suffix": args.suffix_device},
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "seed": 42,
        "model_mode": args.model_mode,
        "dtype": args.dtype,
        "objective": "official detection criterion with synthetic annotated boxes"
        if args.model_mode == "train" and args.model in {"yolov8", "rfdetr"}
        else "synthetic scalar graph AD objective",
        "rng_policy": "native reference and split replay use the same captured per-call RNG states",
        "cross_device_tested": False,
        "results": [],
        "tolerance": {"atol": 1e-4, "rtol": 1e-3},
    }
    report["libraries"] = {}
    for package in ("torchvision", "transformers", "ultralytics", "rfdetr"):
        try:
            report["libraries"][package] = version(package)
        except PackageNotFoundError:
            report["libraries"][package] = None
    revision = subprocess.run(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    report["git_commit"] = revision.stdout.strip() or None
    try:
        for placement in (args.device, args.prefix_device, args.suffix_device):
            device = torch.device(placement)
            if device.type == "cuda" and (
                not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()
            ):
                raise FileNotFoundError(f"requested CUDA device unavailable: {placement}")
        model, dimensions, configuration = load_model(
            args.model, args.weights, model_mode=args.model_mode
        )
        model.to(device=args.device, dtype=getattr(torch, args.dtype)).train(
            args.model_mode == "train"
        )
        args.dropout_active = any(
            isinstance(module, torch.nn.modules.dropout._DropoutNd)
            and module.training
            and module.p > 0
            for module in model.modules()
        )
        report["dropout_active"] = args.dropout_active
        reference = deepcopy(model)
        initial = deepcopy(model.state_dict())
        args.loss_fn = training_objective(reference, args, configuration)
        report["configuration"] = configuration
        capture = pipeline.capture_model
        counts = {"canonical": 0, "shape_witness": 0}

        def counted(*a: Any, **kw: Any) -> Any:
            counts["shape_witness" if kw.get("shape_witness") else "canonical"] += 1
            return capture(*a, **kw)

        with patch.object(pipeline, "capture_model", side_effect=counted):
            synchronize()
            started = time.perf_counter()
            context = prepare_graph(
                model,
                torch.randn(1, *dimensions, device=args.device, dtype=getattr(torch, args.dtype)),
                features=SplitFeatures(training=args.training, retain_trace=False),
                placement=PlacementPlan.across(args.prefix_device, args.suffix_device),
            )
            synchronize()
            report["capture_ms"] = 1000 * (time.perf_counter() - started)
            captured_counts = dict(counts)
            for kind in (
                ("sequential", "noncontiguous_dag")
                if args.partition_kind == "both"
                else (args.partition_kind,)
            ):
                result = validate_kind(context, reference, initial, dimensions, kind, args)
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
        report["status"] = result_status(report["results"])
    except SplitError as error:
        report.update(
            status="unsupported",
            reason=getattr(error.context, "reason", error.code),
            error=str(error),
        )
    except (ImportError, FileNotFoundError) as error:
        report.update(status="skipped", reason=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report["checks"] = {
        kind: next(
            (row["status"] for row in report["results"] if row["partition_kind"] == kind), "not_run"
        )
        for kind in ("sequential", "noncontiguous_dag")
    }
    for check in (
        "forward",
        "backward",
        "optimizer",
        "input_gradient",
        "parameter_gradient",
        "boundary_gradient",
        "buffers",
        "rng_dropout",
        "dynamic_batch",
        "repartition",
        "cross_device",
        "train_mode",
    ):
        report["checks"][check] = (
            result_status(
                [
                    {"status": row.get("checks", {}).get(check, "not_run")}
                    for row in report["results"]
                ]
            )
            if report["results"]
            else "not_run"
        )
    report["cross_device_tested"] = any(
        row.get("cross_device_tested", False) for row in report["results"]
    )
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    if report["status"] == "failed" or args.strict and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
