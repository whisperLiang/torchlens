"""Run an isolated, no-download real-model DAG partition correctness probe.

Example: ``python benchmarks/split_graph_partition.py --model resnet18 --device cuda``.
Every requested batch is reported independently; typed refusals are not passes.
Use ``--training`` to additionally compare gradients and one SGD update. RF-DETR
requires ``--weights`` pointing to an existing trusted checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def main() -> None:
    """Probe a native model and print measured outcomes, including refusals."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("resnet18", "transformer", "yolov8", "rfdetr"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens.split import GraphPartition, SplitFeatures, SplitRequest, percent
    from torchlens.split.errors import SplitUnsupportedError

    torch.set_num_threads(1)
    torch.manual_seed(42)

    if args.model == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None, num_classes=10)
        dimensions = (3, 64, 64)
    elif args.model == "transformer":
        from transformers import DistilBertConfig, DistilBertModel

        model = DistilBertModel(
            DistilBertConfig(
                vocab_size=128,
                n_layers=2,
                dim=32,
                hidden_dim=64,
                n_heads=4,
                dropout=0.0,
                attention_dropout=0.0,
            )
        )
        dimensions = (12,)
    elif args.model == "yolov8":
        from ultralytics import YOLO

        model = YOLO("yolov8n.yaml").model
        dimensions = (3, 160, 160)
    else:
        from rfdetr import RFDETRNano
        from rfdetr.utilities.tensors import NestedTensor

        if args.weights is None or not args.weights.is_file():
            parser.error("RF-DETR requires --weights pointing to a trusted local checkpoint")

        class TensorModel(nn.Module):
            """Give the unchanged official detection core tensor inputs."""

            def __init__(self, core: Any) -> None:
                """Bind the official core."""
                super().__init__()
                self.core = core

            def forward(self, images: Any) -> Any:
                """Supply an all-valid spatial mask."""
                mask = torch.zeros(
                    (images.shape[0], images.shape[2], images.shape[3]),
                    device=images.device,
                    dtype=torch.bool,
                )
                return self.core(NestedTensor(images, mask))

        model = TensorModel(RFDETRNano(pretrain_weights=str(args.weights)).model.model)
        dimensions = (3, 384, 384)
    model.to(args.device).eval()

    def inputs(batch: int) -> Any:
        """Generate reproducible inputs with a fixed non-batch shape."""
        if args.model == "transformer":
            return torch.randint(0, 128, (batch, *dimensions), device=args.device)
        return torch.randn(batch, *dimensions, device=args.device, requires_grad=args.training)

    def leaves(value: Any) -> list[Any]:
        """Collect real floating tensor outputs from nested native structures."""
        if isinstance(value, torch.Tensor):
            return [value] if value.is_floating_point() else []
        if isinstance(value, dict):
            return [tensor for item in value.values() for tensor in leaves(item)]
        if isinstance(value, (tuple, list)):
            return [tensor for item in value for tensor in leaves(item)]
        return []

    def loss(output: Any, targets: Any) -> Any:
        """Construct a legal scalar loss on the model's actual floating outputs."""
        return sum(tensor.square().mean() for tensor in leaves(output))

    def assert_finite_close(actual: Any, expected: Any, *, atol: float) -> None:
        """Reject nonfinite values before applying the unchanged comparison tolerance."""

        def assert_finite(value: Any, path: str) -> None:
            """Check all tensor leaves, including nested outputs and optional gradients."""
            if isinstance(value, torch.Tensor):
                if not bool(torch.isfinite(value).all()):
                    raise AssertionError(f"{path} contains nonfinite tensor values")
            elif isinstance(value, dict):
                for key, item in value.items():
                    assert_finite(item, f"{path}[{key!r}]")
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    assert_finite(item, f"{path}[{index}]")

        assert_finite(actual, "actual")
        assert_finite(expected, "expected")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-3)

    model_version = version("rfdetr") if args.model == "rfdetr" else None
    # RF-DETR 1.8.3's native float32 CUDA backward needs about 2.5e-4
    # absolute allowance at rtol=1e-3 in the repeated native controls.
    # Keep an explicit 2x margin only for its parameter-gradient comparisons.
    gradient_atol = (
        5e-4
        if args.model == "rfdetr"
        and model_version == "1.8.3"
        and args.device == "cuda"
        and all(parameter.dtype == torch.float32 for parameter in model.parameters())
        else 1e-4
    )
    report: dict[str, Any] = {
        "model": args.model,
        "device": args.device,
        "training": args.training,
        "torch_version": torch.__version__,
        "source": str(tl.__file__),
        "batches": [],
        "model_mode": "eval (including gradient probes)",
        "model_version": model_version,
        "tolerances": {
            "forward": {"atol": 1e-4, "rtol": 1e-3},
            "parameter_gradient": {"atol": gradient_atol, "rtol": 1e-3},
            "input_gradient": {"atol": 1e-4, "rtol": 1e-3},
            "parameter_update": {"atol": 1e-4, "rtol": 1e-3},
        },
    }
    try:
        runtime = tl.split.prepare(
            model,
            inputs(1),
            SplitRequest(
                percent(50), features=SplitFeatures(training=args.training, retain_trace=False)
            ),
        )
        graph = runtime.trace_graph
        compute = graph.compute_nodes
        by_id = {node.canonical_id: node for node in graph.nodes}
        aliases = graph.node_id_by_alias
        calls = graph.replay_call_by_output_id
        compute_ids = {node.canonical_id for node in compute}
        midpoint = len(compute) // 2
        candidates = sorted(range(1, len(compute) - 1), key=lambda i: (abs(i - midpoint), i))
        report["partition_attempts"] = []
        cut = None
        selected: set[str] = set()
        policies = [("ancestor_closure", index) for index in candidates[:16]] + [
            ("ordered_prefix", index) for index in candidates[:16]
        ]
        for policy, index in policies:
            selected = set()
            pending = (
                [compute[index].canonical_id]
                if policy == "ancestor_closure"
                else [node.canonical_id for node in compute[: index + 1]]
            )
            while pending:
                node_id = pending.pop()
                if node_id in selected:
                    continue
                selected.add(node_id)
                pending.extend(aliases.get(parent, parent) for parent in by_id[node_id].parents)
                call = calls.get(node_id)
                if call is not None:
                    pending.extend(call.output_node_ids)
            selected.intersection_update(compute_ids)
            try:
                analysis = runtime.analyze_partition(GraphPartition(frozenset(selected)))
            except SplitUnsupportedError as error:
                report["partition_attempts"].append(
                    {"policy": policy, "anchor": compute[index].canonical_id, "refusal": str(error)}
                )
                continue
            cut = runtime.materialize(analysis)
            report["partition_attempts"].append(
                {"policy": policy, "anchor": compute[index].canonical_id, "status": "accepted"}
            )
            break
        if cut is None:
            raise RuntimeError(
                "No accepted partition among nearby ancestor/ordered-prefix candidates"
            )
        report.update(
            compute_nodes=len(compute),
            prefix_compute_nodes=len(selected),
            batch_validation=runtime.batch_validation,
        )
        for batch in args.batches:
            row: dict[str, Any] = {"batch": batch}
            try:
                x = inputs(batch)
                with torch.no_grad():
                    expected = model(x)
                    actual = cut.replay(x)
                    assert_finite_close(actual, expected, atol=1e-4)
                    row["forward_max_abs_error"] = max(
                        float((a - b).abs().max())
                        for a, b in zip(leaves(actual), leaves(expected), strict=True)
                    )
                if args.training:
                    model.zero_grad(set_to_none=True)
                    loss(model(x), None).backward()
                    expected_gradients = [
                        None if p.grad is None else p.grad.clone() for p in model.parameters()
                    ]
                    expected_input = None if x.grad is None else x.grad.clone()
                    expected_updated = [
                        p.detach().clone() if g is None else p.detach() - 0.01 * g
                        for p, g in zip(model.parameters(), expected_gradients, strict=True)
                    ]
                    model.zero_grad(set_to_none=True)
                    x.grad = None
                    boundary = cut.run_training_prefix(x)
                    _, gradients = cut.train_suffix(boundary, None, loss_fn=loss)
                    cut.backward_prefix(boundary, gradients)
                    for parameter, gradient in zip(
                        model.parameters(), expected_gradients, strict=True
                    ):
                        assert_finite_close(parameter.grad, gradient, atol=gradient_atol)
                    assert_finite_close(x.grad, expected_input, atol=1e-4)
                    row["parameter_gradient_max_abs_error"] = max(
                        (
                            float((parameter.grad - gradient).abs().max())
                            for parameter, gradient in zip(
                                model.parameters(), expected_gradients, strict=True
                            )
                            if gradient is not None
                        ),
                        default=0.0,
                    )
                    row["input_gradient_max_abs_error"] = (
                        None
                        if expected_input is None
                        else float((x.grad - expected_input).abs().max())
                    )
                    torch.optim.SGD(model.parameters(), lr=0.01).step()
                    for parameter, updated in zip(
                        model.parameters(), expected_updated, strict=True
                    ):
                        assert_finite_close(parameter, updated, atol=1e-4)
                    row["parameter_update_max_abs_error"] = max(
                        float((parameter - updated).abs().max())
                        for parameter, updated in zip(
                            model.parameters(), expected_updated, strict=True
                        )
                    )
                    row["gradients_and_sgd_update"] = "passed"
                cut = cut.partition(GraphPartition(frozenset(selected)))
                row["recut_reused_graph"] = cut.trace_graph is graph
                row["status"] = "passed"
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            report["batches"].append(row)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    print(json.dumps(report))
    if report.get("status") == "failed" or any(
        row["status"] == "failed" for row in report["batches"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
