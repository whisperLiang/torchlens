"""PyTorch real-model native DAG replay tests."""

from __future__ import annotations

from importlib import import_module

import pytest
import torch
from dag_helpers import close_call_ids, smallest_transfer_partitions
from real_model_helpers import (
    _run_backend_subprocess,
    _skip_if_module_missing,
    _skip_unless_enabled,
)

from torchlens.split import prepare_graph

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def _torch_test_devices() -> list[torch.device]:
    """Return the torch devices covered by cross-device split tests."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


def test_torch_transformer_cross_batch_and_device() -> None:
    """A real Transformer output container replays across batch and device."""
    _skip_unless_enabled()
    _skip_if_module_missing("transformers")
    transformers = import_module("transformers")
    config = transformers.DistilBertConfig(
        vocab_size=128, n_layers=2, dim=32, hidden_dim=64, n_heads=4
    )
    torch.manual_seed(0)
    for device in _torch_test_devices():
        model = transformers.DistilBertModel(config).to(device).eval()
        x = torch.randint(0, 128, (2, 12), device=device)
        dag_graph_1 = prepare_graph(model, x)
        for boundary in smallest_transfer_partitions(dag_graph_1):
            runtime = dag_graph_1.materialize(dag_graph_1.analyze(boundary))
            for batch in (1, 4):
                replay_x = torch.randint(0, 128, (batch, 12), device=device)
                with torch.no_grad():
                    split_output = runtime.replay(replay_x)
                    full_output = model(replay_x)
                assert torch.allclose(
                    split_output.last_hidden_state,
                    full_output.last_hidden_state,
                    atol=0.0001,
                    rtol=0.001,
                )


def test_torchvision_extra_real_models_cross_batch_and_device() -> None:
    """Additional torchvision models replay native DAG partitions across batch/device."""
    _skip_unless_enabled()
    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(0)
    model_cases = (
        (
            "mobilenet_v2",
            lambda: torchvision.models.mobilenet_v2(weights=None, num_classes=10),
            (2, 3, 64, 64),
        ),
        (
            "squeezenet1_0",
            lambda: torchvision.models.squeezenet1_0(weights=None, num_classes=10),
            (2, 3, 64, 64),
        ),
        (
            "efficientnet_b0",
            lambda: torchvision.models.efficientnet_b0(weights=None, num_classes=10),
            (2, 3, 64, 64),
        ),
        (
            "convnext_tiny",
            lambda: torchvision.models.convnext_tiny(weights=None, num_classes=10),
            (2, 3, 64, 64),
        ),
    )
    for device in _torch_test_devices():
        for _name, factory, shape in model_cases:
            model = factory().to(device).eval()
            x = torch.ones(shape, device=device)
            graph = prepare_graph(model, x)
            for partition in smallest_transfer_partitions(graph):
                runtime = graph.materialize(graph.analyze(partition))
                for batch in (1, 3):
                    replay_x = torch.ones((batch, *shape[1:]), device=device)
                    with torch.no_grad():
                        split_output = runtime.replay(replay_x)
                        full_output = model(replay_x)
                    atol = 0.0005 if device.type == "cuda" else 0.0001
                    assert torch.allclose(split_output, full_output, atol=atol, rtol=0.001)


@pytest.mark.parametrize("cross_device", [False, True])
def test_yolo26_detection_split_replay_cross_batch_and_device(cross_device: bool) -> None:
    """YOLO26 official weights cover full detection and backbone split replay."""
    if cross_device and (not torch.cuda.is_available()):
        pytest.skip("CUDA unavailable; CPU detection replay is covered separately.")
    _skip_unless_enabled()
    _skip_if_module_missing("ultralytics")
    _run_backend_subprocess(
        """
        import hashlib
        import os
        from pathlib import Path

        import torch
        from dag_helpers import close_call_ids, smallest_transfer_partitions
        from torch import nn

        from torchlens.split import SplitFeatures, prepare_graph
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        get_dynamo_optimized_module_type(force_probe=True)
        from ultralytics import YOLO

        torch.set_num_threads(1)
        cache_root = Path(
            os.environ.get("TORCHLENS_MODEL_CACHE", str(Path.home() / ".cache" / "torchlens" / "models"))
        )
        weight_path = cache_root / "ultralytics" / "yolo26n.pt"
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        detector = YOLO(str(weight_path)).model.eval()
        digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        assert digest == "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"  # pragma: allowlist secret


        def tensor_leaves(value):
            if isinstance(value, torch.Tensor):
                return [value]
            if isinstance(value, dict):
                return [leaf for key in sorted(value) for leaf in tensor_leaves(value[key])]
            if isinstance(value, (list, tuple)):
                return [leaf for item in value for leaf in tensor_leaves(item)]
            return []


        def assert_same(left, right, *, atol=0.0001):
            left_leaves = tensor_leaves(left)
            right_leaves = tensor_leaves(right)
            assert len(left_leaves) == len(right_leaves)
            for actual, expected in zip(left_leaves, right_leaves, strict=True):
                torch.testing.assert_close(actual, expected, atol=atol, rtol=0.001)


        full_input = torch.zeros(2, 3, 160, 160)
        dag_graph_1 = prepare_graph(detector, full_input, backend="torch")
        for boundary in smallest_transfer_partitions(dag_graph_1):
            runtime = dag_graph_1.materialize(dag_graph_1.analyze(boundary))
            with torch.no_grad():
                assert_same(runtime.replay(full_input), detector(full_input))
        backbone = nn.Sequential(*list(detector.model[:11])).eval()
        trace_input = torch.zeros(2, 3, 160, 160)
        graph = prepare_graph(
            backbone, trace_input, backend="torch", features=SplitFeatures(live_param_sources=True)
        )
        dag = graph.compute_dag()
        partition = dag.to_partition(
            close_call_ids(
                dag, frozenset((call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]))
            )
        )
        runtime = graph.materialize(graph.analyze(partition))
        for batch in (1, 2, 3):
            replay_input = torch.zeros(batch, 3, 160, 160)
            with torch.no_grad():
                assert_same(runtime.replay(replay_input), backbone(replay_input))
        if os.environ["TORCHLENS_REAL_MODEL_CROSS_DEVICE"] != "1":
            raise SystemExit(0)
        for batch in (1, 3):
            backbone.cpu()
            cpu_input = torch.zeros(batch, 3, 160, 160)
            boundary = runtime.run_prefix(cpu_input)
            backbone.cuda()
            gpu_boundary = boundary.to("cuda", adapter=runtime.adapter)
            gpu_input = torch.zeros(batch, 3, 160, 160, device="cuda")
            with torch.no_grad():
                assert_same(runtime.run_suffix(gpu_boundary), backbone(gpu_input), atol=0.005)
        os._exit(0)

        """,
        timeout=300,
        env_overrides={"TORCHLENS_REAL_MODEL_CROSS_DEVICE": "1" if cross_device else "0"},
    )


def test_yolov8n_smallest_dag_partitions_cross_batch_and_device() -> None:
    """YOLOv8n replays the 100 smallest native DAG partitions across batch and device."""
    _skip_unless_enabled()
    _skip_if_module_missing("ultralytics")
    _run_backend_subprocess(
        """
        import copy
        import os

        import torch
        from dag_helpers import smallest_transfer_partitions

        from torchlens.split import prepare_graph
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        torch.set_num_threads(1)
        get_dynamo_optimized_module_type(force_probe=True)
        from ultralytics import YOLO


        def assert_same(actual, expected):
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)
                torch.testing.assert_close(actual, expected, atol=0.0001, rtol=0.001)
                return
            assert type(actual) is type(expected)
            if isinstance(expected, dict):
                assert tuple(actual) == tuple(expected)
                for key in expected:
                    assert_same(actual[key], expected[key])
                return
            if isinstance(expected, (list, tuple)):
                assert len(actual) == len(expected)
                for actual_item, expected_item in zip(actual, expected, strict=True):
                    assert_same(actual_item, expected_item)
                return
            assert actual == expected


        def assert_structure(actual, expected, device):
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)
                assert actual.shape == expected.shape
                assert actual.dtype == expected.dtype
                assert actual.device.type == device
                if actual.is_floating_point() or actual.is_complex():
                    assert torch.isfinite(actual).all()
                return
            assert type(actual) is type(expected)
            if isinstance(expected, dict):
                assert tuple(actual) == tuple(expected)
                for key in expected:
                    assert_structure(actual[key], expected[key], device)
                return
            if isinstance(expected, (list, tuple)):
                assert len(actual) == len(expected)
                for actual_item, expected_item in zip(actual, expected, strict=True):
                    assert_structure(actual_item, expected_item, device)


        torch.manual_seed(0)
        cpu_model = YOLO("yolov8n.yaml").model.eval()
        gpu_model = copy.deepcopy(cpu_model).cuda().eval() if torch.cuda.is_available() else None
        cpu_seed_graph = prepare_graph(cpu_model, torch.zeros(2, 3, 160, 160))
        cpu_seed_dag = cpu_seed_graph.compute_dag()
        cpu_seed_partition = cpu_seed_dag.to_partition(())
        cpu_seed = cpu_seed_graph.materialize(cpu_seed_graph.analyze(cpu_seed_partition))
        assert cpu_seed.trace_graph.shape_program.unresolved == {}
        assert cpu_seed.traced_batch_size == 1
        assert cpu_seed.trace_graph.shape_program.witness_batch_sizes == (2,)
        assert cpu_seed.batch_validation["status"] == "passed"
        gpu_seed = None
        if gpu_model is not None:
            gpu_seed_graph = prepare_graph(gpu_model, torch.zeros(2, 3, 160, 160, device="cuda"))
            gpu_seed_dag = gpu_seed_graph.compute_dag()
            gpu_seed_partition = gpu_seed_dag.to_partition(())
            gpu_seed = gpu_seed_graph.materialize(gpu_seed_graph.analyze(gpu_seed_partition))
            assert [node.canonical_id for node in gpu_seed.trace_graph.compute_nodes] == [
                node.canonical_id for node in cpu_seed.trace_graph.compute_nodes
            ]
        partitions = smallest_transfer_partitions(cpu_seed)
        assert len(partitions) == 100
        points = partitions
        cpu_inputs = {batch: torch.zeros(batch, 3, 160, 160) for batch in (1, 2, 3)}
        with torch.inference_mode():
            cpu_expected = {batch: cpu_model(value) for batch, value in cpu_inputs.items()}
            gpu_inputs = (
                {batch: torch.zeros(batch, 3, 160, 160, device="cuda") for batch in (1, 2, 3)}
                if gpu_model is not None
                else {}
            )
            gpu_expected = (
                {batch: gpu_model(value) for batch, value in gpu_inputs.items()}
                if gpu_model is not None
                else {}
            )
            counts = {"cpu": 0, "cuda": 0, "cpu_cuda": 0, "cuda_cpu": 0}
            for point in points:
                cpu_runtime = cpu_seed.repartition(point)
                gpu_runtime = None if gpu_seed is None else gpu_seed.repartition(point)
                for batch in (1, 2, 3):
                    assert_same(cpu_runtime.replay(cpu_inputs[batch]), cpu_expected[batch])
                    counts["cpu"] += 1
                    if gpu_runtime is not None:
                        assert_same(gpu_runtime.replay(gpu_inputs[batch]), gpu_expected[batch])
                        counts["cuda"] += 1
                if gpu_runtime is None:
                    continue
                for batch in (1, 3):
                    cpu_boundary = cpu_runtime.run_prefix(cpu_inputs[batch])
                    assert cpu_boundary.tensors.keys() == cpu_boundary.spec.keys()
                    cpu_to_cuda = gpu_runtime.run_suffix(
                        cpu_boundary.to("cuda", adapter=gpu_runtime.adapter)
                    )
                    assert_structure(cpu_to_cuda, gpu_expected[batch], "cuda")
                    counts["cpu_cuda"] += 1
                    gpu_boundary = gpu_runtime.run_prefix(gpu_inputs[batch])
                    assert gpu_boundary.tensors.keys() == gpu_boundary.spec.keys()
                    cuda_to_cpu = cpu_runtime.run_suffix(
                        gpu_boundary.to("cpu", adapter=cpu_runtime.adapter)
                    )
                    assert_structure(cuda_to_cpu, cpu_expected[batch], "cpu")
                    counts["cuda_cpu"] += 1
        expected_counts = {
            "cpu": len(points) * 3,
            "cuda": len(points) * 3,
            "cpu_cuda": len(points) * 2,
            "cuda_cpu": len(points) * 2,
        }
        assert counts["cpu"] == expected_counts["cpu"]
        if gpu_seed is not None:
            assert counts == expected_counts
        print(len(points), counts, flush=True)
        os._exit(0)

        """,
        timeout=7200,
    )


def test_rfdetr_smallest_dag_partitions_cross_batch_and_device() -> None:
    """RF-DETR-N replays the 100 smallest native DAG partitions across batch and device."""
    _skip_unless_enabled()
    _skip_if_module_missing("rfdetr")
    _run_backend_subprocess(
        """
        import copy
        import hashlib
        import os
        from pathlib import Path

        import torch
        from dag_helpers import smallest_transfer_partitions
        from torch import nn

        from torchlens.split import prepare_graph
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        torch.set_num_threads(1)
        os.environ.setdefault("RF_HOME", str(Path.home() / ".cache" / "torchlens" / "models" / "rfdetr"))
        get_dynamo_optimized_module_type(force_probe=True)
        from rfdetr import RFDETRNano
        from rfdetr.utilities.tensors import NestedTensor


        class RFDETRTensorModel(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, images):
                mask = torch.zeros(
                    (images.shape[0], images.shape[2], images.shape[3]),
                    device=images.device,
                    dtype=torch.bool,
                )
                return self.core(NestedTensor(images, mask))


        def assert_same(actual, expected):
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)
                torch.testing.assert_close(actual, expected, atol=0.0001, rtol=0.001)
                return
            assert type(actual) is type(expected)
            if isinstance(expected, dict):
                assert tuple(actual) == tuple(expected)
                for key in expected:
                    assert_same(actual[key], expected[key])
                return
            if isinstance(expected, (list, tuple)):
                assert len(actual) == len(expected)
                for actual_item, expected_item in zip(actual, expected, strict=True):
                    assert_same(actual_item, expected_item)
                return
            assert actual == expected


        def assert_structure(actual, expected, device):
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)
                assert actual.shape == expected.shape
                assert actual.dtype == expected.dtype
                assert actual.device.type == device
                if actual.is_floating_point() or actual.is_complex():
                    assert torch.isfinite(actual).all()
                return
            assert type(actual) is type(expected)
            if isinstance(expected, dict):
                assert tuple(actual) == tuple(expected)
                for key in expected:
                    assert_structure(actual[key], expected[key], device)
                return
            if isinstance(expected, (list, tuple)):
                assert len(actual) == len(expected)
                for actual_item, expected_item in zip(actual, expected, strict=True):
                    assert_structure(actual_item, expected_item, device)


        detector = RFDETRNano()
        checkpoint_path = Path(os.environ["RF_HOME"]) / "rf-detr-nano.pth"
        hasher = hashlib.sha256()
        with checkpoint_path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        assert digest == "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"  # pragma: allowlist secret
        cpu_model = RFDETRTensorModel(detector.model.model).eval()
        gpu_model = copy.deepcopy(cpu_model).cuda().eval() if torch.cuda.is_available() else None
        cpu_example = torch.zeros(2, 3, 384, 384)
        with torch.no_grad():
            cpu_seed_graph = prepare_graph(cpu_model, cpu_example)
            cpu_seed_dag = cpu_seed_graph.compute_dag()
            cpu_seed_partition = cpu_seed_dag.to_partition(())
            cpu_seed = cpu_seed_graph.materialize(cpu_seed_graph.analyze(cpu_seed_partition))
        assert cpu_seed.trace_graph.shape_program.unresolved == {}
        assert cpu_seed.traced_batch_size == 1
        assert cpu_seed.trace_graph.shape_program.witness_batch_sizes == (2,)
        assert cpu_seed.batch_validation["status"] == "passed"
        gpu_seed = None
        if gpu_model is not None:
            with torch.no_grad():
                gpu_seed_graph = prepare_graph(gpu_model, torch.zeros(2, 3, 384, 384, device="cuda"))
                gpu_seed_dag = gpu_seed_graph.compute_dag()
                gpu_seed_partition = gpu_seed_dag.to_partition(())
                gpu_seed = gpu_seed_graph.materialize(gpu_seed_graph.analyze(gpu_seed_partition))
            assert [node.canonical_id for node in gpu_seed.trace_graph.compute_nodes] == [
                node.canonical_id for node in cpu_seed.trace_graph.compute_nodes
            ]
        partitions = smallest_transfer_partitions(cpu_seed)
        assert len(partitions) == 100
        progress_prefix = os.environ.get("TORCHLENS_RFDETR_PROGRESS")
        points = partitions
        cpu_inputs = {batch: torch.zeros(batch, 3, 384, 384) for batch in (1, 2, 3)}
        with torch.inference_mode():
            cpu_expected = {batch: cpu_model(value) for batch, value in cpu_inputs.items()}
            gpu_inputs = (
                {batch: torch.zeros(batch, 3, 384, 384, device="cuda") for batch in (1, 2, 3)}
                if gpu_model is not None
                else {}
            )
            gpu_expected = (
                {batch: gpu_model(value) for batch, value in gpu_inputs.items()}
                if gpu_model is not None
                else {}
            )
            counts = {"cpu": 0, "cuda": 0, "cpu_cuda": 0, "cuda_cpu": 0}
            for point in points:
                cpu_runtime = cpu_seed.repartition(point)
                gpu_runtime = None if gpu_seed is None else gpu_seed.repartition(point)
                for batch in (1, 2, 3):
                    cpu_actual = cpu_runtime.replay(cpu_inputs[batch])
                    assert_same(cpu_actual, cpu_expected[batch])
                    del cpu_actual
                    counts["cpu"] += 1
                    if gpu_runtime is not None:
                        gpu_actual = gpu_runtime.replay(gpu_inputs[batch])
                        assert_same(gpu_actual, gpu_expected[batch])
                        del gpu_actual
                        counts["cuda"] += 1
                if gpu_runtime is None:
                    continue
                for batch in (1, 3):
                    cpu_boundary = cpu_runtime.run_prefix(cpu_inputs[batch])
                    assert cpu_boundary.tensors.keys() == cpu_boundary.spec.keys()
                    cpu_to_cuda = gpu_runtime.run_suffix(
                        cpu_boundary.to("cuda", adapter=gpu_runtime.adapter)
                    )
                    assert_structure(cpu_to_cuda, gpu_expected[batch], "cuda")
                    counts["cpu_cuda"] += 1
                    del cpu_boundary, cpu_to_cuda
                    gpu_boundary = gpu_runtime.run_prefix(gpu_inputs[batch])
                    assert gpu_boundary.tensors.keys() == gpu_boundary.spec.keys()
                    cuda_to_cpu = cpu_runtime.run_suffix(
                        gpu_boundary.to("cpu", adapter=cpu_runtime.adapter)
                    )
                    assert_structure(cuda_to_cpu, cpu_expected[batch], "cpu")
                    counts["cuda_cpu"] += 1
                    del gpu_boundary, cuda_to_cpu
                if progress_prefix:
                    with open(f"{progress_prefix}.dag.log", "a") as progress:
                        print(counts["cpu"] // 3, len(points), point, counts, file=progress)
        expected_counts = {
            "cpu": len(points) * 3,
            "cuda": len(points) * 3,
            "cpu_cuda": len(points) * 2,
            "cuda_cpu": len(points) * 2,
        }
        assert counts["cpu"] == expected_counts["cpu"]
        if gpu_seed is not None:
            assert counts == expected_counts
        print(len(points), counts, flush=True)
        assert not cpu_seed.retains_trace
        if gpu_seed is not None:
            assert not gpu_seed.retains_trace
        os._exit(0)

        """,
        timeout=7200,
    )


def test_torchvision_ops_mlp_smallest_dag_partitions_cross_batch_and_device() -> None:
    """The smallest native DAG partitions replay for a torchvision module block."""
    _skip_unless_enabled()
    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(0)
    for device in _torch_test_devices():
        model = torchvision.ops.MLP(in_channels=6, hidden_channels=[8, 4], dropout=0.0).to(device)
        model.eval()
        x = torch.ones((2, 6), device=device)
        seed_graph = prepare_graph(model, x)
        seed_dag = seed_graph.compute_dag()
        seed_partition = seed_dag.to_partition(
            close_call_ids(
                seed_dag,
                frozenset(
                    call.call_id
                    for call in seed_dag.nodes[: (len(seed_dag.nodes) - 1) * 50 // 100 + 1]
                ),
            )
        )
        seed_runtime = seed_graph.materialize(seed_graph.analyze(seed_partition))
        for boundary in smallest_transfer_partitions(seed_runtime):
            runtime = seed_runtime.repartition(boundary)
            for batch in (1, 3):
                replay_x = torch.ones((batch, 6), device=device)
                with torch.no_grad():
                    split_output = runtime.replay(replay_x)
                    full_output = model(replay_x)
                assert torch.allclose(split_output, full_output, atol=1e-05, rtol=0.0001)
