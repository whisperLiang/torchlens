"""Opt-in PyTorch real-world split replay tests."""

from __future__ import annotations

import os
from importlib import import_module

import pytest
import torch
from real_model_helpers import (
    _run_backend_subprocess,
    _skip_if_module_missing,
    _skip_unless_enabled,
)
from v2_helpers import split_request

import torchlens as tl

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def _skip_unless_rfdetr_exhaustive() -> None:
    """Skip the RF-DETR all-boundary matrix unless explicitly requested."""

    if os.environ.get("TORCHLENS_RFDETR_EXHAUSTIVE") != "1":
        pytest.skip("TORCHLENS_RFDETR_EXHAUSTIVE=1 enables the RF-DETR all-boundary matrix.")


def _skip_unless_yolov8_exhaustive() -> None:
    """Skip the YOLOv8n all-boundary matrix unless explicitly requested."""

    if os.environ.get("TORCHLENS_YOLOV8_EXHAUSTIVE") != "1":
        pytest.skip("TORCHLENS_YOLOV8_EXHAUSTIVE=1 enables the YOLOv8n all-boundary matrix.")


def _all_compute_split_boundaries(runtime: object) -> list[str]:
    """Return before/after split boundaries for every compute node in a runtime."""

    return [
        f"{kind}:{node.canonical_id}"
        for node in runtime.trace_graph.compute_nodes
        for kind in ("before", "after")
    ]


def _torch_test_devices() -> list[torch.device]:
    """Return the torch devices covered by cross-device split tests."""

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


def test_resnet18_representative_splits_skip_cleanly_when_disabled() -> None:
    """ResNet18 split replay validates a small number of representative boundaries."""

    _skip_unless_enabled()
    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)

    for spec in (split_request("25%"), split_request("50%"), split_request("before:fc")):
        runtime = tl.split.prepare(model, x, spec)
        boundary = runtime.run_prefix(x)
        runtime.validate_boundary(boundary)
        assert torch.allclose(runtime.run_suffix(boundary), model(x), atol=1e-4, rtol=1e-3)


def test_tiny_transformer_config_split_skip_cleanly_when_disabled() -> None:
    """A no-download transformer config exercises attention-style structure."""

    _skip_unless_enabled()
    _skip_if_module_missing("transformers")
    transformers = import_module("transformers")
    config = transformers.DistilBertConfig(
        vocab_size=128,
        n_layers=1,
        dim=32,
        hidden_dim=64,
        n_heads=4,
    )
    model = transformers.DistilBertModel(config).eval()
    x = torch.randint(0, 128, (2, 8))

    runtime = tl.split.prepare(model, x, split_request("50%"))
    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    split_output = runtime.run_suffix(boundary).last_hidden_state
    full_output = model(x).last_hidden_state

    assert torch.allclose(split_output, full_output, atol=1e-4, rtol=1e-3)


def test_torch_transformer_cross_batch_and_device() -> None:
    """A real Transformer output container replays across batch and device."""

    _skip_unless_enabled()
    _skip_if_module_missing("transformers")
    transformers = import_module("transformers")
    config = transformers.DistilBertConfig(
        vocab_size=128,
        n_layers=2,
        dim=32,
        hidden_dim=64,
        n_heads=4,
    )
    torch.manual_seed(0)

    for device in _torch_test_devices():
        model = transformers.DistilBertModel(config).to(device).eval()
        x = torch.randint(0, 128, (2, 12), device=device)
        for boundary in ("25%", "60%", "85%"):
            runtime = tl.split.prepare(
                model,
                x,
                split_request(boundary),
            )
            for batch in (1, 4):
                replay_x = torch.randint(0, 128, (batch, 12), device=device)
                with torch.no_grad():
                    split_output = runtime.replay(replay_x)
                    full_output = model(replay_x)
                assert torch.allclose(
                    split_output.last_hidden_state,
                    full_output.last_hidden_state,
                    atol=1e-4,
                    rtol=1e-3,
                )


def test_torchvision_extra_real_models_cross_batch_and_device() -> None:
    """Additional torchvision models replay representative splits across batch/device."""

    _skip_unless_enabled()
    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(0)
    model_cases = (
        (
            "mobilenet_v2",
            lambda: torchvision.models.mobilenet_v2(weights=None, num_classes=10),
            (2, 3, 64, 64),
            ("20%", "50%", "80%"),
        ),
        (
            "squeezenet1_0",
            lambda: torchvision.models.squeezenet1_0(weights=None, num_classes=10),
            (2, 3, 64, 64),
            ("20%", "50%", "80%"),
        ),
        (
            "efficientnet_b0",
            lambda: torchvision.models.efficientnet_b0(weights=None, num_classes=10),
            (2, 3, 64, 64),
            ("20%", "50%", "80%"),
        ),
        (
            "convnext_tiny",
            lambda: torchvision.models.convnext_tiny(weights=None, num_classes=10),
            (2, 3, 64, 64),
            ("20%", "50%", "80%"),
        ),
    )

    for device in _torch_test_devices():
        for _name, factory, shape, boundaries in model_cases:
            model = factory().to(device).eval()
            x = torch.ones(shape, device=device)
            for boundary in boundaries:
                runtime = tl.split.prepare(
                    model,
                    x,
                    split_request(boundary),
                )
                for batch in (1, 3):
                    replay_x = torch.ones((batch, *shape[1:]), device=device)
                    with torch.no_grad():
                        split_output = runtime.replay(replay_x)
                        full_output = model(replay_x)
                    atol = 5e-4 if device.type == "cuda" else 1e-4
                    assert torch.allclose(split_output, full_output, atol=atol, rtol=1e-3)


def test_yolo26_detection_split_replay_cross_batch_and_device() -> None:
    """YOLO26 official weights cover full detection and backbone split replay."""

    _skip_unless_enabled()
    _skip_if_module_missing("ultralytics")
    _run_backend_subprocess(
        """
        import os
        import hashlib
        from pathlib import Path

        import torch
        from torch import nn

        import torchlens as tl
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        # Initialize Dynamo/Triton before Ultralytics can import TensorFlow.
        # The default probe is lazy and does not initialize an unseen Dynamo;
        # loading its LLVM extension after TensorFlow can crash this process.
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

        def assert_same(left, right, *, atol=1e-4):
            left_leaves = tensor_leaves(left)
            right_leaves = tensor_leaves(right)
            assert len(left_leaves) == len(right_leaves)
            for actual, expected in zip(left_leaves, right_leaves, strict=True):
                torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-3)

        # End-to-end YOLO26 detection-head replay.  These fixed-batch checks
        # include the model's tuple/dict output and the x[..., :] decode path.
        full_input = torch.zeros(2, 3, 160, 160)
        # 75% keeps the complete decode region on the prefix side; an earlier
        # boundary can expose Ultralytics' auxiliary ``full`` shape seed whose
        # scalar provenance is not present in TorchLens' portable graph yet.
        for boundary in ("75%",):
            runtime = tl.split.prepare(
                detector,
                full_input,
                split_request(boundary, backend="torch"),
            )
            with torch.no_grad():
                assert_same(runtime.replay(full_input), detector(full_input))

        # The first eleven official YOLO26 layers are the backbone and have a
        # tensor output.  Use this real pretrained submodule for dynamic batch
        # and CPU-prefix/GPU-suffix coverage without conflating detection decode
        # shape policy with the backbone's batch-polymorphic contract.
        backbone = nn.Sequential(*list(detector.model[:11])).eval()
        trace_input = torch.zeros(2, 3, 160, 160)
        runtime = tl.split.prepare(
            backbone,
            trace_input,
            split_request(
                "50%",
                backend="torch",
                live_param_sources=True,
            ),
        )
        for batch in (1, 2, 3):
            replay_input = torch.zeros(batch, 3, 160, 160)
            with torch.no_grad():
                assert_same(runtime.replay(replay_input), backbone(replay_input))

        if not torch.cuda.is_available():
            raise SystemExit(75)
        for batch in (1, 3):
            backbone.cpu()
            cpu_input = torch.zeros(batch, 3, 160, 160)
            boundary = runtime.run_prefix(cpu_input)
            backbone.cuda()
            gpu_boundary = boundary.to("cuda", adapter=runtime.adapter)
            gpu_input = torch.zeros(batch, 3, 160, 160, device="cuda")
            with torch.no_grad():
                assert_same(
                    runtime.run_suffix(gpu_boundary),
                    backbone(gpu_input),
                    atol=5e-3,
                )

        os._exit(0)
        """,
        timeout=300,
    )


def test_yolov8n_all_split_nodes_cross_batch_and_device() -> None:
    """YOLOv8n replays all 558 boundaries across batch and device."""

    _skip_unless_enabled()
    _skip_unless_yolov8_exhaustive()
    _skip_if_module_missing("ultralytics")
    _run_backend_subprocess(
        """
        import copy
        import os

        import torch

        import torchlens as tl
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        torch.set_num_threads(1)
        get_dynamo_optimized_module_type(force_probe=True)
        from ultralytics import YOLO

        def assert_same(actual, expected):
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
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
        request = split_request("50%")
        cpu_seed = tl.split.prepare(
            cpu_model,
            torch.zeros(2, 3, 160, 160),
            request,
        )
        # The canonical B=1 capture retains 279 compute nodes for this graph.
        assert len(cpu_seed.trace_graph.compute_nodes) == 279
        assert cpu_seed.trace_graph.shape_program.unresolved == {}
        assert cpu_seed.traced_batch_size == 1
        assert cpu_seed.trace_graph.shape_program.witness_batch_sizes == (2,)
        assert cpu_seed.batch_validation["status"] == "passed"

        gpu_seed = None
        if gpu_model is not None:
            gpu_seed = tl.split.prepare(
                gpu_model,
                torch.zeros(2, 3, 160, 160, device="cuda"),
                request,
            )
            assert [
                node.canonical_id for node in gpu_seed.trace_graph.compute_nodes
            ] == [node.canonical_id for node in cpu_seed.trace_graph.compute_nodes]

        all_points = [
            point
            for node in cpu_seed.trace_graph.compute_nodes
            for point in (tl.split.before(node.canonical_id), tl.split.after(node.canonical_id))
        ]
        assert len(all_points) == 558
        partition_count = int(os.environ.get("TORCHLENS_YOLOV8_PARTITIONS", "1"))
        partition_index = int(os.environ.get("TORCHLENS_YOLOV8_PARTITION", "0"))
        assert partition_count >= 1
        assert 0 <= partition_index < partition_count
        points = all_points[partition_index::partition_count]
        cpu_inputs = {
            batch: torch.zeros(batch, 3, 160, 160) for batch in (1, 2, 3)
        }

        with torch.inference_mode():
            cpu_expected = {batch: cpu_model(value) for batch, value in cpu_inputs.items()}
            gpu_inputs = (
                {
                    batch: torch.zeros(batch, 3, 160, 160, device="cuda")
                    for batch in (1, 2, 3)
                }
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
                cpu_runtime = cpu_seed.at(point)
                gpu_runtime = None if gpu_seed is None else gpu_seed.at(point)
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
        print(partition_index, partition_count, len(points), counts, flush=True)
        os._exit(0)
        """,
        timeout=7200,
    )


def test_rfdetr_detection_split_replay_fixed_batch() -> None:
    """RF-DETR-N official weights replay the full detection core at boundaries."""

    _skip_unless_enabled()
    _skip_if_module_missing("rfdetr")
    _run_backend_subprocess(
        """
        import os
        import hashlib
        from pathlib import Path

        import torch
        from torch import nn

        import torchlens as tl
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        torch.set_num_threads(1)
        os.environ.setdefault(
            "RF_HOME",
            str(Path.home() / ".cache" / "torchlens" / "models" / "rfdetr"),
        )
        get_dynamo_optimized_module_type(force_probe=True)
        from rfdetr import RFDETRNano
        from rfdetr.utilities.tensors import NestedTensor

        class RFDETRTensorModel(nn.Module):
            # Expose the official RF-DETR core through a tensor-only ABI.

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

        detector = RFDETRNano()
        checkpoint_path = Path(os.environ["RF_HOME"]) / "rf-detr-nano.pth"
        hasher = hashlib.sha256()
        with checkpoint_path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        assert digest == "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"  # pragma: allowlist secret
        model = RFDETRTensorModel(detector.model.model).eval()
        inputs = torch.zeros(2, 3, 384, 384)

        def tensor_leaves(value):
            if isinstance(value, torch.Tensor):
                return [value]
            if isinstance(value, dict):
                return [leaf for key in sorted(value) for leaf in tensor_leaves(value[key])]
            if isinstance(value, (list, tuple)):
                return [leaf for item in value for leaf in tensor_leaves(item)]
            return []

        with torch.no_grad():
            seed = tl.split.prepare(
                model,
                inputs,
                split_request("25%", backend="torch"),
            )
            expected = model(inputs)
            for percent in (25, 50, 75):
                runtime = seed.at(tl.split.percent(percent))
                actual = runtime.replay(inputs)
                actual_leaves = tensor_leaves(actual)
                expected_leaves = tensor_leaves(expected)
                assert len(actual_leaves) == len(expected_leaves)
                for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
                    torch.testing.assert_close(actual_leaf, expected_leaf, atol=1e-4, rtol=1e-3)
        assert not seed.retains_trace

        os._exit(0)
        """,
        timeout=420,
    )


def test_rfdetr_all_split_nodes_cross_batch_and_device() -> None:
    """RF-DETR-N replays every before/after boundary across batch and device."""

    _skip_unless_enabled()
    _skip_unless_rfdetr_exhaustive()
    _skip_if_module_missing("rfdetr")
    _run_backend_subprocess(
        """
        import copy
        import os
        import hashlib
        from pathlib import Path

        import torch
        from torch import nn

        import torchlens as tl
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        torch.set_num_threads(1)
        os.environ.setdefault(
            "RF_HOME",
            str(Path.home() / ".cache" / "torchlens" / "models" / "rfdetr"),
        )
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
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
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

        request = split_request("50%")
        cpu_example = torch.zeros(2, 3, 384, 384)
        # This matrix tests inference. Do not retain backward graphs for the
        # B=1 capture and B=2 probe alongside both CPU and GPU runtimes.
        with torch.no_grad():
            cpu_seed = tl.split.prepare(cpu_model, cpu_example, request)
        # The admitted Torch 2.8 lane captures two additional operations.
        # Both inventories are replay-checked against the official
        # core below; do not treat the Torch 2.13 count as cross-version identity.
        expected_compute_nodes = 860 if torch.__version__.split('.')[:2] == ['2', '8'] else 858
        assert len(cpu_seed.trace_graph.compute_nodes) == expected_compute_nodes
        assert cpu_seed.trace_graph.shape_program.unresolved == {}
        assert cpu_seed.traced_batch_size == 1
        assert cpu_seed.trace_graph.shape_program.witness_batch_sizes == (2,)
        assert cpu_seed.batch_validation["status"] == "passed"

        gpu_seed = None
        if gpu_model is not None:
            with torch.no_grad():
                gpu_seed = tl.split.prepare(
                    gpu_model,
                    torch.zeros(2, 3, 384, 384, device="cuda"),
                    request,
                )
            assert [
                node.canonical_id for node in gpu_seed.trace_graph.compute_nodes
            ] == [node.canonical_id for node in cpu_seed.trace_graph.compute_nodes]

        all_points = [
            point
            for node in cpu_seed.trace_graph.compute_nodes
            for point in (tl.split.before(node.canonical_id), tl.split.after(node.canonical_id))
        ]
        assert len(all_points) == 2 * expected_compute_nodes
        partition_count = int(os.environ.get("TORCHLENS_RFDETR_PARTITIONS", "1"))
        partition_index = int(os.environ.get("TORCHLENS_RFDETR_PARTITION", "0"))
        assert partition_count >= 1
        assert 0 <= partition_index < partition_count
        points = all_points[partition_index::partition_count]
        cpu_inputs = {
            batch: torch.zeros(batch, 3, 384, 384) for batch in (1, 2, 3)
        }
        with torch.inference_mode():
            cpu_expected = {batch: cpu_model(value) for batch, value in cpu_inputs.items()}
            gpu_inputs = (
                {
                    batch: torch.zeros(batch, 3, 384, 384, device="cuda")
                    for batch in (1, 2, 3)
                }
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
                cpu_runtime = cpu_seed.at(point)
                gpu_runtime = None if gpu_seed is None else gpu_seed.at(point)
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

        expected_counts = {
            "cpu": len(points) * 3,
            "cuda": len(points) * 3,
            "cpu_cuda": len(points) * 2,
            "cuda_cpu": len(points) * 2,
        }
        assert counts["cpu"] == expected_counts["cpu"]
        if gpu_seed is not None:
            assert counts == expected_counts
        print(partition_index, partition_count, len(points), counts, flush=True)
        assert not cpu_seed.retains_trace
        if gpu_seed is not None:
            assert not gpu_seed.retains_trace
        os._exit(0)
        """,
        timeout=7200,
    )


def test_torchvision_ops_mlp_all_split_nodes_cross_batch_and_device() -> None:
    """Every compute-node before/after split replays for a torchvision module block."""

    _skip_unless_enabled()
    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(0)

    for device in _torch_test_devices():
        model = torchvision.ops.MLP(
            in_channels=6,
            hidden_channels=[8, 4],
            dropout=0.0,
        ).to(device)
        model.eval()
        x = torch.ones((2, 6), device=device)
        seed_runtime = tl.split.prepare(
            model,
            x,
            split_request("50%"),
        )

        for boundary in _all_compute_split_boundaries(seed_runtime):
            runtime = tl.split.prepare(
                model,
                x,
                split_request(boundary),
            )
            for batch in (1, 3):
                replay_x = torch.ones((batch, 6), device=device)
                with torch.no_grad():
                    split_output = runtime.replay(replay_x)
                    full_output = model(replay_x)
                assert torch.allclose(split_output, full_output, atol=1e-5, rtol=1e-4)
