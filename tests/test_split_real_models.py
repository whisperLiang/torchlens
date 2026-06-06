"""Optional real-model smoke tests for native split replay."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import gc
import importlib.util
import os
from pathlib import Path
import subprocess
import textwrap
from typing import Any
import urllib.error
import urllib.request

import pytest
import torch

import torchlens as tl
from torchlens.split.codegen import build_segments
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.planner import plan_split
from torchlens.split.runtime import SplitRuntime
from torchlens.split.trace_graph import trace_graph_from_model_log
from torchlens.split.validation import SplitSweepCaseResult
from torchlens.split.validation import assert_split_sweep_coverage
from torchlens.split.validation import assert_split_unsupported_error_context
from torchlens.split.validation import assert_canonical_abi_equivalent
from torchlens.split.validation import boundary_liveness_zero_probe
from torchlens.split.validation import format_split_sweep_coverage_summary
from torchlens.split.validation import make_split_sweep_case_result
from torchlens.split.validation import summarize_split_sweep_coverage
from torchlens.split.validation import write_split_sweep_coverage_report


REAL_MODEL_WEIGHT_CACHE_DIR = (
    Path(os.environ["TORCHLENS_REAL_MODEL_WEIGHT_CACHE"])
    if "TORCHLENS_REAL_MODEL_WEIGHT_CACHE" in os.environ
    else Path(__file__).resolve().parent / "artifacts" / "real_model_weights"
)
YOLO26N_WEIGHTS_URL = os.environ.get(
    "TORCHLENS_YOLO26N_URL",
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
)
YOLO26N_WEIGHTS = REAL_MODEL_WEIGHT_CACHE_DIR / "yolo26n.pt"
PLANK_ROAD_ROOT = Path("/home/whisperliang/Plank-road")
TINYNEXT_SOURCE = PLANK_ROAD_ROOT / "model_management" / "tinynext.py"
RFDETR_PYTHON = PLANK_ROAD_ROOT / ".venv" / "bin" / "python"
RFDETR_NANO_WEIGHTS_URL = os.environ.get(
    "TORCHLENS_RFDETR_NANO_URL",
    "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
)
RFDETR_WEIGHTS = REAL_MODEL_WEIGHT_CACHE_DIR / "rf-detr-nano.pth"
REAL_MODEL_DYNAMIC_BATCHES = (1, 2, 4, 8, 32)
REAL_MODEL_CPU_DYNAMIC_BATCHES = (1, 2)
REAL_MODEL_DYNAMIC_BATCH_RANGE = (1, 32)
SPLIT_SWEEP_COVERAGE_DIR = Path(__file__).resolve().parent / "artifacts" / "split_sweep_coverage"
YOLO26_DYNAMIC_BATCH_MIN_START_CUDA_FREE_GIB = 12.0
YOLO26_DYNAMIC_BATCH_MIN_CASE_CUDA_FREE_GIB = 6.0
YOLO26_DYNAMIC_BATCH_MIN_SYSTEM_FREE_GIB = 8.0


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yolo26n_detection_head_boundary_is_live_on_cuda() -> None:
    """YOLO26n split after detection head preserves all live boundary tensors."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = _download_weight_file(
        url=YOLO26N_WEIGHTS_URL,
        destination=YOLO26N_WEIGHTS,
        model_name="YOLO26n",
    )

    model = ultralytics.YOLO(str(weight_path)).model.eval().cuda()
    x = torch.randn(1, 3, 640, 640, device="cuda")
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:model.23", dynamic_batch=(1, 4)),
    )
    boundary = runtime.run_prefix(x)

    assert len(boundary.tensors) >= 5
    assert any("detach" in tensor_id for tensor_id in boundary.tensors)
    skip_tensor_ids = [
        tensor_id for tensor_id, spec in boundary.spec.items() if spec.role == "skip"
    ]
    assert skip_tensor_ids

    with torch.no_grad():
        split_output = runtime.run_suffix(boundary)
        full_output = model(x)
    assert _tree_allclose(split_output, full_output, atol=1e-4, rtol=1e-4)

    live = boundary_liveness_zero_probe(runtime, boundary, atol=1e-5, rtol=1e-5)
    assert all(live[tensor_id] for tensor_id in skip_tensor_ids)
    assert any(live.values())


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yolo26n_cpu_prefix_cuda_suffix_across_split_points() -> None:
    """YOLO26n CPU prefix boundaries can drive CUDA suffixes at several split points."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = _download_weight_file(
        url=YOLO26N_WEIGHTS_URL,
        destination=YOLO26N_WEIGHTS,
        model_name="YOLO26n",
    )

    with _cuda_tf32_disabled():
        cpu_model = ultralytics.YOLO(str(weight_path)).model.eval().cpu()
        cuda_model = ultralytics.YOLO(str(weight_path)).model.eval().cuda()
        cuda_model.load_state_dict(cpu_model.state_dict())

        x_cpu = torch.randn(1, 3, 640, 640)
        x_cuda = x_cpu.cuda()
        cpu_graph = _trace_graph(cpu_model, x_cpu)
        cuda_graph = _trace_graph(cuda_model, x_cuda)

        with torch.no_grad():
            full_output = cuda_model(x_cuda)

        for boundary in ("percent:25", "percent:50", "percent:75"):
            spec = tl.SplitSpec(boundary=boundary, dynamic_batch=(1, 4))
            cpu_runtime = _runtime_from_graph(cpu_model, cpu_graph, spec)
            cuda_runtime = _runtime_from_graph(cuda_model, cuda_graph, spec)
            assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)

            cpu_boundary = cpu_runtime.run_prefix(x_cpu)
            cuda_boundary = cuda_runtime.run_prefix(x_cuda)
            _assert_boundary_devices(cpu_boundary, "cpu")
            _assert_boundary_devices(cuda_boundary, "cuda")
            _assert_boundary_shapes(cpu_boundary, cuda_boundary)

            with torch.no_grad():
                cuda_split_output = cuda_runtime.run_suffix(cuda_boundary)
                cross_device_output = cuda_runtime.run_suffix(cpu_boundary)
            _assert_cross_device_output_accepted(
                cuda_split_output,
                full_output,
                device="cuda",
                atol=1e-4,
                rtol=1e-4,
            )
            _assert_cross_device_output_accepted(
                cross_device_output,
                full_output,
                device="cuda",
                atol=1e-4,
                rtol=1e-4,
            )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yolo26n_cpu_prefix_cuda_suffix_all_compute_split_points() -> None:
    """Sweep every YOLO26n compute split point with CPU prefixes and CUDA suffixes."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = _download_weight_file(
        url=YOLO26N_WEIGHTS_URL,
        destination=YOLO26N_WEIGHTS,
        model_name="YOLO26n",
    )

    _skip_if_system_memory_below(YOLO26_DYNAMIC_BATCH_MIN_SYSTEM_FREE_GIB)
    device = _select_cuda_device_with_free_memory(
        min_free_gib=YOLO26_DYNAMIC_BATCH_MIN_START_CUDA_FREE_GIB
    )

    with _cuda_tf32_disabled():
        cpu_model = ultralytics.YOLO(str(weight_path)).model.eval().cpu()
        cuda_model = ultralytics.YOLO(str(weight_path)).model.eval().to(device)
        cuda_model.load_state_dict(cpu_model.state_dict())
        trace_cpu = torch.randn(1, 3, 640, 640)
        trace_cuda = trace_cpu.to(device)
        cpu_graph = _trace_graph(cpu_model, trace_cpu)
        cuda_graph = _trace_graph(cuda_model, trace_cuda)
        del trace_cpu, trace_cuda
        _clear_cuda_working_set(device)
        targets = _compute_split_targets(cpu_graph)
        assert len(targets) >= 100

        cases: list[SplitSweepCaseResult] = []
        for target_index, target in enumerate(targets, start=1):
            if target_index == 1 or target_index % 25 == 0:
                print(
                    f"[YOLO26n CPU->CUDA dynamic batch] target "
                    f"{target_index}/{len(targets)}: {target}",
                    flush=True,
                )
            boundary = f"after:{target}"
            batch_statuses: dict[str, Any] = {}
            tested_batches: list[int] = []
            cpu_runtime = None
            cuda_runtime = None
            cpu_boundary = None
            cuda_boundary = None
            full_output = None
            cuda_split_output = None
            cross_device_output = None
            x_cpu = None
            x_cuda = None

            try:
                spec = tl.SplitSpec(boundary=boundary, dynamic_batch=REAL_MODEL_DYNAMIC_BATCH_RANGE)
                cpu_runtime = _runtime_from_graph(cpu_model, cpu_graph, spec)
                cuda_runtime = _runtime_from_graph(cuda_model, cuda_graph, spec)
                assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)

                for batch_size in REAL_MODEL_CPU_DYNAMIC_BATCHES:
                    _skip_if_cuda_memory_below(
                        device,
                        min_free_gib=YOLO26_DYNAMIC_BATCH_MIN_CASE_CUDA_FREE_GIB,
                        context=f"{boundary} batch={batch_size}",
                    )
                    tested_batches.append(batch_size)
                    x_cpu = torch.randn(batch_size, 3, 640, 640)
                    x_cuda = x_cpu.to(device)
                    cpu_boundary = cpu_runtime.run_prefix(x_cpu)
                    cuda_boundary = cuda_runtime.run_prefix(x_cuda)
                    _assert_boundary_devices(cpu_boundary, "cpu")
                    _assert_boundary_devices(cuda_boundary, "cuda")
                    _assert_boundary_shapes(cpu_boundary, cuda_boundary)
                    with torch.no_grad():
                        full_output = cuda_model(x_cuda)
                        cuda_split_output = cuda_runtime.run_suffix(cuda_boundary)
                        cross_device_output = cuda_runtime.run_suffix(cpu_boundary)
                    _assert_model_output_accepted(
                        cuda_split_output,
                        full_output,
                        device="cuda",
                        atol=1e-4,
                        rtol=1e-4,
                    )
                    _assert_cross_device_output_accepted(
                        cross_device_output,
                        full_output,
                        device="cuda",
                        atol=1e-4,
                        rtol=1e-4,
                    )
                    batch_statuses[str(batch_size)] = "supported"
                    del x_cpu, x_cuda, cpu_boundary, cuda_boundary
                    del full_output, cuda_split_output, cross_device_output
                    x_cpu = None
                    x_cuda = None
                    cpu_boundary = None
                    cuda_boundary = None
                    full_output = None
                    cuda_split_output = None
                    cross_device_output = None
                    _clear_cuda_working_set(device)
            except SplitUnsupportedError as exc:
                assert_split_unsupported_error_context(exc, boundary)
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = "unsupported"
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="unsupported",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            except RuntimeError as exc:
                _clear_cuda_working_set(device)
                if _is_cuda_oom(exc):
                    if tested_batches:
                        batch_statuses[str(tested_batches[-1])] = "failed"
                    cases.append(
                        make_split_sweep_case_result(
                            model_name="YOLO26n",
                            split_target=target,
                            boundary=boundary,
                            status="failed",
                            batch_sizes_tested=tuple(tested_batches),
                            batch_statuses=batch_statuses,
                            exc=exc,
                        )
                    )
                    report = summarize_split_sweep_coverage("YOLO26n", cases)
                    write_split_sweep_coverage_report(
                        report,
                        SPLIT_SWEEP_COVERAGE_DIR / "yolo26n_cpu_prefix_cuda_suffix_compute.json",
                    )
                    pytest.fail(
                        "YOLO26n CPU-prefix/CUDA-suffix dynamic batch sweep hit CUDA OOM "
                        f"at {boundary} with batch statuses {batch_statuses}. "
                        "Partial coverage report was written."
                    )
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = "failed"
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="failed",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            except Exception as exc:
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = "failed"
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="failed",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            else:
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="supported",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                    )
                )
            finally:
                del cpu_runtime, cuda_runtime, cpu_boundary, cuda_boundary
                del full_output, cuda_split_output, cross_device_output
                del x_cpu, x_cuda
                _clear_cuda_working_set(device)

        report = summarize_split_sweep_coverage("YOLO26n", cases)
        write_split_sweep_coverage_report(
            report,
            SPLIT_SWEEP_COVERAGE_DIR / "yolo26n_cpu_prefix_cuda_suffix_compute.json",
        )
        assert_split_sweep_coverage(report, 0.5)
        print(format_split_sweep_coverage_summary(report))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yolo26n_all_compute_split_points_on_cuda() -> None:
    """Sweep every YOLO26n compute split point across all dynamic CUDA batches."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = _download_weight_file(
        url=YOLO26N_WEIGHTS_URL,
        destination=YOLO26N_WEIGHTS,
        model_name="YOLO26n",
    )

    _skip_if_system_memory_below(YOLO26_DYNAMIC_BATCH_MIN_SYSTEM_FREE_GIB)
    device = _select_cuda_device_with_free_memory(
        min_free_gib=YOLO26_DYNAMIC_BATCH_MIN_START_CUDA_FREE_GIB
    )

    with _cuda_tf32_disabled():
        cuda_model = ultralytics.YOLO(str(weight_path)).model.eval().to(device)
        trace_tensor = torch.randn(1, 3, 640, 640, device=device)
        cuda_graph = _trace_graph(cuda_model, trace_tensor)
        del trace_tensor
        _clear_cuda_working_set(device)
        targets = _compute_split_targets(cuda_graph)
        assert len(targets) >= 100

        cases: list[SplitSweepCaseResult] = []
        for target_index, target in enumerate(targets, start=1):
            if target_index == 1 or target_index % 25 == 0:
                print(
                    f"[YOLO26n dynamic batch] target {target_index}/{len(targets)}: {target}",
                    flush=True,
                )
            boundary = f"after:{target}"
            batch_statuses: dict[str, Any] = {}
            tested_batches: list[int] = []
            cuda_runtime = None

            try:
                spec = tl.SplitSpec(boundary=boundary, dynamic_batch=REAL_MODEL_DYNAMIC_BATCH_RANGE)
                cuda_runtime = _runtime_from_graph(cuda_model, cuda_graph, spec)

                for batch_size in REAL_MODEL_DYNAMIC_BATCHES:
                    _skip_if_cuda_memory_below(
                        device,
                        min_free_gib=YOLO26_DYNAMIC_BATCH_MIN_CASE_CUDA_FREE_GIB,
                        context=f"{boundary} batch={batch_size}",
                    )
                    tested_batches.append(batch_size)
                    _assert_cuda_split_replay_matches(
                        cuda_model,
                        cuda_runtime,
                        batch_size=batch_size,
                        sample_shape=(3, 640, 640),
                        device=device,
                        atol=1e-4,
                        rtol=1e-4,
                    )
                    batch_statuses[str(batch_size)] = "supported"
                    _clear_cuda_working_set(device)
            except SplitUnsupportedError as exc:
                assert_split_unsupported_error_context(exc, boundary)
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = "unsupported"
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="unsupported",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            except RuntimeError as exc:
                _clear_cuda_working_set(device)
                if _is_cuda_oom(exc):
                    if tested_batches:
                        batch_statuses[str(tested_batches[-1])] = "failed"
                    cases.append(
                        make_split_sweep_case_result(
                            model_name="YOLO26n",
                            split_target=target,
                            boundary=boundary,
                            status="failed",
                            batch_sizes_tested=tuple(tested_batches),
                            batch_statuses=batch_statuses,
                            exc=exc,
                        )
                    )
                    report = summarize_split_sweep_coverage("YOLO26n", cases)
                    write_split_sweep_coverage_report(
                        report,
                        SPLIT_SWEEP_COVERAGE_DIR / "yolo26n_cuda_dynamic_batch_compute.json",
                    )
                    pytest.fail(
                        "YOLO26n dynamic batch sweep hit CUDA OOM at "
                        f"{boundary} with batch statuses {batch_statuses}. "
                        "Partial coverage report was written."
                    )
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = "failed"
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="failed",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            except Exception as exc:
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = "failed"
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="failed",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            else:
                cases.append(
                    make_split_sweep_case_result(
                        model_name="YOLO26n",
                        split_target=target,
                        boundary=boundary,
                        status="supported",
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                    )
                )
            finally:
                del cuda_runtime
                _clear_cuda_working_set(device)

        report = summarize_split_sweep_coverage("YOLO26n", cases)
        write_split_sweep_coverage_report(
            report,
            SPLIT_SWEEP_COVERAGE_DIR / "yolo26n_cuda_dynamic_batch_compute.json",
        )
        print(format_split_sweep_coverage_summary(report))
        assert_split_sweep_coverage(report, 0.5)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tinynext_backbone_split_replay_on_cuda() -> None:
    """TinyNeXt backbone split replay preserves tuple feature outputs."""

    if not TINYNEXT_SOURCE.exists():
        pytest.skip("TinyNeXt source file is not available.")
    module = _load_module_from_file("torchlens_split_tinynext", TINYNEXT_SOURCE)
    model = module.TinyNeXtBackbone(
        cfg=module.TINYNEXT_VARIANTS["tinynext_s"]["cfg"],
        pretrained_path=None,
    ).cuda().eval()
    x = torch.rand(2, 3, 320, 320, device="cuda")
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="percent:50", dynamic_batch=(1, 4)),
    )
    boundary = runtime.run_prefix(x)
    assert len(boundary.tensors) >= 1

    with torch.no_grad():
        split_output = runtime.run_suffix(boundary)
        full_output = model(x)
    assert _tree_allclose(split_output, full_output, atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tinynext_cpu_prefix_cuda_suffix_across_split_points() -> None:
    """TinyNeXt CPU prefix boundaries can drive CUDA suffixes at several split points."""

    if not TINYNEXT_SOURCE.exists():
        pytest.skip("TinyNeXt source file is not available.")
    module = _load_module_from_file("torchlens_split_tinynext_cross_device", TINYNEXT_SOURCE)
    cpu_model = module.TinyNeXtBackbone(
        cfg=module.TINYNEXT_VARIANTS["tinynext_s"]["cfg"],
        pretrained_path=None,
    ).eval()
    cuda_model = module.TinyNeXtBackbone(
        cfg=module.TINYNEXT_VARIANTS["tinynext_s"]["cfg"],
        pretrained_path=None,
    ).cuda().eval()
    cuda_model.load_state_dict(cpu_model.state_dict())

    x_cpu = torch.rand(2, 3, 320, 320)
    x_cuda = x_cpu.cuda()
    with _cuda_tf32_disabled():
        for boundary in ("percent:25", "percent:50", "percent:75"):
            spec = tl.SplitSpec(boundary=boundary, dynamic_batch=(1, 4))
            cpu_runtime = tl.prepare_split(cpu_model, x_cpu, spec)
            cuda_runtime = tl.prepare_split(cuda_model, x_cuda, spec)
            cpu_boundary = cpu_runtime.run_prefix(x_cpu)
            assert {tensor.device.type for tensor in cpu_boundary.tensors.values()} == {"cpu"}
            with torch.no_grad():
                split_output = cuda_runtime.run_suffix(cpu_boundary)
                full_output = cuda_model(x_cuda)
            _assert_cross_device_output_accepted(
                split_output,
                full_output,
                device="cuda",
                atol=1e-4,
                rtol=1e-4,
            )


@pytest.mark.slow
def test_tinynext_all_compute_split_points_on_cuda() -> None:
    """Sweep every TorchLens compute node split point in TinyNeXt."""

    if not TINYNEXT_SOURCE.exists():
        pytest.skip("TinyNeXt source file is not available.")
    module = _load_module_from_file("torchlens_split_tinynext_all", TINYNEXT_SOURCE)
    with _cuda_tf32_disabled():
        has_cuda = torch.cuda.is_available()
        cpu_model = module.TinyNeXtBackbone(
            cfg=module.TINYNEXT_VARIANTS["tinynext_s"]["cfg"],
            pretrained_path=None,
        ).eval().cpu()
        cuda_model = None
        if has_cuda:
            cuda_model = module.TinyNeXtBackbone(
                cfg=module.TINYNEXT_VARIANTS["tinynext_s"]["cfg"],
                pretrained_path=None,
            ).eval().cuda()
            cuda_model.load_state_dict(cpu_model.state_dict())
        runtime_cases = _tensor_runtime_cases(
            (3, 320, 320),
            batch_sizes=REAL_MODEL_DYNAMIC_BATCHES if has_cuda else REAL_MODEL_CPU_DYNAMIC_BATCHES,
        )
        trace_cpu = runtime_cases[0]
        cpu_graph = _trace_graph(cpu_model, trace_cpu)
        cuda_graph = _trace_graph(cuda_model, trace_cpu.cuda()) if cuda_model is not None else None
        targets = _compute_split_targets(cpu_graph)

        report = _run_real_model_compute_split_sweep(
            model_name="TinyNeXt",
            artifact_name=f"tinynext_{'cuda' if has_cuda else 'cpu'}_compute.json",
            cpu_model=cpu_model,
            cuda_model=cuda_model,
            cpu_graph=cpu_graph,
            cuda_graph=cuda_graph,
            targets=targets,
            runtime_cases=runtime_cases,
            min_support_ratio=0.5,
        )
    print(format_split_sweep_coverage_summary(report))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_resnet18_cpu_prefix_cuda_suffix_allclose_without_tf32() -> None:
    """ResNet18 CPU-prefix/CUDA-suffix replay is numerically close without TF32."""

    torchvision = pytest.importorskip("torchvision")
    with _cuda_tf32_disabled():
        torch.manual_seed(0)
        cpu_model = torchvision.models.resnet18(weights=None).eval().cpu()
        cuda_model = torchvision.models.resnet18(weights=None).eval().cuda()
        cuda_model.load_state_dict(cpu_model.state_dict())
        runtime_cases = _tensor_runtime_cases(
            (3, 224, 224),
            batch_sizes=REAL_MODEL_DYNAMIC_BATCHES,
        )
        trace_cpu = runtime_cases[0]
        cpu_graph = _trace_graph(cpu_model, trace_cpu)
        cuda_graph = _trace_graph(cuda_model, trace_cpu.cuda())
        targets = _compute_split_targets(cpu_graph)

        assert len(targets) >= 100
        for target in targets:
            boundary = f"after:{target}"
            cpu_runtime = None
            cuda_runtime = None
            cpu_boundary = None
            cuda_boundary = None
            full_output = None
            cuda_split_output = None
            cross_device_output = None
            try:
                spec = tl.SplitSpec(boundary=boundary, dynamic_batch=REAL_MODEL_DYNAMIC_BATCH_RANGE)
                cpu_runtime = _runtime_from_graph(cpu_model, cpu_graph, spec)
                cuda_runtime = _runtime_from_graph(cuda_model, cuda_graph, spec)
                assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)
                for x_cpu in runtime_cases:
                    x_cuda = x_cpu.cuda()
                    cpu_boundary = cpu_runtime.run_prefix(x_cpu)
                    cuda_boundary = cuda_runtime.run_prefix(x_cuda)
                    _assert_boundary_shapes(cpu_boundary, cuda_boundary)
                    with torch.no_grad():
                        full_output = cuda_model(x_cuda)
                        cuda_split_output = cuda_runtime.run_suffix(cuda_boundary)
                        cross_device_output = cuda_runtime.run_suffix(cpu_boundary)
                    assert _tree_allclose(cuda_split_output, full_output, atol=1e-4, rtol=1e-4)
                    _assert_cross_device_output_accepted(
                        cross_device_output,
                        full_output,
                        device="cuda",
                        atol=1e-4,
                        rtol=1e-4,
                    )
            finally:
                del cpu_runtime, cuda_runtime, cpu_boundary, cuda_boundary
                del full_output, cuda_split_output, cross_device_output
                torch.cuda.empty_cache()


@pytest.mark.smoke
def test_rfdetr_nano_split_replay_on_cuda_external_env() -> None:
    """RF-DETR Nano split replay smoke using the remote RF-DETR environment."""

    if os.name == "nt":
        pytest.skip("RF-DETR external CUDA smoke is configured for the remote Linux host.")
    if not RFDETR_PYTHON.exists():
        pytest.skip("RF-DETR external Python environment is not available.")
    weight_path = _rfdetr_weight_file()

    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import torch
        import torchlens as tl
        from rfdetr import RFDETRNano
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        if not torch.cuda.is_available():
            raise SystemExit('CUDA is required for this smoke.')

        wrapper = RFDETRNano(pretrain_weights={str(weight_path)!r}, num_classes=8)
        ctx = wrapper.get_model(wrapper.model_config)
        model = ctx.model.cuda().eval()
        x = torch.rand(1, 3, ctx.resolution, ctx.resolution, device='cuda')
        nested = nested_tensor_from_tensor_list([x[0]])
        runtime = tl.prepare_split(
            model,
            nested,
            tl.SplitSpec(boundary='percent:50', dynamic_batch=(1, 2)),
        )
        boundary = runtime.run_prefix(nested)
        assert len(boundary.tensors) >= 1
        with torch.no_grad():
            split_output = runtime.run_suffix(boundary)
            full_output = model(nested)
        for key in ('pred_logits', 'pred_boxes'):
            assert torch.allclose(split_output[key], full_output[key], atol=1e-4, rtol=1e-4)
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [str(RFDETR_PYTHON), "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
def test_rfdetr_nano_cuda_training_trajectory_external_env() -> None:
    """RF-DETR Nano CUDA training keeps trace and split close to eager."""

    if os.name == "nt":
        pytest.skip("RF-DETR external CUDA training is configured for the remote Linux host.")
    if not RFDETR_PYTHON.exists():
        pytest.skip("RF-DETR external Python environment is not available.")
    weight_path = _rfdetr_weight_file()

    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        import gc
        import random

        import numpy as np
        import torch
        import torchlens as tl
        from torchlens.options import CaptureOptions, VisualizationOptions
        from rfdetr import RFDETRNano
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        if not torch.cuda.is_available():
            raise SystemExit('CUDA is required for this training smoke.')

        WEIGHTS = Path({str(weight_path)!r})
        SEED = 20240606
        STEPS = 2
        LR = 1e-5
        BOUNDARY = 'after:permute_69_402'
        LOSS_RTOL = 5e-3
        LOSS_ATOL = 1e-4
        STATE_RTOL = 5e-4
        STATE_ATOL = 1e-4


        def seed_all():
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)


        def make_context_model():
            wrapper = RFDETRNano(pretrain_weights=str(WEIGHTS), num_classes=8)
            ctx = wrapper.get_model(wrapper.model_config)
            model = ctx.model.cuda().eval()
            return ctx, model


        def clone_state_dict_cpu(model):
            return {{name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}


        def load_fresh_model(base_state):
            ctx, model = make_context_model()
            missing, unexpected = model.load_state_dict(base_state, strict=True)
            assert not missing and not unexpected
            model.cuda().eval()
            return ctx, model


        def make_input(resolution):
            generator = torch.Generator(device='cpu')
            generator.manual_seed(SEED + 17)
            x = torch.rand(1, 3, resolution, resolution, generator=generator).cuda()
            return nested_tensor_from_tensor_list([x[0]])


        def output_loss(out):
            return out['pred_logits'].float().square().mean() + out['pred_boxes'].float().square().mean()


        def trace_loss(model_log):
            logits = model_log[model_log.output_layers[0]].activation
            boxes = model_log[model_log.output_layers[1]].activation
            assert tuple(logits.shape[-2:]) == (300, 9)
            assert tuple(boxes.shape[-2:]) == (300, 4)
            return logits.float().square().mean() + boxes.float().square().mean()


        def train_eager(base_state, resolution):
            _, model = load_fresh_model(base_state)
            nested = make_input(resolution)
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            losses = []
            for _step in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                loss = output_loss(model(nested))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            state = clone_state_dict_cpu(model)
            del model, nested, optimizer, loss
            torch.cuda.empty_cache()
            gc.collect()
            return losses, state


        def train_trace(base_state, resolution):
            _, model = load_fresh_model(base_state)
            nested = make_input(resolution)
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            losses = []
            capture = CaptureOptions(train_mode=True)
            visualization = VisualizationOptions(view='none')
            for _step in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                model_log = tl.log_forward_pass(
                    model,
                    nested,
                    capture=capture,
                    visualization=visualization,
                )
                loss = trace_loss(model_log)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                model_log.cleanup()
                del model_log, loss
                torch.cuda.empty_cache()
            state = clone_state_dict_cpu(model)
            del model, nested, optimizer
            torch.cuda.empty_cache()
            gc.collect()
            return losses, state


        def train_split(base_state, resolution):
            _, model = load_fresh_model(base_state)
            nested = make_input(resolution)
            runtime = tl.prepare_split(
                model,
                nested,
                tl.SplitSpec(boundary=BOUNDARY, dynamic_batch=(1, 1), trainable=True),
            )
            prefix_params = [param for name, param in model.named_parameters() if name.startswith('backbone.')]
            suffix_params = [param for name, param in model.named_parameters() if not name.startswith('backbone.')]
            prefix_optimizer = torch.optim.SGD(prefix_params, lr=LR)
            suffix_optimizer = torch.optim.SGD(suffix_params, lr=LR)
            losses = []
            for _step in range(STEPS):
                prefix_optimizer.zero_grad(set_to_none=True)
                suffix_optimizer.zero_grad(set_to_none=True)
                boundary = runtime.run_training_prefix(nested)
                loss, boundary_grads = runtime.train_suffix(
                    boundary,
                    targets=None,
                    loss_fn=lambda out, _targets: output_loss(out),
                    optimizer=suffix_optimizer,
                )
                runtime.backward_prefix(boundary, boundary_grads, prefix_optimizer)
                losses.append(float(loss.detach().cpu()))
                del boundary, boundary_grads, loss
                torch.cuda.empty_cache()
            state = clone_state_dict_cpu(model)
            del model, nested, runtime, prefix_optimizer, suffix_optimizer
            torch.cuda.empty_cache()
            gc.collect()
            return losses, state


        def assert_loss_allclose(name, actual, expected):
            actual_tensor = torch.tensor(actual)
            expected_tensor = torch.tensor(expected)
            assert torch.allclose(
                actual_tensor,
                expected_tensor,
                rtol=LOSS_RTOL,
                atol=LOSS_ATOL,
            ), f"{{name}} losses differ: actual={{actual}} expected={{expected}}"


        def assert_state_allclose(name, actual, expected):
            bad = []
            for key, expected_value in expected.items():
                actual_value = actual[key]
                if torch.is_floating_point(expected_value) or torch.is_complex(expected_value):
                    close = torch.allclose(
                        actual_value,
                        expected_value,
                        rtol=STATE_RTOL,
                        atol=STATE_ATOL,
                    )
                else:
                    close = torch.equal(actual_value, expected_value)
                if not close:
                    diff = (
                        float((actual_value - expected_value).abs().max())
                        if torch.is_tensor(actual_value) and torch.is_floating_point(actual_value)
                        else float('inf')
                    )
                    bad.append((key, diff))
            assert not bad, f"{{name}} state differs: {{bad[:10]}}"


        seed_all()
        base_ctx, base_model = make_context_model()
        resolution = int(base_ctx.resolution)
        base_state = clone_state_dict_cpu(base_model)
        del base_model
        torch.cuda.empty_cache()

        eager_losses, eager_state = train_eager(base_state, resolution)
        trace_losses, trace_state = train_trace(base_state, resolution)
        split_losses, split_state = train_split(base_state, resolution)

        assert_loss_allclose('trace', trace_losses, eager_losses)
        assert_loss_allclose('split', split_losses, eager_losses)
        assert_state_allclose('trace', trace_state, eager_state)
        assert_state_allclose('split', split_state, eager_state)
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [str(RFDETR_PYTHON), "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
def test_rfdetr_nano_cpu_prefix_cuda_suffix_external_env() -> None:
    """RF-DETR Nano CPU prefix boundaries can execute through CUDA suffixes.

    RF-DETR's full CPU and CUDA outputs are not numerically close on this
    environment, so this smoke validates ABI equivalence, boundary transfer,
    CUDA suffix execution, and output structure across every TorchLens compute node.
    CUDA-only RF-DETR replay equivalence is covered separately above.
    """

    if os.name == "nt":
        pytest.skip("RF-DETR external CUDA smoke is configured for the remote Linux host.")
    if not RFDETR_PYTHON.exists():
        pytest.skip("RF-DETR external Python environment is not available.")
    weight_path = _rfdetr_weight_file()

    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import gc
        import copy
        import torch
        import torchlens as tl
        from torchlens.split.codegen import build_segments
        from torchlens.split.errors import SplitUnsupportedError
        from torchlens.split.planner import plan_split
        from torchlens.split.runtime import SplitRuntime
        from torchlens.split.trace_graph import trace_graph_from_model_log
        from torchlens.split.validation import assert_canonical_abi_equivalent
        from torchlens.split.validation import assert_split_sweep_coverage
        from torchlens.split.validation import assert_split_unsupported_error_context
        from torchlens.split.validation import format_split_sweep_coverage_summary
        from torchlens.split.validation import make_split_sweep_case_result
        from torchlens.split.validation import summarize_split_sweep_coverage
        from torchlens.split.validation import write_split_sweep_coverage_report
        from rfdetr import RFDETRNano
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        if not torch.cuda.is_available():
            raise SystemExit('CUDA is required for this smoke.')

        wrapper = RFDETRNano(pretrain_weights={str(weight_path)!r}, num_classes=8)
        ctx = wrapper.get_model(wrapper.model_config)
        cpu_model = ctx.model.cpu().eval()
        cuda_model = copy.deepcopy(cpu_model).cuda().eval()
        runtime_tensors = [torch.rand(1, 3, ctx.resolution, ctx.resolution)]
        trace_tensor = runtime_tensors[0]
        trace_cpu = nested_tensor_from_tensor_list([trace_tensor[index] for index in range(trace_tensor.shape[0])])
        trace_cuda = nested_tensor_from_tensor_list(
            [trace_tensor[index].cuda() for index in range(trace_tensor.shape[0])]
        )

        def trace_graph(model, inputs):
            model_log = tl.log_forward_pass(
                model,
                inputs,
                vis_opt='none',
                layers_to_save='all',
                keep_unsaved_layers=True,
                detach_saved_tensors=False,
                save_function_args=True,
                intervention_ready=True,
            )
            return trace_graph_from_model_log(
                model_log,
                traced_batch_size=1,
                batch_symbol='B',
                dynamic_batch={REAL_MODEL_DYNAMIC_BATCH_RANGE!r},
            )

        def runtime_from_graph(model, graph, spec):
            plan = plan_split(graph, spec)
            segments = build_segments(graph, plan, mode=spec.mode)
            return SplitRuntime(
                model=model,
                trace_graph=graph,
                split_spec=spec,
                plan=plan,
                segments=segments,
            )

        def compute_split_targets(graph):
            targets = []
            for node in graph.ordered_nodes():
                if node.is_input or node.is_output:
                    continue
                if node.torchlens_label:
                    targets.append(node.torchlens_label)
            return targets

        def assert_boundary_shapes(left, right):
            assert set(left.tensors) == set(right.tensors)
            for tensor_id in left.tensors:
                assert left.tensors[tensor_id].shape == right.tensors[tensor_id].shape
                assert left.tensors[tensor_id].dtype == right.tensors[tensor_id].dtype

        def assert_cross_device_output_accepted(output, reference):
            for key in ('pred_logits', 'pred_boxes'):
                assert output[key].device.type == 'cuda'
                assert output[key].shape == reference[key].shape
                assert torch.isfinite(output[key]).all()

        cpu_graph = trace_graph(cpu_model, trace_cpu)
        cuda_graph = trace_graph(cuda_model, trace_cuda)
        targets = compute_split_targets(cpu_graph)
        assert len(targets) >= 100

        cases = []
        for target in targets:
            boundary = 'after:' + target
            batch_statuses = {{}}
            tested_batches = []
            cpu_runtime = None
            cuda_runtime = None
            payload = None
            cuda_payload = None
            full_output = None
            cuda_split_output = None
            cross_device_output = None
            try:
                spec = tl.SplitSpec(boundary=boundary, dynamic_batch={REAL_MODEL_DYNAMIC_BATCH_RANGE!r})
                cpu_runtime = runtime_from_graph(cpu_model, cpu_graph, spec)
                cuda_runtime = runtime_from_graph(cuda_model, cuda_graph, spec)
                assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)
                for tensor in runtime_tensors:
                    batch_size = int(tensor.shape[0])
                    tested_batches.append(batch_size)
                    nested_cpu = nested_tensor_from_tensor_list(
                        [tensor[index] for index in range(tensor.shape[0])]
                    )
                    nested_cuda = nested_tensor_from_tensor_list(
                        [tensor[index].cuda() for index in range(tensor.shape[0])]
                    )
                    payload = cpu_runtime.run_prefix(nested_cpu)
                    cuda_payload = cuda_runtime.run_prefix(nested_cuda)
                    assert set(tensor.device.type for tensor in payload.tensors.values()) == set(['cpu'])
                    assert set(tensor.device.type for tensor in cuda_payload.tensors.values()) <= set(['cpu', 'cuda'])
                    assert_boundary_shapes(payload, cuda_payload)
                    with torch.no_grad():
                        full_output = cuda_model(nested_cuda)
                        cuda_split_output = cuda_runtime.run_suffix(cuda_payload)
                        cross_device_output = cuda_runtime.run_suffix(payload)
                    for key in ('pred_logits', 'pred_boxes'):
                        assert torch.allclose(
                            cuda_split_output[key],
                            full_output[key],
                            atol=1e-4,
                            rtol=1e-4,
                        )
                    assert_cross_device_output_accepted(cross_device_output, full_output)
                    batch_statuses[str(batch_size)] = 'supported'
            except SplitUnsupportedError as exc:
                assert_split_unsupported_error_context(exc, boundary)
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = 'unsupported'
                cases.append(
                    make_split_sweep_case_result(
                        model_name='RF-DETR',
                        split_target=target,
                        boundary=boundary,
                        status='unsupported',
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            except Exception as exc:
                if tested_batches:
                    batch_statuses[str(tested_batches[-1])] = 'failed'
                cases.append(
                    make_split_sweep_case_result(
                        model_name='RF-DETR',
                        split_target=target,
                        boundary=boundary,
                        status='failed',
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                        exc=exc,
                    )
                )
            else:
                cases.append(
                    make_split_sweep_case_result(
                        model_name='RF-DETR',
                        split_target=target,
                        boundary=boundary,
                        status='supported',
                        batch_sizes_tested=tuple(tested_batches),
                        batch_statuses=batch_statuses,
                    )
                )
            finally:
                del cpu_runtime, cuda_runtime, payload, cuda_payload
                del full_output, cuda_split_output, cross_device_output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        report = summarize_split_sweep_coverage('RF-DETR', cases)
        write_split_sweep_coverage_report(
            report,
            {str(SPLIT_SWEEP_COVERAGE_DIR / "rfdetr_cuda_compute.json")!r},
        )
        print(format_split_sweep_coverage_summary(report))
        assert_split_sweep_coverage(report, 0.3)
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [str(RFDETR_PYTHON), "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _load_module_from_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _download_weight_file(*, url: str, destination: Path, model_name: str) -> Path:
    """Download real-model weights into the test artifact cache when missing."""

    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.tmp")
    try:
        urllib.request.urlretrieve(url, temporary_path)
    except (OSError, urllib.error.URLError) as exc:
        temporary_path.unlink(missing_ok=True)
        pytest.skip(f"{model_name} weights could not be downloaded from {url}: {exc}")
    temporary_path.replace(destination)
    return destination


def _rfdetr_weight_file() -> Path:
    """Return the downloaded RF-DETR Nano weight file."""

    return _download_weight_file(
        url=RFDETR_NANO_WEIGHTS_URL,
        destination=RFDETR_WEIGHTS,
        model_name="RF-DETR Nano",
    )


def _tree_allclose(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        return all(
            _tree_allclose(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_tree_allclose(left[key], right[key], atol=atol, rtol=rtol) for key in left)
    return left == right


def _assert_cross_device_output_accepted(
    output: Any,
    reference: Any,
    *,
    device: str,
    atol: float,
    rtol: float,
    require_numeric: bool = True,
) -> None:
    """Validate cross-device replay without overfitting to detection postprocessing order."""

    _assert_model_output_accepted(
        output,
        reference,
        device=device,
        atol=atol,
        rtol=rtol,
        require_numeric=require_numeric,
    )


def _assert_model_output_accepted(
    output: Any,
    reference: Any,
    *,
    device: str,
    atol: float,
    rtol: float,
    require_numeric: bool = True,
) -> None:
    """Validate replay output without overfitting to unstable detector postprocessing."""

    _assert_tree_shape_device_finite(output, reference, device)
    output_numeric = _stable_numeric_output(output)
    reference_numeric = _stable_numeric_output(reference)
    if output_numeric is not None and reference_numeric is not None:
        assert _tree_allclose(output_numeric, reference_numeric, atol=atol, rtol=rtol)
        return
    if require_numeric:
        assert _tree_allclose(output, reference, atol=atol, rtol=rtol)


def _stable_numeric_output(output: Any) -> Any | None:
    """Extract the continuous part of model output that is stable across CPU/CUDA backends.

    Classification models usually return tensor trees, so the full output is the
    stable target. YOLO-style detectors return ``(postprocessed, raw_preds)`` in
    eval mode; the postprocessed top-k rows are intentionally excluded because
    tiny CPU/CUDA score differences can legitimately change discrete candidate
    ordering while raw boxes/scores remain close.
    """

    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        return _stable_numeric_output(output[1])
    if isinstance(output, dict):
        selected: dict[str, Any] = {}
        for key in ("pred_logits", "pred_boxes", "logits", "boxes", "scores"):
            if key in output and _is_tensor_tree(output[key]):
                selected[key] = output[key]
        for key in ("one2one", "one2many"):
            child = output.get(key)
            if isinstance(child, dict):
                child_selected = {
                    child_key: child[child_key]
                    for child_key in ("pred_logits", "pred_boxes", "logits", "boxes", "scores")
                    if child_key in child and _is_tensor_tree(child[child_key])
                }
                if child_selected:
                    selected[key] = child_selected
        if selected:
            return selected
    if _is_tensor_tree(output):
        return output
    return None


def _is_tensor_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_tensor_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_is_tensor_tree(item) for item in value.values())
    return False


def _assert_tree_shape_device_finite(left: Any, right: Any, device: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert left.device.type == device
        assert left.shape == right.shape
        assert torch.isfinite(left).all()
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        for a, b in zip(left, right, strict=True):
            _assert_tree_shape_device_finite(a, b, device)
        return
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        for key in left:
            _assert_tree_shape_device_finite(left[key], right[key], device)
        return
    assert left == right


def _trace_graph(model: torch.nn.Module, inputs: Any) -> Any:
    model_log = tl.log_forward_pass(
        model,
        inputs,
        vis_opt="none",
        layers_to_save="all",
        keep_unsaved_layers=True,
        detach_saved_tensors=False,
        save_function_args=True,
        intervention_ready=True,
    )
    return trace_graph_from_model_log(
        model_log,
        traced_batch_size=int(inputs.shape[0]) if isinstance(inputs, torch.Tensor) else 1,
        batch_symbol="B",
        dynamic_batch=REAL_MODEL_DYNAMIC_BATCH_RANGE,
    )


def _tensor_runtime_cases(
    sample_shape: tuple[int, ...],
    *,
    batch_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn((batch_size, *sample_shape)) for batch_size in batch_sizes)


def _runtime_from_graph(
    model: torch.nn.Module,
    graph: Any,
    spec: tl.SplitSpec,
) -> SplitRuntime:
    plan = plan_split(graph, spec)
    segments = build_segments(graph, plan, mode=spec.mode)
    return SplitRuntime(
        model=model,
        trace_graph=graph,
        split_spec=spec,
        plan=plan,
        segments=segments,
    )


def _run_real_model_compute_split_sweep(
    *,
    model_name: str,
    artifact_name: str,
    cpu_model: torch.nn.Module,
    cuda_model: torch.nn.Module | None,
    cpu_graph: Any,
    cuda_graph: Any | None,
    targets: list[str],
    runtime_cases: tuple[torch.Tensor, ...],
    min_support_ratio: float,
) -> Any:
    assert len(targets) >= 100
    cases: list[SplitSweepCaseResult] = []
    for target in targets:
        boundary = f"after:{target}"
        batch_statuses: dict[str, Any] = {}
        tested_batches: list[int] = []
        try:
            spec = tl.SplitSpec(boundary=boundary, dynamic_batch=REAL_MODEL_DYNAMIC_BATCH_RANGE)
            cpu_runtime = _runtime_from_graph(cpu_model, cpu_graph, spec)
            cuda_runtime = None
            if cuda_model is not None and cuda_graph is not None:
                cuda_runtime = _runtime_from_graph(cuda_model, cuda_graph, spec)
                assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)

            for x_cpu in runtime_cases:
                batch_size = int(x_cpu.shape[0])
                tested_batches.append(batch_size)
                cpu_boundary = cpu_runtime.run_prefix(x_cpu)
                _assert_boundary_devices(cpu_boundary, "cpu")

                with torch.no_grad():
                    if cuda_model is not None and cuda_runtime is not None:
                        x_cuda = x_cpu.cuda()
                        cuda_boundary = cuda_runtime.run_prefix(x_cuda)
                        _assert_boundary_devices(cuda_boundary, "cuda")
                        _assert_boundary_shapes(cpu_boundary, cuda_boundary)
                        full_output = cuda_model(x_cuda)
                        split_output = cuda_runtime.run_suffix(cuda_boundary)
                        cross_device_output = cuda_runtime.run_suffix(cpu_boundary)
                        assert _tree_allclose(split_output, full_output, atol=1e-4, rtol=1e-4)
                        _assert_cross_device_output_accepted(
                            cross_device_output,
                            full_output,
                            device="cuda",
                            atol=1e-4,
                            rtol=1e-4,
                        )
                    else:
                        full_output = cpu_model(x_cpu)
                        split_output = cpu_runtime.run_suffix(cpu_boundary)
                        assert _tree_allclose(split_output, full_output, atol=1e-4, rtol=1e-4)
                batch_statuses[str(batch_size)] = "supported"
        except SplitUnsupportedError as exc:
            assert_split_unsupported_error_context(exc, boundary)
            if tested_batches:
                batch_statuses[str(tested_batches[-1])] = "unsupported"
            cases.append(
                make_split_sweep_case_result(
                    model_name=model_name,
                    split_target=target,
                    boundary=boundary,
                    status="unsupported",
                    batch_sizes_tested=tuple(tested_batches),
                    batch_statuses=batch_statuses,
                    exc=exc,
                )
            )
        except Exception as exc:
            if tested_batches:
                batch_statuses[str(tested_batches[-1])] = "failed"
            cases.append(
                make_split_sweep_case_result(
                    model_name=model_name,
                    split_target=target,
                    boundary=boundary,
                    status="failed",
                    batch_sizes_tested=tuple(tested_batches),
                    batch_statuses=batch_statuses,
                    exc=exc,
                )
            )
        else:
            cases.append(
                make_split_sweep_case_result(
                    model_name=model_name,
                    split_target=target,
                    boundary=boundary,
                    status="supported",
                    batch_sizes_tested=tuple(tested_batches),
                    batch_statuses=batch_statuses,
                )
            )

    report = summarize_split_sweep_coverage(model_name, cases)
    write_split_sweep_coverage_report(report, SPLIT_SWEEP_COVERAGE_DIR / artifact_name)
    assert_split_sweep_coverage(report, min_support_ratio)
    return report


def _compute_split_targets(graph: Any) -> list[str]:
    targets: list[str] = []
    for node in graph.ordered_nodes():
        if node.is_input or node.is_output:
            continue
        if node.torchlens_label:
            targets.append(node.torchlens_label)
    return targets


def _assert_boundary_devices(boundary: Any, expected: str) -> None:
    devices = {tensor.device.type for tensor in boundary.tensors.values()}
    if expected == "cpu":
        assert devices == {"cpu"}
        return
    assert devices <= {"cpu", expected}


def _assert_boundary_shapes(left: Any, right: Any) -> None:
    assert set(left.tensors) == set(right.tensors)
    for tensor_id in left.tensors:
        assert left.tensors[tensor_id].shape == right.tensors[tensor_id].shape
        assert left.tensors[tensor_id].dtype == right.tensors[tensor_id].dtype


def _assert_cuda_split_replay_matches(
    model: torch.nn.Module,
    runtime: SplitRuntime,
    *,
    batch_size: int,
    sample_shape: tuple[int, ...],
    device: torch.device,
    atol: float,
    rtol: float,
) -> None:
    """Run one CUDA split replay case in a tight scope so tensors can be freed."""

    x_cuda = torch.randn((batch_size, *sample_shape), device=device)
    with torch.inference_mode():
        boundary = runtime.run_prefix(x_cuda)
        full_output = model(x_cuda)
        split_output = runtime.run_suffix(boundary)
    _assert_model_output_accepted(
        split_output,
        full_output,
        device=device.type,
        atol=atol,
        rtol=rtol,
    )


def _select_cuda_device_with_free_memory(*, min_free_gib: float) -> torch.device:
    """Select the CUDA device with the most free memory, skipping if too small."""

    candidates = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        candidates.append((free_bytes, total_bytes, index))
    free_bytes, total_bytes, index = max(candidates)
    min_free_bytes = _gib_to_bytes(min_free_gib)
    if free_bytes < min_free_bytes:
        pytest.skip(
            "YOLO26n dynamic batch sweep needs at least "
            f"{min_free_gib:.1f} GiB free CUDA memory; best device cuda:{index} has "
            f"{_bytes_to_gib(free_bytes):.1f}/{_bytes_to_gib(total_bytes):.1f} GiB free."
        )
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def _skip_if_cuda_memory_below(
    device: torch.device,
    *,
    min_free_gib: float,
    context: str,
) -> None:
    """Skip the heavy real-model sweep before a case can pressure CUDA memory."""

    _clear_cuda_working_set(device)
    with torch.cuda.device(device):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    min_free_bytes = _gib_to_bytes(min_free_gib)
    if free_bytes < min_free_bytes:
        pytest.skip(
            "Stopping YOLO26n dynamic batch sweep before "
            f"{context}: free CUDA memory is {_bytes_to_gib(free_bytes):.1f}/"
            f"{_bytes_to_gib(total_bytes):.1f} GiB, below the {min_free_gib:.1f} GiB "
            "safety floor."
        )


def _skip_if_system_memory_below(min_available_gib: float) -> None:
    """Skip memory-heavy real-model sweeps when host RAM is already tight."""

    available_bytes = _available_system_memory_bytes()
    if available_bytes is None:
        return
    min_available_bytes = _gib_to_bytes(min_available_gib)
    if available_bytes < min_available_bytes:
        pytest.skip(
            "YOLO26n dynamic batch sweep needs at least "
            f"{min_available_gib:.1f} GiB available system memory; host has "
            f"{_bytes_to_gib(available_bytes):.1f} GiB available."
        )


def _available_system_memory_bytes() -> int | None:
    """Return available host memory from Linux procfs when available."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _clear_cuda_working_set(device: torch.device | None = None) -> None:
    """Release Python and CUDA caches between memory-heavy real-model cases."""

    gc.collect()
    if not torch.cuda.is_available():
        return
    if device is not None:
        torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        return


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return whether an exception is a CUDA out-of-memory failure."""

    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    message = str(exc).lower()
    return isinstance(exc, cuda_oom_type) or ("cuda" in message and "out of memory" in message)


def _gib_to_bytes(value: float) -> int:
    return int(value * 1024**3)


def _bytes_to_gib(value: int) -> float:
    return value / 1024**3


@contextmanager
def _cuda_tf32_disabled() -> Iterator[None]:
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
