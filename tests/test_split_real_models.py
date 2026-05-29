"""Optional real-model smoke tests for native split replay."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import textwrap
from typing import Any

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


YOLO26N_CANDIDATES = (
    Path("/home/whisperliang/TSBOW-main/yolo26n.pt"),
    Path("/home/whisperliang/TSBOW-main/baselines/yolo26n.pt"),
    Path("/home/whisperliang/Plank-road/model_management/models/yolo26n.pt"),
)
PLANK_ROAD_ROOT = Path("/home/whisperliang/Plank-road")
TINYNEXT_SOURCE = PLANK_ROAD_ROOT / "model_management" / "tinynext.py"
RFDETR_PYTHON = PLANK_ROAD_ROOT / ".venv" / "bin" / "python"
RFDETR_WEIGHTS = Path("/home/whisperliang/TSBOW-main/baselines/rf-detr-nano.pth")
REAL_MODEL_DYNAMIC_BATCHES = (1, 2, 4, 8, 32)
REAL_MODEL_DYNAMIC_BATCH_RANGE = (1, 32)
SPLIT_SWEEP_COVERAGE_DIR = Path(__file__).resolve().parent / "artifacts" / "split_sweep_coverage"


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yolo26n_detection_head_boundary_is_live_on_cuda() -> None:
    """YOLO26n split after detection head preserves all live boundary tensors."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = next((path for path in YOLO26N_CANDIDATES if path.exists()), None)
    if weight_path is None:
        pytest.skip("YOLO26n weights are not available.")

    model = ultralytics.YOLO(str(weight_path)).model.eval().cuda()
    x = torch.randn(2, 3, 640, 640, device="cuda")
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:model.23", dynamic_batch=(1, 4)),
    )
    boundary = runtime.run_prefix(x)

    assert len(boundary.tensors) >= 5
    assert any("detach" in tensor_id for tensor_id in boundary.tensors)
    assert any(spec.role == "skip" for spec in boundary.spec.values())

    with torch.no_grad():
        split_output = runtime.run_suffix(boundary)
        full_output = model(x)
    assert _tree_allclose(split_output, full_output, atol=1e-4, rtol=1e-4)

    live = boundary_liveness_zero_probe(runtime, boundary, atol=1e-5, rtol=1e-5)
    assert all(live.values())


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yolo26n_all_semantic_split_points_on_cuda() -> None:
    """Sweep every semantic split point in YOLO26n across CPU and CUDA traces."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = next((path for path in YOLO26N_CANDIDATES if path.exists()), None)
    if weight_path is None:
        pytest.skip("YOLO26n weights are not available.")

    cpu_model = ultralytics.YOLO(str(weight_path)).model.eval().cpu()
    cuda_model = ultralytics.YOLO(str(weight_path)).model.eval().cuda()
    runtime_cases = _tensor_runtime_cases((3, 640, 640), batch_sizes=REAL_MODEL_DYNAMIC_BATCHES)
    trace_cpu = runtime_cases[0]
    trace_cuda = trace_cpu.cuda()
    cpu_graph = _trace_graph(cpu_model, trace_cpu)
    cuda_graph = _trace_graph(cuda_model, trace_cuda)
    targets = _semantic_split_targets(cpu_graph)

    report = _run_real_model_semantic_split_sweep(
        model_name="YOLO26n",
        artifact_name="yolo26n_cuda.json",
        cpu_model=cpu_model,
        cuda_model=cuda_model,
        cpu_graph=cpu_graph,
        cuda_graph=cuda_graph,
        targets=targets,
        runtime_cases=runtime_cases,
        min_support_ratio=0.5,
    )
    print(format_split_sweep_coverage_summary(report))


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
    for boundary in ("percent:25", "percent:50", "percent:75"):
        spec = tl.SplitSpec(boundary=boundary, dynamic_batch=(1, 4))
        cpu_runtime = tl.prepare_split(cpu_model, x_cpu, spec)
        cuda_runtime = tl.prepare_split(cuda_model, x_cuda, spec)
        cpu_boundary = cpu_runtime.run_prefix(x_cpu)
        assert {tensor.device.type for tensor in cpu_boundary.tensors.values()} == {"cpu"}
        with torch.no_grad():
            split_output = cuda_runtime.run_suffix(cpu_boundary)
            full_output = cuda_model(x_cuda)
    assert _tree_allclose(split_output, full_output, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tinynext_all_semantic_split_points_on_cuda() -> None:
    """Sweep every semantic split point in TinyNeXt across CPU and CUDA traces."""

    if not TINYNEXT_SOURCE.exists():
        pytest.skip("TinyNeXt source file is not available.")
    module = _load_module_from_file("torchlens_split_tinynext_all", TINYNEXT_SOURCE)
    cpu_model = module.TinyNeXtBackbone(
        cfg=module.TINYNEXT_VARIANTS["tinynext_s"]["cfg"],
        pretrained_path=None,
    ).eval().cpu()
    cuda_model = module.TinyNeXtBackbone(
        cfg=module.TINYNEXT_VARIANTS["tinynext_s"]["cfg"],
        pretrained_path=None,
    ).eval().cuda()
    cuda_model.load_state_dict(cpu_model.state_dict())
    runtime_cases = _tensor_runtime_cases((3, 320, 320), batch_sizes=REAL_MODEL_DYNAMIC_BATCHES)
    trace_cpu = runtime_cases[0]
    trace_cuda = trace_cpu.cuda()
    cpu_graph = _trace_graph(cpu_model, trace_cpu)
    cuda_graph = _trace_graph(cuda_model, trace_cuda)
    targets = _semantic_split_targets(cpu_graph)

    report = _run_real_model_semantic_split_sweep(
        model_name="TinyNeXt",
        artifact_name="tinynext_cuda.json",
        cpu_model=cpu_model,
        cuda_model=cuda_model,
        cpu_graph=cpu_graph,
        cuda_graph=cuda_graph,
        targets=targets,
        runtime_cases=runtime_cases,
        min_support_ratio=0.5,
    )
    print(format_split_sweep_coverage_summary(report))


@pytest.mark.smoke
def test_rfdetr_nano_split_replay_on_cuda_external_env() -> None:
    """RF-DETR Nano split replay smoke using the remote RF-DETR environment."""

    if os.name == "nt":
        pytest.skip("RF-DETR external CUDA smoke is configured for the remote Linux host.")
    if not RFDETR_PYTHON.exists():
        pytest.skip("RF-DETR external Python environment is not available.")
    if not RFDETR_WEIGHTS.exists():
        pytest.skip("RF-DETR Nano weights are not available.")

    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import torch
        import torchlens as tl
        from rfdetr import RFDETRNano
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        if not torch.cuda.is_available():
            raise SystemExit('CUDA is required for this smoke.')

        wrapper = RFDETRNano(pretrain_weights={str(RFDETR_WEIGHTS)!r}, num_classes=8)
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
def test_rfdetr_nano_cpu_prefix_cuda_suffix_external_env() -> None:
    """RF-DETR Nano CPU prefix boundaries can execute through CUDA suffixes.

    RF-DETR's full CPU and CUDA outputs are not numerically close on this
    environment, so this smoke validates ABI equivalence, boundary transfer,
    CUDA suffix execution, and output structure across every semantic split point.
    CUDA-only RF-DETR replay equivalence is covered separately above.
    """

    if os.name == "nt":
        pytest.skip("RF-DETR external CUDA smoke is configured for the remote Linux host.")
    if not RFDETR_PYTHON.exists():
        pytest.skip("RF-DETR external Python environment is not available.")
    if not RFDETR_WEIGHTS.exists():
        pytest.skip("RF-DETR Nano weights are not available.")

    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
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

        wrapper = RFDETRNano(pretrain_weights={str(RFDETR_WEIGHTS)!r}, num_classes=8)
        ctx = wrapper.get_model(wrapper.model_config)
        cpu_model = ctx.model.cpu().eval()
        cuda_model = copy.deepcopy(cpu_model).cuda().eval()
        runtime_tensors = [
            torch.rand(batch_size, 3, ctx.resolution, ctx.resolution)
            for batch_size in {REAL_MODEL_DYNAMIC_BATCHES!r}
        ]
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

        def semantic_split_targets(graph):
            targets = []
            seen = set()
            for node in graph.ordered_nodes():
                if node.is_input or node.is_output:
                    continue
                if node.module_path and node.module_path not in seen:
                    seen.add(node.module_path)
                    targets.append(node.module_path)
            return targets

        def assert_boundary_shapes(left, right):
            assert set(left.tensors) == set(right.tensors)
            for tensor_id in left.tensors:
                assert left.tensors[tensor_id].shape == right.tensors[tensor_id].shape
                assert left.tensors[tensor_id].dtype == right.tensors[tensor_id].dtype

        cpu_graph = trace_graph(cpu_model, trace_cpu)
        cuda_graph = trace_graph(cuda_model, trace_cuda)
        targets = semantic_split_targets(cpu_graph)
        assert len(targets) >= 100

        cases = []
        for target in targets:
            boundary = 'after:' + target
            batch_statuses = {{}}
            tested_batches = []
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
                        assert cross_device_output[key].device.type == 'cuda'
                        assert cross_device_output[key].shape == full_output[key].shape
                        assert torch.isfinite(cross_device_output[key]).all()
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

        report = summarize_split_sweep_coverage('RF-DETR', cases)
        write_split_sweep_coverage_report(
            report,
            {str(SPLIT_SWEEP_COVERAGE_DIR / "rfdetr_cuda.json")!r},
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


def _run_real_model_semantic_split_sweep(
    *,
    model_name: str,
    artifact_name: str,
    cpu_model: torch.nn.Module,
    cuda_model: torch.nn.Module,
    cpu_graph: Any,
    cuda_graph: Any,
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
            cuda_runtime = _runtime_from_graph(cuda_model, cuda_graph, spec)
            assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)

            for x_cpu in runtime_cases:
                batch_size = int(x_cpu.shape[0])
                tested_batches.append(batch_size)
                x_cuda = x_cpu.cuda()
                cpu_boundary = cpu_runtime.run_prefix(x_cpu)
                cuda_boundary = cuda_runtime.run_prefix(x_cuda)
                _assert_boundary_devices(cpu_boundary, "cpu")
                _assert_boundary_devices(cuda_boundary, "cuda")
                _assert_boundary_shapes(cpu_boundary, cuda_boundary)

                with torch.no_grad():
                    full_output = cuda_model(x_cuda)
                    cuda_split_output = cuda_runtime.run_suffix(cuda_boundary)
                    cross_device_output = cuda_runtime.run_suffix(cpu_boundary)
                assert _tree_allclose(cuda_split_output, full_output, atol=1e-4, rtol=1e-4)
                _assert_tree_shape_device_finite(cross_device_output, full_output, "cuda")
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


def _semantic_split_targets(graph: Any) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for node in graph.ordered_nodes():
        if node.is_input or node.is_output:
            continue
        if node.module_path and node.module_path not in seen:
            seen.add(node.module_path)
            targets.append(node.module_path)
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
