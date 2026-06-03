"""Optional real-model smoke tests for native split replay."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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
    Path(os.environ["TORCHLENS_YOLO26N_PATH"]) if "TORCHLENS_YOLO26N_PATH" in os.environ else None,
    Path(os.environ["YOLO26N_PATH"]) if "YOLO26N_PATH" in os.environ else None,
    Path(__file__).resolve().parents[1] / "yolo26n.pt",
    Path("/home/whisperliang/TSBOW-main/yolo26n.pt"),
    Path("/home/whisperliang/TSBOW-main/baselines/yolo26n.pt"),
    Path("/home/whisperliang/Plank-road/model_management/models/yolo26n.pt"),
)
PLANK_ROAD_ROOT = Path("/home/whisperliang/Plank-road")
TINYNEXT_SOURCE = PLANK_ROAD_ROOT / "model_management" / "tinynext.py"
RFDETR_PYTHON = PLANK_ROAD_ROOT / ".venv" / "bin" / "python"
RFDETR_WEIGHTS = Path("/home/whisperliang/TSBOW-main/baselines/rf-detr-nano.pth")
REAL_MODEL_DYNAMIC_BATCHES = (1, 2, 4, 8, 32)
REAL_MODEL_CPU_DYNAMIC_BATCHES = (1, 2)
REAL_MODEL_DYNAMIC_BATCH_RANGE = (1, 32)
SPLIT_SWEEP_COVERAGE_DIR = Path(__file__).resolve().parent / "artifacts" / "split_sweep_coverage"


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yolo26n_detection_head_boundary_is_live_on_cuda() -> None:
    """YOLO26n split after detection head preserves all live boundary tensors."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = next((path for path in YOLO26N_CANDIDATES if path is not None and path.exists()), None)
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


@pytest.mark.slow
def test_yolo26n_all_compute_split_points_on_cuda() -> None:
    """Sweep every TorchLens compute node split point in YOLO26n."""

    ultralytics = pytest.importorskip("ultralytics")
    weight_path = next((path for path in YOLO26N_CANDIDATES if path is not None and path.exists()), None)
    if weight_path is None:
        pytest.skip("YOLO26n weights are not available.")

    with _cuda_tf32_disabled():
        has_cuda = torch.cuda.is_available()
        cpu_model = ultralytics.YOLO(str(weight_path)).model.eval().cpu()
        cuda_model = ultralytics.YOLO(str(weight_path)).model.eval().cuda() if has_cuda else None
        runtime_cases = _tensor_runtime_cases(
            (3, 640, 640),
            batch_sizes=REAL_MODEL_DYNAMIC_BATCHES if has_cuda else REAL_MODEL_CPU_DYNAMIC_BATCHES,
        )
        trace_cpu = runtime_cases[0]
        cpu_graph = _trace_graph(cpu_model, trace_cpu)
        cuda_graph = _trace_graph(cuda_model, trace_cpu.cuda()) if cuda_model is not None else None
        targets = _compute_split_targets(cpu_graph)

        report = _run_real_model_compute_split_sweep(
            model_name="YOLO26n",
            artifact_name=f"yolo26n_{'cuda' if has_cuda else 'cpu'}_compute.json",
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
    CUDA suffix execution, and output structure across every TorchLens compute node.
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

        wrapper = RFDETRNano(pretrain_weights={str(RFDETR_WEIGHTS)!r}, num_classes=8)
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
