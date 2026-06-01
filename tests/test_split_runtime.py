"""Smoke tests for TorchLens native split replay and split training."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens.split.cache import load_boundary, save_boundary
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.trace_graph import trace_graph_from_model_log
from torchlens.split.training import BoundaryCacheDataset, build_feature_cache
from torchlens.split.validation import assert_split_sweep_coverage
from torchlens.split.validation import assert_canonical_abi_equivalent, boundary_liveness_zero_probe
from torchlens.split.validation import make_split_sweep_case_result
from torchlens.split.validation import summarize_split_sweep_coverage


class TinyResidual(torch.nn.Module):
    """Small residual CNN used by split smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 5, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features[0](x)
        y = self.features[1](h)
        z = self.features[2](y)
        return z + y.mean(dim=1, keepdim=True)


class TinySkip(torch.nn.Module):
    """Skip-connected network with a live boundary cone."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        y = torch.relu(h)
        return self.proj(y) + h


class TinyTrain(torch.nn.Module):
    """Two-layer MLP for split-training comparisons."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(6, 6)
        self.head = torch.nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.backbone(x)))


class TinyBranchTrain(torch.nn.Module):
    """Branching MLP whose early splits carry input passthrough boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Linear(6, 6)
        self.right = torch.nn.Linear(6, 6)
        self.head = torch.nn.Linear(12, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = torch.relu(self.left(x))
        right = torch.sigmoid(self.right(x))
        return self.head(torch.cat([left, right], dim=-1))


class DynamicReshape(torch.nn.Module):
    """Model whose shape expression depends on runtime batch size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return x.reshape(batch, -1) + 1


class RandomSuffix(torch.nn.Module):
    """Model with a stochastic suffix op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1
        return y + torch.rand_like(y)


class MultiOutputDict(torch.nn.Module):
    """Model with a multi-output op and dict return."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        values, indices = torch.max(x, dim=1)
        indices_float = indices.float()
        return {"values": values, "indices": indices_float, "sum": values + indices_float}


@pytest.mark.smoke
def test_split_replay_matches_full_model() -> None:
    """Generated eager prefix/suffix replay matches the full model."""

    torch.manual_seed(0)
    model = TinyResidual().eval()
    x = torch.randn(2, 3, 8, 8)
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:features.1", dynamic_batch=(1, 8)),
    )
    boundary = runtime.run_prefix(x)
    assert boundary.validate(runtime.plan.boundary_specs, shape_env=runtime.trace_graph.shape_env, split_id=runtime.split_id) is None
    split = runtime.run_suffix(boundary)
    full = model(x)
    assert torch.allclose(split, full, atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_split_replay_dynamic_batch_sizes() -> None:
    """Replay stays stable for multiple dynamic batch sizes."""

    torch.manual_seed(0)
    model = TinyResidual().eval()
    runtime = tl.prepare_split(
        model,
        torch.randn(2, 3, 8, 8),
        tl.SplitSpec(boundary="after:features.1", dynamic_batch=(1, 8)),
    )
    for batch_size in (1, 2, 4, 8):
        x = torch.randn(batch_size, 3, 8, 8)
        assert torch.allclose(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_toy_model_compute_split_sweep_has_full_support() -> None:
    """Every TorchLens compute node split point in the toy CNN replays successfully."""

    torch.manual_seed(0)
    model = TinyResidual().eval()
    trace_inputs = torch.randn(2, 3, 8, 8)
    trace_log = tl.log_forward_pass(
        model,
        trace_inputs,
        vis_opt="none",
        layers_to_save="all",
        keep_unsaved_layers=True,
        detach_saved_tensors=False,
        save_function_args=True,
        intervention_ready=True,
    )
    trace_graph = trace_graph_from_model_log(
        trace_log,
        traced_batch_size=int(trace_inputs.shape[0]),
        batch_symbol="B",
        dynamic_batch=(1, 8),
    )
    targets: list[str] = []
    for node in trace_graph.ordered_nodes():
        if node.is_input or node.is_output:
            continue
        if node.torchlens_label:
            targets.append(node.torchlens_label)

    assert any(target.startswith("Conv2d") or target.startswith("relu") for target in targets)
    cases = []
    for target in targets:
        boundary = f"after:{target}"
        batch_statuses = {}
        tested_batches = []
        try:
            runtime = tl.prepare_split(
                model,
                trace_inputs,
                tl.SplitSpec(boundary=boundary, dynamic_batch=(1, 8)),
            )
            for batch_size in (1, 2, 4, 8):
                tested_batches.append(batch_size)
                x = torch.randn(batch_size, 3, 8, 8)
                assert torch.allclose(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)
                batch_statuses[str(batch_size)] = "supported"
        except SplitUnsupportedError as exc:
            if tested_batches:
                batch_statuses[str(tested_batches[-1])] = "unsupported"
            cases.append(
                make_split_sweep_case_result(
                    model_name="TinyResidual",
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
                    model_name="TinyResidual",
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
                    model_name="TinyResidual",
                    split_target=target,
                    boundary=boundary,
                    status="supported",
                    batch_sizes_tested=tuple(tested_batches),
                    batch_statuses=batch_statuses,
                )
            )

    report = summarize_split_sweep_coverage("TinyResidual", cases)
    assert_split_sweep_coverage(report, 1.0, allow_unsupported=False)


@pytest.mark.smoke
def test_split_replay_dynamicizes_trace_batch_shape_literal() -> None:
    """Shape args derived from trace batch are rewritten to runtime batch."""

    model = DynamicReshape().eval()
    runtime = tl.prepare_split(
        model,
        torch.randn(2, 3, 4),
        tl.SplitSpec(boundary="after:reshape_1_1", dynamic_batch=(1, 8)),
    )
    for batch_size in (1, 2, 4, 8):
        x = torch.randn(batch_size, 3, 4)
        assert runtime.replay(x).shape == (batch_size, 12)
        assert torch.allclose(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_split_replay_restores_rng_for_stochastic_suffix_ops() -> None:
    """Stochastic suffix ops keep deterministic replay semantics."""

    torch.manual_seed(0)
    model = RandomSuffix().train()
    x = torch.randn(2, 4)
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:add_1_1", dynamic_batch=(1, 8)),
    )
    first = runtime.replay(x)
    second = runtime.replay(x)
    assert torch.allclose(first, second, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_split_replay_multi_output_dict_return() -> None:
    """Split replay preserves multi-output op use and dict model returns."""

    model = MultiOutputDict().eval()
    x = torch.randn(2, 5)
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:max_2_2", dynamic_batch=(1, 8)),
    )
    actual = runtime.replay(x)
    expected = model(x)
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_split_compiled_mode_replays_or_falls_back_to_eager() -> None:
    """Compiled mode preserves replay semantics even when backend falls back."""

    torch.manual_seed(0)
    model = TinyTrain().eval()
    runtime = tl.prepare_split(
        model,
        torch.randn(2, 6),
        tl.SplitSpec(boundary="after:relu_1_2", dynamic_batch=(1, 4), mode="compiled"),
    )
    x = torch.randn(3, 6)
    assert torch.allclose(runtime.replay(x), model(x), atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_split_boundary_liveness_zero_probe() -> None:
    """Zero-probe flags live skip and residual boundary tensors."""

    torch.manual_seed(0)
    model = TinySkip().eval()
    x = torch.randn(2, 8)
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:relu_1_2", dynamic_batch=(1, 8)),
    )
    boundary = runtime.run_prefix(x)
    live = boundary_liveness_zero_probe(runtime, boundary)
    assert any(live.values())
    assert len(boundary.tensors) >= 2


@pytest.mark.smoke
def test_split_cache_roundtrip_and_collate(tmp_path: Path) -> None:
    """ReplayBoundary cache save/load and collate remain device-neutral."""

    torch.manual_seed(0)
    model = TinyResidual().eval()
    runtime = tl.prepare_split(
        model,
        torch.randn(2, 3, 8, 8),
        tl.SplitSpec(boundary="after:features.1", dynamic_batch=(1, 8)),
    )
    boundary = runtime.run_prefix(torch.randn(2, 3, 8, 8))
    cache_path = tmp_path / "boundary.pt"
    save_boundary(boundary, cache_path)
    loaded = load_boundary(cache_path)
    loaded.validate(runtime.plan.boundary_specs, shape_env=runtime.trace_graph.shape_env, split_id=runtime.split_id)
    collated = type(boundary).collate([boundary.detach(), boundary.detach()])
    assert collated.batch_size == 4
    assert set(collated.tensors) == set(boundary.tensors)
    dataset = BoundaryCacheDataset(tmp_path)
    assert len(dataset) == 1
    assert dataset[0].split_id == boundary.split_id
    cache_dir = tmp_path / "feature_cache"
    build_feature_cache(runtime, [(torch.randn(1, 3, 8, 8),)], cache_dir)
    assert len(BoundaryCacheDataset(cache_dir)) == 1
    runtime_path = tmp_path / "runtime.pt"
    runtime.save(runtime_path)
    loaded_runtime_meta = tl.SplitRuntime.load(runtime_path)
    assert loaded_runtime_meta["split_spec"].boundary == "after:features.1"
    assert set(loaded_runtime_meta["boundary_spec"]) == set(runtime.plan.boundary_specs)


@pytest.mark.smoke
def test_suffix_only_training_matches_frozen_full_training() -> None:
    """Cached suffix-only training matches a frozen-prefix full training step."""

    torch.manual_seed(0)
    x = torch.randn(4, 6)
    targets = torch.randn(4, 3)

    full_model = TinyTrain().train()
    split_model = copy.deepcopy(full_model).train()

    for param in full_model.backbone.parameters():
        param.requires_grad_(False)
    for param in split_model.backbone.parameters():
        param.requires_grad_(False)

    full_opt = torch.optim.SGD(full_model.head.parameters(), lr=0.1)
    split_opt = torch.optim.SGD(split_model.head.parameters(), lr=0.1)

    full_out = full_model(x)
    full_loss = torch.nn.functional.mse_loss(full_out, targets)
    full_opt.zero_grad(set_to_none=True)
    full_loss.backward()
    full_opt.step()

    runtime = tl.prepare_split(
        split_model,
        x,
        tl.SplitSpec(boundary="after:relu_1_2", dynamic_batch=(1, 8), trainable=True),
    )
    boundary = runtime.run_prefix(x)
    loss, _ = runtime.train_suffix(
        boundary,
        targets,
        torch.nn.functional.mse_loss,
        split_opt,
    )
    assert torch.allclose(loss, full_loss.detach(), atol=1e-5, rtol=1e-4)
    for p_full, p_split in zip(full_model.head.parameters(), split_model.head.parameters(), strict=True):
        assert torch.allclose(p_full, p_split, atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_full_split_training_with_boundary_gradients_matches_full_step() -> None:
    """Full split training updates prefix and suffix like a full backward step."""

    torch.manual_seed(0)
    x = torch.randn(4, 6)
    targets = torch.randn(4, 3)

    full_model = TinyTrain().train()
    split_model = copy.deepcopy(full_model).train()

    full_prefix_opt = torch.optim.SGD(full_model.backbone.parameters(), lr=0.05)
    full_suffix_opt = torch.optim.SGD(full_model.head.parameters(), lr=0.05)
    split_prefix_opt = torch.optim.SGD(split_model.backbone.parameters(), lr=0.05)
    split_suffix_opt = torch.optim.SGD(split_model.head.parameters(), lr=0.05)

    full_prefix_opt.zero_grad(set_to_none=True)
    full_suffix_opt.zero_grad(set_to_none=True)
    full_loss = torch.nn.functional.mse_loss(full_model(x), targets)
    full_loss.backward()
    full_suffix_opt.step()
    full_prefix_opt.step()

    runtime = tl.prepare_split(
        split_model,
        x,
        tl.SplitSpec(boundary="after:backbone", dynamic_batch=(1, 8), trainable=True),
    )
    boundary = runtime.run_training_prefix(x)
    loss, boundary_grads = runtime.train_suffix(
        boundary,
        targets,
        torch.nn.functional.mse_loss,
        split_suffix_opt,
    )
    runtime.backward_prefix(boundary, boundary_grads, split_prefix_opt)

    assert torch.allclose(loss, full_loss.detach(), atol=1e-5, rtol=1e-4)
    for p_full, p_split in zip(full_model.parameters(), split_model.parameters(), strict=True):
        assert torch.allclose(p_full, p_split, atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_split_training_skips_nondifferentiable_passthrough_boundaries() -> None:
    """Early branch splits can carry raw-input boundary grads without input grads."""

    torch.manual_seed(0)
    x = torch.randn(4, 6)
    targets = torch.randn(4, 3)

    full_model = TinyBranchTrain().train()
    split_model = copy.deepcopy(full_model).train()

    full_loss = torch.nn.functional.mse_loss(full_model(x), targets)
    full_model.zero_grad(set_to_none=True)
    full_loss.backward()

    runtime = tl.prepare_split(
        split_model,
        x,
        tl.SplitSpec(boundary="after:relu_1_2", dynamic_batch=(1, 8), trainable=True),
    )
    boundary = runtime.run_training_prefix(x)
    assert any(not tensor.requires_grad for tensor in boundary.tensors.values())
    loss, boundary_grads = runtime.train_suffix(boundary, targets, torch.nn.functional.mse_loss)
    runtime.backward_prefix(boundary, boundary_grads)

    assert torch.allclose(loss, full_loss.detach(), atol=1e-5, rtol=1e-4)
    for full_param, split_param in zip(full_model.parameters(), split_model.parameters(), strict=True):
        assert full_param.grad is not None
        assert split_param.grad is not None
        assert torch.allclose(full_param.grad, split_param.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.smoke
def test_replay_boundary_validate_ignores_device_mismatch() -> None:
    """Boundary validation does not treat actual device as ABI."""

    torch.manual_seed(0)
    model = TinyResidual().eval()
    runtime = tl.prepare_split(
        model,
        torch.randn(2, 3, 8, 8),
        tl.SplitSpec(boundary="after:features.1", dynamic_batch=(1, 8)),
    )
    boundary = runtime.run_prefix(torch.randn(2, 3, 8, 8))
    boundary.validate(runtime.plan.boundary_specs, shape_env=runtime.trace_graph.shape_env, split_id=runtime.split_id)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_cuda_canonical_abi_equivalence() -> None:
    """Canonical split ABI matches across CPU and CUDA traces."""

    torch.manual_seed(0)
    base_model = TinyResidual().eval()
    cpu_model = TinyResidual().eval()
    cpu_model.load_state_dict(base_model.state_dict())
    cpu_inputs = torch.randn(2, 3, 8, 8)
    cpu_runtime = tl.prepare_split(
        cpu_model,
        cpu_inputs,
        tl.SplitSpec(boundary="after:features.1", dynamic_batch=(1, 8)),
    )

    cuda_model = TinyResidual().cuda().eval()
    cuda_model.load_state_dict(base_model.state_dict())
    cuda_inputs = cpu_inputs.cuda()
    cuda_runtime = tl.prepare_split(
        cuda_model,
        cuda_inputs,
        tl.SplitSpec(boundary="after:features.1", dynamic_batch=(1, 8)),
    )

    assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("boundary", ("after:features.0", "after:features.1", "50%"))
def test_cpu_prefix_cuda_suffix_replay_across_split_points(boundary: str) -> None:
    """CPU-traced prefix boundaries can replay on a CUDA-traced suffix runtime."""

    torch.manual_seed(0)
    base_model = TinyResidual().eval()
    cpu_model = TinyResidual().eval()
    cuda_model = TinyResidual().cuda().eval()
    cpu_model.load_state_dict(base_model.state_dict())
    cuda_model.load_state_dict(base_model.state_dict())

    x_cpu = torch.randn(2, 3, 8, 8)
    x_cuda = x_cpu.cuda()
    spec = tl.SplitSpec(boundary=boundary, dynamic_batch=(1, 8))
    cpu_runtime = tl.prepare_split(cpu_model, x_cpu, spec)
    cuda_runtime = tl.prepare_split(cuda_model, x_cuda, spec)
    assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)

    cpu_boundary = cpu_runtime.run_prefix(x_cpu)
    assert {tensor.device.type for tensor in cpu_boundary.tensors.values()} == {"cpu"}
    split_output = cuda_runtime.run_suffix(cpu_boundary)
    full_output = cuda_model(x_cuda)
    assert split_output.device.type == "cuda"
    assert torch.allclose(split_output, full_output, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_prefix_cuda_suffix_split_training_roundtrip() -> None:
    """Boundary gradients from a CUDA suffix backpropagate into a CPU prefix graph."""

    torch.manual_seed(0)
    x_cpu = torch.randn(4, 6)
    targets_cpu = torch.randn(4, 3)

    base_model = TinyTrain().train()
    full_model = copy.deepcopy(base_model).train()
    prefix_model = copy.deepcopy(base_model).train()
    suffix_model = copy.deepcopy(base_model).cuda().train()

    full_prefix_opt = torch.optim.SGD(full_model.backbone.parameters(), lr=0.05)
    full_suffix_opt = torch.optim.SGD(full_model.head.parameters(), lr=0.05)
    prefix_opt = torch.optim.SGD(prefix_model.backbone.parameters(), lr=0.05)
    suffix_opt = torch.optim.SGD(suffix_model.head.parameters(), lr=0.05)

    full_prefix_opt.zero_grad(set_to_none=True)
    full_suffix_opt.zero_grad(set_to_none=True)
    full_loss = torch.nn.functional.mse_loss(full_model(x_cpu), targets_cpu)
    full_loss.backward()
    full_suffix_opt.step()
    full_prefix_opt.step()

    spec = tl.SplitSpec(boundary="after:backbone", dynamic_batch=(1, 8), trainable=True)
    cpu_runtime = tl.prepare_split(prefix_model, x_cpu, spec)
    cuda_runtime = tl.prepare_split(suffix_model, x_cpu.cuda(), spec)
    assert_canonical_abi_equivalent(cpu_runtime, cuda_runtime)

    boundary = cpu_runtime.run_training_prefix(x_cpu)
    loss, boundary_grads = cuda_runtime.train_suffix(
        boundary,
        targets_cpu.cuda(),
        torch.nn.functional.mse_loss,
        suffix_opt,
    )
    assert all(grad is None or grad.device.type == "cuda" for grad in boundary_grads.values())
    cpu_runtime.backward_prefix(boundary, boundary_grads, prefix_opt)

    assert torch.allclose(loss.cpu(), full_loss.detach(), atol=1e-5, rtol=1e-4)
    for p_full, p_split in zip(full_model.backbone.parameters(), prefix_model.backbone.parameters(), strict=True):
        assert torch.allclose(p_full, p_split, atol=1e-5, rtol=1e-4)
    for p_full, p_split in zip(full_model.head.parameters(), suffix_model.head.parameters(), strict=True):
        assert torch.allclose(p_full, p_split.cpu(), atol=1e-5, rtol=1e-4)
