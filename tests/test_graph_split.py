"""Tests for graph splitting functionality."""

import copy
from typing import List
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass, replay_forward_pass
from torchlens.user_funcs import (
    replay_forward_pass_differentiable,
    split_and_replay_graph,
    split_and_replay_graph_differentiable,
    split_graph,
)


def _tensor_allclose(t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-5) -> bool:
    """Compare two tensors with NaN handling."""
    if t1.shape != t2.shape:
        return False
    nan_mask = torch.isnan(t1) == torch.isnan(t2)
    valid_mask = ~torch.isnan(t1) & ~torch.isnan(t2)
    if not torch.all(nan_mask):
        return False
    if torch.any(valid_mask):
        return bool(torch.allclose(t1[valid_mask], t2[valid_mask], atol=atol))
    return True


def _compare_outputs(output1, output2, atol: float = 1e-5) -> bool:
    """Recursively compare outputs."""
    if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return _tensor_allclose(output1, output2, atol=atol)
    if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
        if len(output1) != len(output2):
            return False
        return all(_compare_outputs(o1, o2, atol) for o1, o2 in zip(output1, output2))
    if isinstance(output1, dict) and isinstance(output2, dict):
        if output1.keys() != output2.keys():
            return False
        return all(_compare_outputs(output1[k], output2[k], atol) for k in output1)
    return output1 == output2


def _reduce_output_to_scalar(output):
    """Reduce nested tensor outputs to a differentiable scalar."""
    if hasattr(output, "to_tuple"):
        return _reduce_output_to_scalar(output.to_tuple())

    if isinstance(output, torch.Tensor):
        if output.is_floating_point() or output.is_complex():
            return output.sum()
        return None

    if isinstance(output, dict):
        scalar_terms = [_reduce_output_to_scalar(v) for v in output.values()]
    elif isinstance(output, (list, tuple)):
        scalar_terms = [_reduce_output_to_scalar(v) for v in output]
    else:
        return None

    scalar_terms = [term for term in scalar_terms if term is not None]
    if not scalar_terms:
        return None
    total = scalar_terms[0]
    for term in scalar_terms[1:]:
        total = total + term
    return total


def _normalize_model_output(output):
    """Convert HF ModelOutput objects to tuples for stable comparisons."""
    if hasattr(output, "to_tuple"):
        return output.to_tuple()
    return output


def _build_lightweight_vision_model(model_family: str, *, train_mode: bool = False):
    """Build a lightweight vision model and representative image input."""
    if model_family == "deit":
        transformers = pytest.importorskip("transformers")
        config = transformers.DeiTConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = transformers.DeiTModel(config)
        model.train(mode=train_mode)
        return model, torch.randn(1, 3, 32, 32)

    if model_family == "detr":
        transformers = pytest.importorskip("transformers")
        config = transformers.DetrConfig(
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            num_queries=20,
            num_labels=5,
            use_pretrained_backbone=False,
            backbone="resnet18",
        )
        model = transformers.DetrForObjectDetection(config)
        model.train(mode=train_mode)
        return model, torch.randn(1, 3, 64, 64)

    if model_family in {"yolov8", "yolov10"}:
        ultralytics = pytest.importorskip("ultralytics")
        cfg_name = "yolov8n.yaml" if model_family == "yolov8" else "yolov10n.yaml"
        model = ultralytics.YOLO(cfg_name).model
        model.train(mode=train_mode)
        return model, torch.randn(1, 3, 64, 64)

    raise ValueError(f"Unknown model family: {model_family}")


def _build_lightweight_torchvision_detection_model(model_name: str, *, train_mode: bool = False):
    """Build a lightweight torchvision detection model."""
    torchvision = pytest.importorskip("torchvision")
    model = getattr(torchvision.models.detection, model_name)(
        weights=None,
        weights_backbone=None,
    )
    model.train(mode=train_mode)
    return model


def _build_lightweight_torchvision_detection_input(model_name: str, *, train_mode: bool):
    """Build representative inputs for lightweight torchvision detection models."""
    image_count = 2 if train_mode and model_name == "ssdlite320_mobilenet_v3_large" else 1
    images = [torch.rand(3, 96, 96) for _ in range(image_count)]
    if not train_mode:
        return [images]

    targets = []
    for image_idx in range(image_count):
        box_offset = float(8 + 4 * image_idx)
        targets.append(
            {
                "boxes": torch.tensor(
                    [[box_offset, box_offset, box_offset + 40.0, box_offset + 40.0]]
                ),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        )
    return images, targets


def _assert_tensor_close(t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-5) -> None:
    assert torch.allclose(t1, t2, atol=atol), (t1 - t2).abs().max().item()


def _assert_param_grads_close(model: nn.Module, expected_param_grads, atol: float = 1e-5) -> None:
    for name, param in model.named_parameters():
        assert name in expected_param_grads
        expected_grad = expected_param_grads[name]
        if expected_grad is None:
            assert param.grad is None
        else:
            assert param.grad is not None
            _assert_tensor_close(param.grad, expected_grad, atol=atol)


def _clone_input_structure_for_grad(val):
    """Clone nested inputs and enable grad on floating-point tensors."""
    if isinstance(val, torch.Tensor):
        cloned = val.detach().clone()
        if cloned.is_floating_point() or cloned.is_complex():
            cloned.requires_grad_(True)
        return cloned
    if isinstance(val, list):
        return [_clone_input_structure_for_grad(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_clone_input_structure_for_grad(v) for v in val)
    if isinstance(val, dict):
        return {k: _clone_input_structure_for_grad(v) for k, v in val.items()}
    return copy.deepcopy(val)


def _capture_reference_grads(model: nn.Module, input_args, atol: float = 1e-5):
    """Run the original model once and capture output/input/parameter gradients."""
    if isinstance(input_args, tuple):
        direct_inputs = tuple(arg.detach().clone().requires_grad_(True) for arg in input_args)
        model.zero_grad(set_to_none=True)
        direct_output = model(*direct_inputs)
        direct_output.sum().backward()
        input_grads = tuple(arg.grad.detach().clone() for arg in direct_inputs)
    else:
        direct_inputs = input_args.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        direct_output = model(direct_inputs)
        direct_output.sum().backward()
        input_grads = direct_inputs.grad.detach().clone()

    param_grads = {
        name: (param.grad.detach().clone() if param.grad is not None else None)
        for name, param in model.named_parameters()
    }
    return direct_output.detach(), input_grads, param_grads


def _assert_differentiable_split_matches_reference(
    model: nn.Module,
    model_log,
    split_idx: int,
    input_args,
    atol: float = 1e-5,
) -> None:
    """Assert split differentiable replay matches original forward/backward."""
    ref_output, ref_input_grads, ref_param_grads = _capture_reference_grads(
        model, input_args, atol=atol
    )

    if isinstance(input_args, tuple):
        split_inputs = tuple(arg.detach().clone().requires_grad_(True) for arg in input_args)
    else:
        split_inputs = input_args.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    intermediate_features, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=split_inputs,
    )
    split_output.sum().backward()

    assert split_output.requires_grad
    assert intermediate_features
    _assert_tensor_close(split_output.detach(), ref_output, atol=atol)

    if isinstance(split_inputs, tuple):
        assert isinstance(ref_input_grads, tuple)
        for actual, expected in zip(split_inputs, ref_input_grads):
            _assert_tensor_close(actual.grad, expected, atol=atol)
    else:
        _assert_tensor_close(split_inputs.grad, ref_input_grads, atol=atol)

    _assert_param_grads_close(model, ref_param_grads, atol=atol)


def _assert_one_step_weight_update_matches_split_replay(
    model: nn.Module,
    input_args,
    split_idx: int,
    atol: float = 1e-5,
    lr: float = 1e-3,
    train_mode: bool = False,
    allow_stochastic_update_mismatch: bool = False,
    loss_atol: float = 1e-2,
    min_update_cosine: float = 0.65,
    update_norm_ratio_bounds: Tuple[float, float] = (0.8, 1.2),
) -> None:
    """Compare one optimizer step between direct forward/backward and split replay."""

    def _looks_like_detection_targets(val) -> bool:
        return (
            isinstance(val, list)
            and len(val) > 0
            and all(isinstance(item, dict) for item in val)
            and all(("boxes" in item and "labels" in item) for item in val)
        )

    initial_state = copy.deepcopy(model.state_dict())
    model.train(mode=train_mode)

    model_log = log_forward_pass(
        model,
        input_args,
        layers_to_save="all",
        save_function_args=True,
    )

    if isinstance(input_args, tuple):
        direct_inputs = tuple(
            copy.deepcopy(arg)
            if _looks_like_detection_targets(arg)
            else _clone_input_structure_for_grad(arg)
            for arg in input_args
        )
        split_inputs = tuple(
            copy.deepcopy(arg)
            if _looks_like_detection_targets(arg)
            else _clone_input_structure_for_grad(arg)
            for arg in input_args
        )
    else:
        direct_inputs = _clone_input_structure_for_grad(input_args)
        split_inputs = _clone_input_structure_for_grad(input_args)

    cpu_rng_state = torch.random.get_rng_state()

    model.load_state_dict(initial_state)
    model.train(mode=train_mode)
    direct_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    initial_param_values = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }

    torch.random.set_rng_state(cpu_rng_state)
    direct_optimizer.zero_grad(set_to_none=True)
    if isinstance(direct_inputs, tuple):
        direct_output = model(*direct_inputs)
    else:
        direct_output = model(direct_inputs)
    direct_loss = _reduce_output_to_scalar(direct_output)
    if direct_loss is None:
        raise ValueError("Could not reduce direct model output to a differentiable scalar loss.")
    direct_loss.backward()
    direct_optimizer.step()
    direct_updated_params = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }

    model.load_state_dict(initial_state)
    model.train(mode=train_mode)
    split_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    torch.random.set_rng_state(cpu_rng_state)
    split_optimizer.zero_grad(set_to_none=True)
    _, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=split_inputs,
    )
    split_loss = _reduce_output_to_scalar(split_output)
    if split_loss is None:
        raise ValueError("Could not reduce split replay output to a differentiable scalar loss.")
    split_loss.backward()
    split_optimizer.step()

    if allow_stochastic_update_mismatch:
        direct_loss_value = float(direct_loss.detach())
        split_loss_value = float(split_loss.detach())
        assert abs(split_loss_value - direct_loss_value) <= loss_atol

        direct_update = torch.cat(
            [
                (direct_updated_params[name] - initial_param_values[name]).reshape(-1)
                for name in direct_updated_params
            ]
        )
        split_update = torch.cat(
            [
                (param.detach() - initial_param_values[name]).reshape(-1)
                for name, param in model.named_parameters()
            ]
        )
        update_cosine = torch.nn.functional.cosine_similarity(
            direct_update.reshape(1, -1), split_update.reshape(1, -1)
        ).item()
        assert update_cosine >= min_update_cosine

        split_update_norm = float(split_update.norm())
        direct_update_norm = float(direct_update.norm())
        if direct_update_norm > 0.0:
            update_norm_ratio = split_update_norm / direct_update_norm
            lower_bound, upper_bound = update_norm_ratio_bounds
            assert lower_bound <= update_norm_ratio <= upper_bound

        for _, split_param in model.named_parameters():
            assert torch.isfinite(split_param).all()
        return

    for name, split_param in model.named_parameters():
        _assert_tensor_close(split_param.detach(), direct_updated_params[name], atol=atol)


def _pick_spread_indices(total_layers: int) -> List[int]:
    """Pick distinct split indices spread across the graph."""
    if total_layers <= 1:
        return [0]

    candidates = [
        max(0, total_layers // 5),
        max(0, total_layers // 3),
        max(0, total_layers // 2),
        max(0, (2 * total_layers) // 3),
        max(0, (4 * total_layers) // 5),
    ]
    return sorted(set(min(total_layers - 1, idx) for idx in candidates))


def _collect_successful_split_point_combinations(model_log, new_input, atol: float = 1e-5):
    """Collect successful split-point index combinations across all boundaries."""
    full_output = replay_forward_pass(model_log, new_input)
    successful_combinations = []
    seen = set()

    for split_idx in range(len(model_log.layer_list)):
        _, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        if not _compare_outputs(full_output, split_output, atol=atol):
            continue

        _, _, split_labels = split_graph(model_log, split_idx)
        combination = tuple(sorted(model_log[label].creation_order - 1 for label in split_labels))
        if combination not in seen:
            successful_combinations.append(combination)
            seen.add(combination)

    return successful_combinations


@pytest.mark.smoke
@pytest.mark.parametrize("target_device", ["cpu", "cuda"])
def test_split_simple_sequential(target_device):
    if target_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    """Test splitting a simple sequential model."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 5),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    # Test split at middle layer
    new_input = torch.randn(2, 10)

    # Full replay on target device
    full_output = replay_forward_pass(model_log, new_input, device=target_device)

    # Split replay at layer index 5 (middle of network) on target device
    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=5,
        new_input=new_input,
        device=target_device,
    )

    # Outputs should match (they will be on the target device)
    assert _compare_outputs(full_output, split_output, atol=1e-5)
    assert len(intermediate_features) > 0
    if isinstance(full_output, torch.Tensor):
        assert str(full_output.device).startswith(target_device)


@pytest.mark.smoke
def test_split_at_multiple_points():
    """Test splitting at multiple layer indices."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 5),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(2, 10)
    full_output = replay_forward_pass(model_log, new_input)

    # Split at multiple points
    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=[3, 5],
        new_input=new_input,
    )

    assert _compare_outputs(full_output, split_output, atol=1e-5)
    assert len(intermediate_features) == 1


@pytest.mark.smoke
def test_split_with_negative_index():
    """Test splitting with negative indices."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(2, 10)
    full_output = replay_forward_pass(model_log, new_input)

    # Split at -3 (third from end)
    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=-3,
        new_input=new_input,
    )

    assert _compare_outputs(full_output, split_output, atol=1e-5)


def test_split_branching_model():
    """Test splitting a model with branches (DAG structure)."""

    class BranchingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2a = nn.Conv2d(16, 32, 3, padding=1)
            self.conv2b = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 10, 1)

        def forward(self, x):
            x = self.conv1(x)
            a = self.conv2a(x)
            b = self.conv2b(x)
            x = torch.cat([a, b], dim=1)
            x = self.conv3(x)
            return x

    model = BranchingModel()
    model.eval()

    original_input = torch.randn(1, 3, 8, 8)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(1, 3, 8, 8)
    full_output = replay_forward_pass(model_log, new_input)

    # Split in the middle of the branching structure
    total_layers = len(model_log.layer_list)
    split_idx = total_layers // 2

    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=split_idx,
        new_input=new_input,
    )

    assert _compare_outputs(full_output, split_output, atol=1e-5)


def test_split_multi_input_branching_model():
    """Split replay should preserve correctness for DAGs with multiple inputs."""

    class MultiInputBranchingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(10, 8)
            self.right = nn.Linear(6, 8)
            self.merge = nn.Linear(16, 4)

        def forward(self, x, y):
            left = torch.relu(self.left(x))
            right = torch.sigmoid(self.right(y))
            merged = torch.cat([left, right], dim=-1)
            return self.merge(merged)

    model = MultiInputBranchingModel()
    model.eval()

    original_input = (torch.randn(2, 10), torch.randn(2, 6))
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = (torch.randn(2, 10), torch.randn(2, 6))
    full_output = replay_forward_pass(model_log, new_input)

    for split_idx in _pick_spread_indices(len(model_log.layer_list)):
        _, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        assert _compare_outputs(full_output, split_output, atol=1e-5)


def test_split_resnet_block():
    """Test splitting a ResNet-style block with skip connections."""

    class ResNetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = out + identity
            out = self.relu(out)
            return out

    model = ResNetBlock()
    model.eval()

    original_input = torch.randn(1, 16, 8, 8)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(1, 16, 8, 8)
    full_output = replay_forward_pass(model_log, new_input)

    # Split at various points
    for split_idx in [5, 10, 15]:
        if split_idx < len(model_log.layer_list):
            intermediate_features, split_output = split_and_replay_graph(
                model_log,
                split_layer_indices=split_idx,
                new_input=new_input,
            )
            assert _compare_outputs(full_output, split_output, atol=1e-5)


@pytest.mark.parametrize("split_idx", [1, 3, 7, -5, -3])
def test_split_at_various_indices(split_idx):
    """Test splitting at various layer indices."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 40),
        nn.ReLU(),
        nn.Linear(40, 5),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    # Skip if index out of range
    total_layers = len(model_log.layer_list)
    actual_idx = split_idx if split_idx >= 0 else total_layers + split_idx
    if actual_idx < 0 or actual_idx >= total_layers:
        pytest.skip(f"Index {split_idx} out of range for {total_layers} layers")

    new_input = torch.randn(2, 10)
    full_output = replay_forward_pass(model_log, new_input)

    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=split_idx,
        new_input=new_input,
    )

    assert _compare_outputs(full_output, split_output, atol=1e-5)


def test_split_graph_function():
    """Test the split_graph function directly."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    subgraph1, subgraph2, split_labels = split_graph(model_log, 3)

    # Verify split labels are in subgraph1
    for label in split_labels:
        assert label in subgraph1
        assert label not in subgraph2

    # Verify subgraph1 contains input layers
    assert model_log.layer_list[0].layer_label in subgraph1

    # Verify subgraph2 contains output layers
    for output_label in model_log.output_layers:
        assert output_label in subgraph2


def test_split_graph_avoids_dynamic_module_internals():
    """Split points inside a dynamic submodule should snap to a module boundary."""

    class DynamicBlock(nn.Module):
        def forward(self, x):
            topk_vals, topk_indices = torch.topk(x, k=2, dim=1)
            gathered = x.gather(1, topk_indices)
            return topk_vals + gathered

    class DynamicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(6, 6)
            self.block = DynamicBlock()
            self.post = nn.Linear(2, 1)

        def forward(self, x):
            x = self.pre(x)
            x = self.block(x)
            return self.post(x)

    model = DynamicModel()
    model.eval()

    original_input = torch.randn(2, 6)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    block_indices = [
        idx
        for idx, layer in enumerate(model_log.layer_list)
        if getattr(layer, "containing_module", None) == "block:1"
    ]
    assert len(block_indices) >= 2

    requested_split_idx = block_indices[0]
    subgraph1, subgraph2, _ = split_graph(model_log, requested_split_idx)

    label_to_index = {layer.layer_label: idx for idx, layer in enumerate(model_log.layer_list)}
    actual_split_idx = max(label_to_index[label] for label in subgraph1)
    assert actual_split_idx in {block_indices[0] - 1, block_indices[-1]}

    block_labels = {model_log.layer_list[idx].layer_label for idx in block_indices}
    if actual_split_idx == block_indices[0] - 1:
        assert block_labels.isdisjoint(subgraph1)
        assert block_labels.intersection(subgraph2)
    else:
        assert block_labels.issubset(subgraph1)
        assert block_labels.isdisjoint(subgraph2)


def test_dynamic_atomic_module_replay_matches_direct_execution():
    """Atomic dynamic modules should replay via live module forward on new inputs."""

    class DynamicSelector(nn.Module):
        def forward(self, x):
            selected = x[x[:, 0] > 0]
            if selected.shape[0] == 0:
                selected = x[:1]
            return selected

    class DynamicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Identity()
            self.block = DynamicSelector()
            self.post = nn.Linear(4, 1, bias=False)

        def forward(self, x):
            x = self.pre(x)
            x = self.block(x)
            return self.post(x).sum()

    model = DynamicModel()
    model.eval()

    original_input = torch.tensor(
        [
            [1.0, 0.5, 0.2, -0.1],
            [0.7, -0.2, 0.4, 0.3],
            [1.3, 0.1, -0.4, 0.2],
        ]
    )
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.tensor(
        [
            [1.2, 0.1, 0.3, -0.2],
            [-1.0, 0.2, 0.1, 0.0],
            [0.8, -0.4, 0.2, 0.5],
            [-0.7, 0.3, -0.2, 0.6],
        ]
    )

    direct_input = new_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    direct_output = model(direct_input)
    direct_output.backward()
    direct_input_grad = direct_input.grad.detach().clone()
    direct_param_grads = {
        name: param.grad.detach().clone() for name, param in model.named_parameters()
    }

    replay_output_value = replay_forward_pass(model_log, new_input)
    _assert_tensor_close(replay_output_value.detach(), direct_output.detach())

    replay_input = new_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    replay_output = replay_forward_pass_differentiable(model_log, model, replay_input)
    replay_output.backward()
    _assert_tensor_close(replay_output.detach(), direct_output.detach())
    _assert_tensor_close(replay_input.grad, direct_input_grad)
    _assert_param_grads_close(model, direct_param_grads)

    block_indices = [
        idx
        for idx, layer in enumerate(model_log.layer_list)
        if getattr(layer, "containing_module", None) == "block:1"
    ]
    assert block_indices
    split_idx = block_indices[len(block_indices) // 2]

    split_input = new_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    _, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=split_input,
    )
    split_output.backward()
    _assert_tensor_close(split_output.detach(), direct_output.detach())
    _assert_tensor_close(split_input.grad, direct_input_grad)
    _assert_param_grads_close(model, direct_param_grads)


def test_split_graph_avoids_root_dynamic_regions():
    """Split points inside root-level dynamic regions should snap outside the region."""

    class RootDynamicPostprocessModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Identity()

        def forward(self, x):
            x = self.pre(x)
            mask = x[:, 0] > 0
            selected = x[mask]
            out = torch.zeros_like(x)
            out[mask] = selected
            return out

    model = RootDynamicPostprocessModel()
    model.eval()

    original_input = torch.tensor(
        [
            [1.0, 0.2, -0.3],
            [-0.5, 0.4, 0.1],
            [0.7, -0.1, 0.6],
        ]
    )
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    root_dynamic_indices = [
        idx
        for idx, layer in enumerate(model_log.layer_list)
        if getattr(layer, "containing_module", None) is None
        and getattr(layer, "layer_type", "") in {"getitem", "setitem"}
    ]
    assert len(root_dynamic_indices) >= 2

    requested_split_idx = root_dynamic_indices[len(root_dynamic_indices) // 2]
    root_span_start = requested_split_idx
    while (
        root_span_start > 0
        and getattr(model_log.layer_list[root_span_start - 1], "containing_module", None) is None
    ):
        root_span_start -= 1

    root_span_end = requested_split_idx
    while (
        root_span_end + 1 < len(model_log.layer_list)
        and getattr(model_log.layer_list[root_span_end + 1], "containing_module", None) is None
    ):
        root_span_end += 1

    subgraph1, subgraph2, _ = split_graph(model_log, requested_split_idx)

    label_to_index = {layer.layer_label: idx for idx, layer in enumerate(model_log.layer_list)}
    actual_split_idx = max(label_to_index[label] for label in subgraph1)
    assert actual_split_idx in {root_span_start - 1, root_span_end}

    root_dynamic_labels = {
        model_log.layer_list[idx].layer_label for idx in range(root_span_start, root_span_end + 1)
    }
    if actual_split_idx == root_span_start - 1:
        assert root_dynamic_labels.isdisjoint(subgraph1)
        assert root_dynamic_labels.intersection(subgraph2)
    else:
        assert root_dynamic_labels.issubset(subgraph1)
        assert root_dynamic_labels.isdisjoint(subgraph2)

    new_input = original_input.detach().clone()
    full_output = replay_forward_pass(model_log, new_input)
    _, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=requested_split_idx,
        new_input=new_input,
    )
    assert _compare_outputs(full_output, split_output, atol=1e-5)


def test_split_preserves_intermediate_features():
    """Test that intermediate features are correctly extracted."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 5),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(2, 10)

    # Get full replay activations
    full_replay_output = replay_forward_pass(model_log, new_input)

    # Split at layer 5
    split_idx = 5
    split_label = model_log.layer_list[split_idx].layer_label

    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=split_idx,
        new_input=new_input,
    )

    # Verify intermediate feature exists
    assert split_label in intermediate_features
    assert isinstance(intermediate_features[split_label], torch.Tensor)

    # Verify final outputs match
    assert _compare_outputs(full_replay_output, split_output, atol=1e-5)


@pytest.mark.smoke
def test_collect_successful_split_point_combinations():
    """Report real cross-boundary split-point combinations for successful boundaries."""

    class ResidualBranchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 12)
            self.fc2 = nn.Linear(12, 12)
            self.fc3 = nn.Linear(12, 12)
            self.out = nn.Linear(12, 4)

        def forward(self, x):
            base = torch.relu(self.fc1(x))
            branch1 = torch.relu(self.fc2(base))
            branch2 = torch.relu(self.fc3(base))
            merged = branch1 + branch2
            return self.out(merged)

    model = ResidualBranchModel()
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(2, 10)
    combinations = _collect_successful_split_point_combinations(model_log, new_input)

    assert combinations
    assert all(len(combo) > 0 for combo in combinations)
    assert any(len(combo) > 1 for combo in combinations)


def test_split_replay_backward_is_not_autograd_connected():
    """Current replay outputs cannot be backpropagated like the original full graph."""
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    split_idx = len(model_log.layer_list) // 2
    direct_input = torch.randn(2, 10, requires_grad=True)
    direct_output = model(direct_input)
    direct_output.sum().backward()

    direct_first_weight_grad = model[0].weight.grad.detach().clone()
    direct_second_weight_grad = model[2].weight.grad.detach().clone()
    assert not torch.allclose(direct_first_weight_grad, torch.zeros_like(direct_first_weight_grad))
    assert not torch.allclose(
        direct_second_weight_grad, torch.zeros_like(direct_second_weight_grad)
    )

    model.zero_grad(set_to_none=True)

    full_replay_input = direct_input.detach().clone().requires_grad_(True)
    full_replay_output = replay_forward_pass(model_log, full_replay_input)
    assert not full_replay_output.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        full_replay_output.sum().backward()

    split_input = direct_input.detach().clone().requires_grad_(True)
    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=split_idx,
        new_input=split_input,
    )

    assert intermediate_features
    assert not split_output.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        split_output.sum().backward()


def test_differentiable_replay_matches_full_forward_and_backward():
    """Differentiable full replay should match the original model's forward/backward."""
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    direct_input = torch.randn(2, 10, requires_grad=True)
    model.zero_grad(set_to_none=True)
    direct_output = model(direct_input)
    direct_output.sum().backward()
    direct_input_grad = direct_input.grad.detach().clone()
    direct_param_grads = {
        name: param.grad.detach().clone() for name, param in model.named_parameters()
    }

    replay_input = direct_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    replay_output = replay_forward_pass_differentiable(model_log, model, replay_input)
    replay_output.sum().backward()

    assert replay_output.requires_grad
    _assert_tensor_close(replay_output.detach(), direct_output.detach())
    _assert_tensor_close(replay_input.grad, direct_input_grad)
    for name, param in model.named_parameters():
        _assert_tensor_close(param.grad, direct_param_grads[name])


def test_differentiable_split_replay_matches_full_forward_and_backward():
    """Differentiable split replay should preserve the same end-to-end gradients."""
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
    )
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    split_idx = len(model_log.layer_list) // 2

    direct_input = torch.randn(2, 10, requires_grad=True)
    model.zero_grad(set_to_none=True)
    direct_output = model(direct_input)
    direct_output.sum().backward()
    direct_input_grad = direct_input.grad.detach().clone()
    direct_param_grads = {
        name: param.grad.detach().clone() for name, param in model.named_parameters()
    }

    split_input = direct_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    intermediate_features, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=split_input,
    )
    split_output.sum().backward()

    assert split_output.requires_grad
    assert intermediate_features
    assert all(
        not isinstance(feature, torch.Tensor) or feature.requires_grad
        for feature in intermediate_features.values()
    )
    _assert_tensor_close(split_output.detach(), direct_output.detach())
    _assert_tensor_close(split_input.grad, direct_input_grad)
    _assert_param_grads_close(model, direct_param_grads)


def test_differentiable_split_replay_matches_residual_block_backward():
    """Differentiable split replay should preserve gradients on a residual DAG."""

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 12)
            self.fc2 = nn.Linear(12, 12)
            self.fc3 = nn.Linear(12, 12)
            self.out = nn.Linear(12, 4)

        def forward(self, x):
            base = torch.relu(self.fc1(x))
            left = torch.relu(self.fc2(base))
            right = torch.relu(self.fc3(base))
            return self.out(left + right)

    model = ResidualBlock()
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    split_idx = len(model_log.layer_list) // 2
    direct_input = torch.randn(2, 10, requires_grad=True)
    model.zero_grad(set_to_none=True)
    direct_output = model(direct_input)
    direct_output.sum().backward()
    direct_input_grad = direct_input.grad.detach().clone()
    direct_param_grads = {
        name: param.grad.detach().clone() for name, param in model.named_parameters()
    }

    split_input = direct_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    intermediate_features, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=split_input,
    )
    split_output.sum().backward()

    assert split_output.requires_grad
    assert any(
        isinstance(feature, torch.Tensor) and feature.requires_grad
        for feature in intermediate_features.values()
    )
    _assert_tensor_close(split_output.detach(), direct_output.detach())
    _assert_tensor_close(split_input.grad, direct_input_grad)
    _assert_param_grads_close(model, direct_param_grads)


def test_differentiable_split_replay_matches_multi_input_dag_backward():
    """Differentiable split replay should preserve gradients on a multi-input DAG."""

    class MultiInputDag(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(10, 8)
            self.right = nn.Linear(6, 8)
            self.merge = nn.Linear(16, 4)

        def forward(self, x, y):
            left = torch.relu(self.left(x))
            right = torch.sigmoid(self.right(y))
            merged = torch.cat([left, right], dim=-1)
            return self.merge(merged)

    model = MultiInputDag()
    model.eval()

    original_input = (torch.randn(2, 10), torch.randn(2, 6))
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    split_idx = len(model_log.layer_list) // 2
    direct_x = torch.randn(2, 10, requires_grad=True)
    direct_y = torch.randn(2, 6, requires_grad=True)
    model.zero_grad(set_to_none=True)
    direct_output = model(direct_x, direct_y)
    direct_output.sum().backward()
    direct_x_grad = direct_x.grad.detach().clone()
    direct_y_grad = direct_y.grad.detach().clone()
    direct_param_grads = {
        name: param.grad.detach().clone() for name, param in model.named_parameters()
    }

    split_x = direct_x.detach().clone().requires_grad_(True)
    split_y = direct_y.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    intermediate_features, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=(split_x, split_y),
    )
    split_output.sum().backward()

    assert split_output.requires_grad
    assert intermediate_features
    _assert_tensor_close(split_output.detach(), direct_output.detach())
    _assert_tensor_close(split_x.grad, direct_x_grad)
    _assert_tensor_close(split_y.grad, direct_y_grad)
    _assert_param_grads_close(model, direct_param_grads)


def test_split_replay_matches_one_step_weight_update_on_residual_dag():
    """A single SGD step after split replay should match the direct model step."""

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 12)
            self.fc2 = nn.Linear(12, 12)
            self.fc3 = nn.Linear(12, 12)
            self.out = nn.Linear(12, 4)

        def forward(self, x):
            base = torch.relu(self.fc1(x))
            left = torch.relu(self.fc2(base))
            right = torch.relu(self.fc3(base))
            return self.out(left + right)

    model = ResidualBlock()
    model.eval()
    sample_input = torch.randn(2, 10)
    base_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )
    split_idx = len(base_log.layer_list) // 2

    train_input = torch.randn(2, 10)
    _assert_one_step_weight_update_matches_split_replay(
        model,
        train_input,
        split_idx=split_idx,
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_family", ["deit", "detr", "yolov8", "yolov10"])
def test_differentiable_split_lightweight_vision_models(model_family: str):
    """Differentiable split replay should preserve forward/backward on lightweight models."""
    train_mode = model_family in {"yolov8", "yolov10"}
    model, sample_input = _build_lightweight_vision_model(
        model_family,
        train_mode=train_mode,
    )

    model_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )
    split_indices = _pick_spread_indices(len(model_log.layer_list))
    split_idx = split_indices[len(split_indices) // 2]

    direct_input = sample_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    direct_output = model(direct_input)
    direct_loss = _reduce_output_to_scalar(direct_output)
    if direct_loss is None:
        raise ValueError("Could not reduce direct model output to a differentiable scalar loss.")
    direct_loss.backward()
    direct_input_grad = direct_input.grad.detach().clone()
    direct_param_grads = {
        name: (param.grad.detach().clone() if param.grad is not None else None)
        for name, param in model.named_parameters()
    }

    split_input = sample_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    _, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=split_input,
    )
    split_loss = _reduce_output_to_scalar(split_output)
    if split_loss is None:
        raise ValueError("Could not reduce split replay output to a differentiable scalar loss.")
    split_loss.backward()

    assert split_output is not None
    assert _compare_outputs(
        _normalize_model_output(split_output),
        _normalize_model_output(direct_output),
        atol=1e-5,
    )
    _assert_tensor_close(split_input.grad, direct_input_grad)
    _assert_param_grads_close(model, direct_param_grads)


@pytest.mark.slow
@pytest.mark.parametrize("model_family", ["deit", "detr", "yolov8", "yolov10"])
def test_split_replay_matches_one_step_weight_update_lightweight_vision_models(
    model_family: str,
):
    """One-step updates should match on lightweight vision models."""
    train_mode = model_family in {"yolov8", "yolov10"}
    model, sample_input = _build_lightweight_vision_model(
        model_family,
        train_mode=train_mode,
    )

    base_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )
    split_indices = _pick_spread_indices(len(base_log.layer_list))
    split_idx = split_indices[len(split_indices) // 2]

    train_input = torch.randn_like(sample_input)
    _assert_one_step_weight_update_matches_split_replay(
        model,
        train_input,
        split_idx=split_idx,
        atol=1e-5,
        train_mode=train_mode,
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_family", ["deit", "detr", "yolov8", "yolov10"])
def test_split_lightweight_vision_models(model_family: str):
    """Split replay should match direct execution on lightweight vision models."""
    model, sample_input = _build_lightweight_vision_model(model_family)
    comparison_atol = 1e-4 if model_family == "yolov10" else 1e-5

    model_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn_like(sample_input)
    full_output = replay_forward_pass(model_log, new_input)
    with torch.no_grad():
        direct_output = model(new_input)

    normalized_full_output = _normalize_model_output(full_output)
    normalized_direct_output = _normalize_model_output(direct_output)
    assert _compare_outputs(
        normalized_full_output,
        normalized_direct_output,
        atol=comparison_atol,
    )

    split_indices = _pick_spread_indices(len(model_log.layer_list))
    if len(split_indices) > 2:
        split_indices = [split_indices[1], split_indices[-2]]

    for split_idx in split_indices:
        intermediate_features, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        normalized_split_output = _normalize_model_output(split_output)
        assert _compare_outputs(
            normalized_full_output,
            normalized_split_output,
            atol=comparison_atol,
        )
        assert _compare_outputs(
            normalized_direct_output,
            normalized_split_output,
            atol=comparison_atol,
        )
        assert intermediate_features


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    ["fasterrcnn_mobilenet_v3_large_320_fpn", "ssdlite320_mobilenet_v3_large"],
)
def test_differentiable_split_lightweight_torchvision_detection_models(model_name: str):
    """Differentiable split replay should preserve training-time gradients on lightweight detection models."""
    model = _build_lightweight_torchvision_detection_model(model_name, train_mode=True)
    sample_input = _build_lightweight_torchvision_detection_input(model_name, train_mode=True)

    model_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )
    split_indices = _pick_spread_indices(len(model_log.layer_list))
    split_idx = split_indices[len(split_indices) // 2]
    cpu_rng_state = torch.random.get_rng_state()

    direct_images = _clone_input_structure_for_grad(sample_input[0])
    direct_targets = copy.deepcopy(sample_input[1])
    model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng_state)
    direct_output = model(direct_images, direct_targets)
    direct_loss = _reduce_output_to_scalar(direct_output)
    if direct_loss is None:
        raise ValueError("Could not reduce direct model output to a differentiable scalar loss.")
    direct_loss.backward()
    direct_image_grads = [image.grad.detach().clone() for image in direct_images]
    direct_param_grads = {
        name: (param.grad.detach().clone() if param.grad is not None else None)
        for name, param in model.named_parameters()
    }

    split_images = _clone_input_structure_for_grad(sample_input[0])
    split_targets = copy.deepcopy(sample_input[1])
    model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng_state)
    _, split_output = split_and_replay_graph_differentiable(
        model_log,
        model,
        split_layer_indices=split_idx,
        new_input=(split_images, split_targets),
    )
    split_loss = _reduce_output_to_scalar(split_output)
    if split_loss is None:
        raise ValueError("Could not reduce split replay output to a differentiable scalar loss.")
    split_loss.backward()

    assert _compare_outputs(split_output, direct_output, atol=1e-5)
    for actual_grad, expected_grad in zip((image.grad for image in split_images), direct_image_grads):
        _assert_tensor_close(actual_grad, expected_grad)
    _assert_param_grads_close(model, direct_param_grads)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    ["fasterrcnn_mobilenet_v3_large_320_fpn", "ssdlite320_mobilenet_v3_large"],
)
def test_split_replay_matches_one_step_weight_update_lightweight_torchvision_detection_models(
    model_name: str,
):
    """One-step updates should match on lightweight torchvision detection models."""
    model = _build_lightweight_torchvision_detection_model(model_name, train_mode=True)
    sample_input = _build_lightweight_torchvision_detection_input(model_name, train_mode=True)

    base_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )
    split_indices = _pick_spread_indices(len(base_log.layer_list))
    split_idx = split_indices[len(split_indices) // 2]

    train_input = _build_lightweight_torchvision_detection_input(model_name, train_mode=True)
    _assert_one_step_weight_update_matches_split_replay(
        model,
        train_input,
        split_idx=split_idx,
        atol=1e-5,
        lr=1e-5,
        train_mode=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    ["fasterrcnn_mobilenet_v3_large_320_fpn", "ssdlite320_mobilenet_v3_large"],
)
def test_split_lightweight_torchvision_detection_models(model_name: str):
    """Split replay should match direct execution on lightweight torchvision detection models."""
    model = _build_lightweight_torchvision_detection_model(model_name)
    sample_input = _build_lightweight_torchvision_detection_input(model_name, train_mode=False)

    model_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = _build_lightweight_torchvision_detection_input(model_name, train_mode=False)
    full_output = replay_forward_pass(model_log, new_input)
    with torch.no_grad():
        direct_output = model(new_input[0])

    assert _compare_outputs(full_output, direct_output, atol=1e-5)

    split_indices = _pick_spread_indices(len(model_log.layer_list))
    if len(split_indices) > 2:
        split_indices = [split_indices[1], split_indices[-2]]

    for split_idx in split_indices:
        intermediate_features, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        assert _compare_outputs(full_output, split_output, atol=1e-5)
        assert _compare_outputs(direct_output, split_output, atol=1e-5)
        assert intermediate_features


def test_split_multiple_outputs_model():
    """Test splitting a model that returns multiple outputs."""

    class MultiOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            self.fc3 = nn.Linear(20, 3)

        def forward(self, x):
            x = self.fc1(x)
            out1 = self.fc2(x)
            out2 = self.fc3(x)
            return out1, out2

    model = MultiOutputModel()
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(2, 10)
    full_output = replay_forward_pass(model_log, new_input)

    # Split in the middle
    split_idx = len(model_log.layer_list) // 2
    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=split_idx,
        new_input=new_input,
    )

    assert _compare_outputs(full_output, split_output, atol=1e-5)


def test_split_error_handling():
    """Test error handling for invalid split indices."""
    model = nn.Sequential(nn.Linear(10, 5))
    model.eval()

    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    # Test out of range index
    with pytest.raises(ValueError, match="out of range"):
        split_and_replay_graph(
            model_log,
            split_layer_indices=1000,
            new_input=original_input,
        )

    # Test very negative index
    with pytest.raises(ValueError, match="out of range"):
        split_and_replay_graph(
            model_log,
            split_layer_indices=-1000,
            new_input=original_input,
        )

