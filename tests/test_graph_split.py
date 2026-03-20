"""Tests for graph splitting functionality."""

from typing import List

import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass, replay_forward_pass
from torchlens.user_funcs import split_and_replay_graph, split_graph


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


@pytest.mark.smoke
def test_split_simple_sequential():
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

    # Full replay
    full_output = replay_forward_pass(model_log, new_input)

    # Split replay at layer index 5 (middle of network)
    intermediate_features, split_output = split_and_replay_graph(
        model_log,
        split_layer_indices=5,
        new_input=new_input,
    )

    # Outputs should match
    assert _compare_outputs(full_output, split_output, atol=1e-5)
    assert len(intermediate_features) > 0


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


@pytest.mark.parametrize(
    "model_name,split_indices",
    [
        ("alexnet", [10, 20]),
        ("vgg16", [15, 30]),
        ("resnet18", [20, 40]),
        ("resnet50", [30, 60]),
        ("mobilenet_v2", [15, 30]),
        ("squeezenet1_0", [10, 20]),
        ("densenet121", [25, 50]),
        ("efficientnet_b0", [20, 40]),
        ("googlenet", [20, 50]),
        ("shufflenet_v2_x1_0", [15, 35]),
        ("convnext_tiny", [20, 60]),
        ("mnasnet1_0", [30, 50]),
        ("swin_t", [30, 80]),
    ],
)
def test_split_real_world_models(model_name, split_indices, default_input1):
    """Test graph splitting on real-world torchvision models."""
    torchvision = pytest.importorskip("torchvision")

    model = getattr(torchvision.models, model_name)(weights=None)
    model.eval()

    model_log = log_forward_pass(
        model,
        default_input1,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn_like(default_input1)
    full_output = replay_forward_pass(model_log, new_input)

    # Test each split index
    for split_idx in split_indices:
        if split_idx >= len(model_log.layer_list):
            continue

        intermediate_features, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )

        assert _compare_outputs(full_output, split_output, atol=1e-4)
        assert len(intermediate_features) > 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    [
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "retinanet_resnet50_fpn",
        "ssdlite320_mobilenet_v3_large",
    ],
)
def test_split_detection_models_multi_indices(model_name):
    """Test graph splitting on detection models with multiple split indices."""
    torchvision = pytest.importorskip("torchvision")

    model = getattr(torchvision.models.detection, model_name)(weights=None, weights_backbone=None)
    model.eval()

    # Detection models consume list[Tensor] as a single positional argument.
    original_input = [[torch.rand(3, 224, 224)]]
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.rand_like(original_input[0][0])
    full_output = replay_forward_pass(model_log, new_input)

    split_indices = _pick_spread_indices(len(model_log.layer_list))
    for split_idx in split_indices:
        intermediate_features, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        assert _compare_outputs(full_output, split_output, atol=1e-4)
        assert len(intermediate_features) > 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    [
        "fcn_resnet50",
        "deeplabv3_resnet50",
        "lraspp_mobilenet_v3_large",
    ],
)
def test_split_segmentation_models_multi_indices(model_name):
    """Test graph splitting on segmentation models with multiple split indices."""
    torchvision = pytest.importorskip("torchvision")

    model = getattr(torchvision.models.segmentation, model_name)(
        weights=None, weights_backbone=None
    )
    model.eval()

    original_input = torch.rand(1, 3, 224, 224)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.rand_like(original_input)
    full_output = replay_forward_pass(model_log, new_input)

    split_indices = _pick_spread_indices(len(model_log.layer_list))
    for split_idx in split_indices:
        intermediate_features, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        assert _compare_outputs(full_output, split_output, atol=1e-4)
        assert len(intermediate_features) > 0


@pytest.mark.slow
def test_split_inception_v3():
    """Test splitting Inception v3 with auxiliary outputs."""
    torchvision = pytest.importorskip("torchvision")

    model = torchvision.models.inception_v3(weights=None, aux_logits=False)
    model.eval()

    original_input = torch.rand(1, 3, 299, 299)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.rand(1, 3, 299, 299)
    full_output = replay_forward_pass(model_log, new_input)

    # Split at multiple points
    total_layers = len(model_log.layer_list)
    split_indices = [total_layers // 4, total_layers // 2, 3 * total_layers // 4]

    for split_idx in split_indices:
        intermediate_features, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        assert _compare_outputs(full_output, split_output, atol=1e-4)


@pytest.mark.slow
def test_split_transformer_model():
    """Test splitting a transformer model (ViT)."""
    torchvision = pytest.importorskip("torchvision")

    model = torchvision.models.vit_b_16(weights=None)
    model.eval()

    original_input = torch.rand(1, 3, 224, 224)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.rand(1, 3, 224, 224)
    full_output = replay_forward_pass(model_log, new_input)

    # Split at various points in the transformer
    total_layers = len(model_log.layer_list)
    split_indices = [total_layers // 3, total_layers // 2, 2 * total_layers // 3]

    for split_idx in split_indices:
        intermediate_features, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=split_idx,
            new_input=new_input,
        )
        assert _compare_outputs(full_output, split_output, atol=1e-4)


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
