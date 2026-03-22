import pytest
import torch
import torch.nn as nn

import example_models
from torchlens import log_forward_pass, replay_forward_pass


def _tensor_nanequal(t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-5) -> bool:
    if t1.shape != t2.shape:
        return False
    nan_mask_equal = torch.isnan(t1) == torch.isnan(t2)
    valid_mask = ~torch.isnan(t1) & ~torch.isnan(t2)
    values_equal = torch.allclose(t1[valid_mask], t2[valid_mask], atol=atol)
    return bool(torch.all(nan_mask_equal) and values_equal)


def _assert_outputs_equal(lhs, rhs, atol: float = 1e-5) -> None:
    """Recursively compare replay output with direct forward output."""
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        assert _tensor_nanequal(lhs, rhs, atol=atol)
        return
    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
        assert len(lhs) == len(rhs)
        for left_item, right_item in zip(lhs, rhs):
            _assert_outputs_equal(left_item, right_item, atol=atol)
        return
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            _assert_outputs_equal(lhs[key], rhs[key], atol=atol)
        return
    assert lhs == rhs


def _normalize_model_output(output):
    """Convert HF ModelOutput objects to tuples for stable comparisons."""
    if hasattr(output, "to_tuple"):
        return output.to_tuple()
    return output


def _build_lightweight_vision_model(model_family: str):
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
        return transformers.DeiTModel(config).eval(), torch.randn(1, 3, 32, 32)

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
        return transformers.DetrForObjectDetection(config).eval(), torch.randn(1, 3, 64, 64)

    if model_family in {"yolov8", "yolov10"}:
        ultralytics = pytest.importorskip("ultralytics")
        cfg_name = "yolov8n.yaml" if model_family == "yolov8" else "yolov10n.yaml"
        return ultralytics.YOLO(cfg_name).model.eval(), torch.randn(1, 3, 64, 64)

    raise ValueError(f"Unknown model family: {model_family}")


@pytest.mark.smoke
@pytest.mark.parametrize("target_device", ["cpu", "cuda"])
def test_replay_simple_sequential(target_device) -> None:
    if target_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    original_input = torch.randn(2, 10)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn(2, 10)
    replayed_output = replay_forward_pass(model_log, new_input, device=target_device)
    
    # We must run the real model on the target device to compare
    model = model.to(target_device)
    real_output = model(new_input.to(target_device))

    _assert_outputs_equal(replayed_output, real_output, atol=1e-5)
    if isinstance(replayed_output, torch.Tensor):
        assert str(replayed_output.device).startswith(target_device)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_ctor",
    [
        lambda: example_models.SimpleFF(),
        lambda: example_models.ConcatTensors(),
        lambda: example_models.Identity(),
        lambda: example_models.DropoutModelReal(),
    ],
)
def test_replay_toy_models(model_ctor, default_input1) -> None:
    model = model_ctor()
    model.eval()
    model_log = log_forward_pass(
        model,
        default_input1,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn_like(default_input1)
    replayed_output = replay_forward_pass(model_log, new_input)
    real_output = model(new_input)
    _assert_outputs_equal(replayed_output, real_output, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("model_family", ["deit", "detr", "yolov8", "yolov10"])
def test_replay_lightweight_vision_models(model_family: str) -> None:
    model, sample_input = _build_lightweight_vision_model(model_family)

    model_log = log_forward_pass(
        model,
        sample_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn_like(sample_input)
    replayed_output = replay_forward_pass(model_log, new_input)

    with torch.no_grad():
        real_output = model(new_input)

    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(
        _normalize_model_output(replayed_output),
        _normalize_model_output(real_output),
        atol=1e-5,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    ["fasterrcnn_mobilenet_v3_large_320_fpn", "ssdlite320_mobilenet_v3_large"],
)
def test_replay_lightweight_torchvision_detection_models(model_name: str) -> None:
    torchvision = pytest.importorskip("torchvision")
    model = getattr(torchvision.models.detection, model_name)(
        weights=None,
        weights_backbone=None,
    )
    model.eval()

    original_input = [[torch.rand(3, 96, 96)]]
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = [[torch.rand(3, 96, 96)]]
    replayed_output = replay_forward_pass(model_log, new_input)
    with torch.no_grad():
        real_output = model(new_input[0])

    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output, real_output, atol=1e-5)
