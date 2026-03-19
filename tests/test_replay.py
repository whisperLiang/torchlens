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


@pytest.mark.smoke
def test_replay_simple_sequential() -> None:
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
    replayed_output = replay_forward_pass(model_log, new_input)
    real_output = model(new_input)

    _assert_outputs_equal(replayed_output, real_output, atol=1e-5)


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


@pytest.mark.parametrize(
    "model_ctor, atol",
    [
        ("alexnet", 1e-5),
        ("vgg16", 1e-5),
        ("vgg19", 1e-5),
        ("resnet18", 1e-5),
        ("resnet50", 1e-5),
        ("wide_resnet50_2", 1e-5),
        ("wide_resnet101_2", 1e-5),
        ("resnext50_32x4d", 1e-5),
        ("resnext101_64x4d", 1e-5),
        ("squeezenet1_0", 1e-5),
        ("squeezenet1_1", 1e-5),
        ("mobilenet_v2", 1e-5),
        ("mobilenet_v3_small", 1e-5),
        ("mobilenet_v3_large", 1e-5),
        ("shufflenet_v2_x1_0", 1e-5),
        ("shufflenet_v2_x1_5", 1e-5),
        ("mnasnet0_5", 1e-5),
        ("mnasnet1_3", 1e-5),
        ("googlenet", 1e-5),
        ("densenet121", 1e-5),
        ("densenet169", 1e-5),
        ("efficientnet_b0", 1e-5),
        ("efficientnet_b3", 1e-5),
        ("efficientnet_b6", 1e-5),
        ("convnext_tiny", 1e-5),
        ("convnext_large", 1e-5),
        ("regnet_x_400mf", 1e-5),
        ("regnet_y_400mf", 1e-5),
        ("regnet_x_32gf", 1e-5),
        ("vit_b_16", 1e-5),
        ("swin_v2_b", 1e-5),
        ("maxvit_t", 1e-5),
    ],
)
def test_replay_real_world_models(model_ctor: str, atol: float, default_input1) -> None:
    torchvision = pytest.importorskip("torchvision")
    model = getattr(torchvision.models, model_ctor)()
    model.eval()

    model_log = log_forward_pass(
        model,
        default_input1,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.randn_like(default_input1)
    replayed_output = replay_forward_pass(model_log, new_input)

    with torch.no_grad():
        real_output = model(new_input)

    _assert_outputs_equal(replayed_output, real_output, atol=atol)


@pytest.mark.slow
def test_replay_fasterrcnn_resnet50_fpn() -> None:
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )
    model.eval()

    original_images = [torch.rand(3, 224, 224)]
    model_log = log_forward_pass(
        model,
        [original_images],
        layers_to_save="all",
        save_function_args=True,
    )

    # Test with same input - should match exactly
    replayed_output = replay_forward_pass(model_log, original_images[0])
    with torch.no_grad():
        real_output = model(original_images)

    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output, real_output, atol=1e-5)


@pytest.mark.slow
def test_replay_bert_model() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=30522,
    )
    model = transformers.BertModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    model_log = log_forward_pass(
        model,
        input_ids,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input_ids = torch.randint(0, config.vocab_size, (2, 16))
    replayed_output = replay_forward_pass(model_log, new_input_ids)

    with torch.no_grad():
        real_output = model(new_input_ids)

    # Both outputs should have the same type and structure now
    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output.to_tuple(), real_output.to_tuple(), atol=1e-5)


@pytest.mark.slow
def test_replay_inception_v3() -> None:
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.inception_v3(weights=None)
    model.eval()

    original_input = torch.rand(1, 3, 299, 299)
    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input = torch.rand(1, 3, 299, 299)
    replayed_output = replay_forward_pass(model_log, new_input)

    with torch.no_grad():
        real_output = model(new_input)

    _assert_outputs_equal(replayed_output, real_output, atol=1e-5)


@pytest.mark.slow
def test_replay_gpt2_model() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_positions=64,
        n_ctx=64,
        vocab_size=50257,
        use_cache=False,
    )
    model = transformers.GPT2Model(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    model_log = log_forward_pass(
        model,
        input_ids,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input_ids = torch.randint(0, config.vocab_size, (2, 16))
    replayed_output = replay_forward_pass(model_log, new_input_ids)

    with torch.no_grad():
        real_output = model(new_input_ids)

    # Both outputs should have the same type and structure now
    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output.to_tuple(), real_output.to_tuple(), atol=1e-5)


@pytest.mark.slow
def test_replay_maskrcnn_resnet50_fpn() -> None:
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )
    model.eval()

    original_images = [torch.rand(3, 224, 224)]
    model_log = log_forward_pass(
        model,
        [original_images],
        layers_to_save="all",
        save_function_args=True,
    )

    # Test with same input - should match exactly
    replayed_output = replay_forward_pass(model_log, original_images[0])
    with torch.no_grad():
        real_output = model(original_images)

    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output, real_output, atol=1e-5)


@pytest.mark.slow
def test_replay_roberta_model() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.RobertaConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=50265,
    )
    model = transformers.RobertaModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    model_log = log_forward_pass(
        model,
        input_ids,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input_ids = torch.randint(0, config.vocab_size, (2, 16))
    replayed_output = replay_forward_pass(model_log, new_input_ids)

    with torch.no_grad():
        real_output = model(new_input_ids)

    # Both outputs should have the same type and structure now
    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output.to_tuple(), real_output.to_tuple(), atol=1e-5)


@pytest.mark.slow
def test_replay_distilbert_model() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.DistilBertConfig(
        dim=128,
        hidden_dim=256,
        n_layers=2,
        n_heads=4,
        vocab_size=30522,
    )
    model = transformers.DistilBertModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    model_log = log_forward_pass(
        model,
        input_ids,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input_ids = torch.randint(0, config.vocab_size, (2, 16))
    replayed_output = replay_forward_pass(model_log, new_input_ids)

    with torch.no_grad():
        real_output = model(new_input_ids)

    # Both outputs should have the same type and structure now
    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output.to_tuple(), real_output.to_tuple(), atol=1e-5)


@pytest.mark.slow
def test_replay_t5_encoder_model() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.T5Config(
        d_model=128,
        d_ff=256,
        num_layers=2,
        num_heads=4,
        vocab_size=32128,
    )
    model = transformers.T5EncoderModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    model_log = log_forward_pass(
        model,
        input_ids,
        layers_to_save="all",
        save_function_args=True,
    )

    new_input_ids = torch.randint(0, config.vocab_size, (2, 16))
    replayed_output = replay_forward_pass(model_log, new_input_ids)

    with torch.no_grad():
        real_output = model(new_input_ids)

    # Both outputs should have the same type and structure now
    assert type(replayed_output) == type(real_output)
    _assert_outputs_equal(replayed_output.to_tuple(), real_output.to_tuple(), atol=1e-5)

