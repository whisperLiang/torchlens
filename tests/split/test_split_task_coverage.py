"""PyTorch task-conformance tests for the batch-polymorphic split runtime.

These tests cover the four classic families on the Torch backend.  They use
synthetic inputs and, where pretrained weights already exist in the local
cache, the existing model-cache strategy.  They do not download datasets.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import pytest
import torch
from torch import nn
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def _skip_if_module_missing(module_name: str) -> None:
    """Skip when an optional model dependency is unavailable."""

    if find_spec(module_name) is None:
        pytest.skip(f"{module_name!r} is not installed.")


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-4) -> None:
    """Compare two tensors on CPU so mixed-device placement still asserts."""

    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=1e-3)


def _nested_allclose(actual: object, expected: object, *, atol: float = 1e-4) -> None:
    """Recursively compare nested tensor containers."""

    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        _assert_close(actual, expected, atol=atol)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert tuple(actual) == tuple(expected)
        for key in expected:
            _nested_allclose(actual[key], expected[key], atol=atol)
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _nested_allclose(left, right, atol=atol)
        return
    assert actual == expected


def test_image_classification_resnet18_replay_and_train() -> None:
    """ResNet18: residual CNN, canonical B=1, replay + split training."""

    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(0)
    # eval() keeps BatchNorm on running stats so replay is comparable and a
    # canonical B=1 capture is legal for this architecture.
    model = torchvision.models.resnet18(weights=None, num_classes=10).eval()
    example = torch.randn(2, 3, 64, 64)
    runtime = tl.split.prepare(model, example, split_request("50%", trainable=True))

    assert runtime.traced_batch_size == 1
    capture_id = id(runtime.trace)
    graph_id = runtime.graph_identity
    for batch in (1, 2, 3):
        x = torch.randn(batch, 3, 64, 64)
        with torch.no_grad():
            _assert_close(runtime.replay(x), model(x))
        y = torch.randint(0, 10, (batch,))
        boundary = runtime.run_training_prefix(x)
        loss, grads = runtime.train_suffix(boundary, y)
        assert torch.isfinite(loss.detach())
        assert grads
        runtime.backward_prefix(boundary, grads)
        assert id(runtime.trace) == capture_id
        assert runtime.graph_identity == graph_id


def test_text_classification_distilbert_replay_and_train() -> None:
    """DistilBertForSequenceClassification: kwargs, attention, classification loss."""

    _skip_if_module_missing("transformers")
    transformers = import_module("transformers")
    torch.manual_seed(0)
    config = transformers.DistilBertConfig(
        vocab_size=128,
        n_layers=1,
        dim=32,
        hidden_dim=64,
        n_heads=4,
        num_labels=4,
    )
    # eval() disables dropout so split replay is numerically comparable.
    model = transformers.DistilBertForSequenceClassification(config).eval()
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    runtime = tl.split.prepare(
        model,
        input_ids,
        split_request("50%", trainable=True),
        input_kwargs={"attention_mask": attention_mask},
    )
    assert runtime.traced_batch_size == 1
    capture_id = id(runtime.trace)
    for batch in (1, 2, 3):
        ids = torch.randint(0, 128, (batch, 8))
        mask = torch.ones(batch, 8, dtype=torch.long)
        with torch.no_grad():
            split_out = runtime.replay(ids, input_kwargs={"attention_mask": mask})
            full_out = model(ids, attention_mask=mask)
        _assert_close(split_out.logits, full_out.logits)
        labels = torch.randint(0, 4, (batch,))

        def _loss(output: object, target: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(output.logits, target)

        boundary = runtime.run_training_prefix(ids, input_kwargs={"attention_mask": mask})
        loss, grads = runtime.train_suffix(boundary, labels, loss_fn=_loss)
        assert torch.isfinite(loss.detach())
        assert grads
        runtime.backward_prefix(boundary, grads)
        assert id(runtime.trace) == capture_id


def test_object_detection_rfdetr_representative_splits() -> None:
    """RF-DETR Nano: branched detector with nested outputs, reused capture."""

    _skip_if_module_missing("rfdetr")
    torch.manual_seed(0)
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import NestedTensor

    class RFDETRTensorModel(nn.Module):
        def __init__(self, core: nn.Module) -> None:
            super().__init__()
            self.core = core

        def forward(self, images: torch.Tensor) -> object:
            mask = torch.zeros(
                (images.shape[0], images.shape[2], images.shape[3]),
                device=images.device,
                dtype=torch.bool,
            )
            return self.core(NestedTensor(images, mask))

    detector = RFDETRNano()
    model = RFDETRTensorModel(detector.model.model).eval()
    example = torch.randn(1, 3, 384, 384)
    seed = tl.split.prepare(model, example, split_request("25%"))
    assert seed.traced_batch_size == 1
    capture_id = id(seed.trace)
    graph_id = seed.graph_identity
    for point in ("25%", "50%", "75%"):
        runtime = seed if point == "25%" else seed.at(tl.split.percent(float(point.rstrip("%"))))
        x = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            _nested_allclose(runtime.replay(x), model(x), atol=2e-3)
        assert id(runtime.trace) == capture_id
        assert runtime.graph_identity == graph_id


def test_semantic_segmentation_deeplabv3_replay_and_train() -> None:
    """DeepLabV3-ResNet50: ASPP, dense spatial output, segmentation loss."""

    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(0)
    # DeepLabV3's ASPP global-pool path is 1x1 spatially; train-mode BatchNorm
    # at B=1 then has one value per channel and refuses.  eval() uses running
    # stats, so the canonical B=1 capture is legal and still exercises the
    # dense spatial output.
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=5,
        aux_loss=False,
    ).eval()
    example = torch.randn(2, 3, 64, 64)
    runtime = tl.split.prepare(model, example, split_request("50%", trainable=True))
    assert runtime.traced_batch_size == 1
    capture_id = id(runtime.trace)
    for batch in (1, 2):
        x = torch.randn(batch, 3, 64, 64)
        with torch.no_grad():
            split_out = runtime.replay(x)
            full_out = model(x)
        _assert_close(split_out["out"], full_out["out"], atol=2e-3)
        target = torch.randint(0, 5, (batch, 64, 64))

        def _seg_loss(output: object, labels: torch.Tensor) -> torch.Tensor:
            logits = output["out"] if isinstance(output, dict) else output
            return torch.nn.functional.cross_entropy(logits, labels)

        boundary = runtime.run_training_prefix(x)
        loss, grads = runtime.train_suffix(boundary, target, loss_fn=_seg_loss)
        assert torch.isfinite(loss.detach())
        runtime.backward_prefix(boundary, grads)
        assert id(runtime.trace) == capture_id


def test_resnet18_cpu_prefix_cuda_suffix_when_available() -> None:
    """ResNet18 heterogeneous placement: CPU prefix, CUDA suffix."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for heterogeneous placement.")
    _skip_if_module_missing("torchvision")
    torchvision = import_module("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None, num_classes=10).eval()
    example = torch.randn(1, 3, 64, 64)
    runtime = tl.split.prepare(
        model,
        example,
        split_request("50%", placement=PlacementPlan.across("cpu", "cuda:0")),
    )
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        replayed = runtime.replay(x)
        full = model(x)
    _assert_close(replayed, full, atol=5e-4)
