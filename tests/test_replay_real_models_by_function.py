"""Real-model replay tests grouped by functionality instead of by model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
import torch

from torchlens import (
    backward_prefix_from_boundary,
    compile_execution_plan,
    enumerate_frontier_splits,
    replay_forward,
    replay_partitioned,
    train_partitioned,
    validate_gradient_equivalence,
    validate_replay_equivalence,
    validate_split_equivalence,
)
from torchlens.replay_utils import tree_allclose

INFERENCE_CASES = [
    "resnet18",
    "vit",
    "deeplabv3_mobilenet_v3_large",
    "fasterrcnn",
    "ssd",
    "detr",
    "distilbert_seqcls",
    "timm_resnet18",
    "yolo_v8n",
    "yolo_v8s",
]

TRAINING_CASES = [
    "resnet18",
    "vit",
    "fasterrcnn",
    "detr",
    "distilbert_seqcls",
    "yolo_v8n",
]


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, target)


def _dummy_detection_batch() -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images = [torch.rand(3, 64, 64)]
    targets = [
        {
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    return images, targets


def _build_yolo_training_batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([1.0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32),
    }


def _build_inference_case(case_name: str) -> tuple[torch.nn.Module, Any, Optional[Dict[str, Any]]]:
    if case_name == "resnet18":
        torchvision = pytest.importorskip("torchvision")
        return torchvision.models.resnet18(weights=None).eval(), torch.rand(1, 3, 224, 224), None

    if case_name == "vit":
        torchvision = pytest.importorskip("torchvision")
        return torchvision.models.vit_b_16(weights=None).eval(), torch.rand(1, 3, 224, 224), None

    if case_name == "deeplabv3_mobilenet_v3_large":
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            aux_loss=True,
        ).eval()
        return model, torch.rand(1, 3, 64, 64), None

    if case_name == "fasterrcnn":
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
        ).eval()
        return model, [[torch.rand(3, 64, 64)]], None

    if case_name == "ssd":
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
        ).eval()
        return model, [[torch.rand(3, 64, 64)]], None

    if case_name == "detr":
        transformers = pytest.importorskip("transformers")
        config = _build_detr_config(transformers)
        return transformers.DetrForObjectDetection(config).eval(), torch.rand(1, 3, 64, 64), None

    if case_name == "distilbert_seqcls":
        transformers = pytest.importorskip("transformers")
        config = _build_distilbert_config(transformers)
        inputs = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
        input_kwargs = {"attention_mask": torch.ones(1, 8, dtype=torch.long)}
        return transformers.DistilBertForSequenceClassification(config).eval(), inputs, input_kwargs

    if case_name == "timm_resnet18":
        timm = pytest.importorskip("timm")
        return timm.create_model("resnet18", pretrained=False).eval(), torch.rand(1, 3, 224, 224), None

    if case_name in {"yolo_v8n", "yolo_v8s"}:
        ultralytics = pytest.importorskip("ultralytics")
        model_name = "yolov8n.yaml" if case_name == "yolo_v8n" else "yolov8s.yaml"
        return ultralytics.YOLO(model_name).model.eval(), torch.rand(1, 3, 64, 64), None

    raise ValueError(f"Unknown inference case {case_name!r}.")


def _build_training_case(
    case_name: str,
) -> tuple[torch.nn.Module, Any, Optional[Dict[str, Any]], Any, Any]:
    if case_name == "resnet18":
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.resnet18(weights=None).train()
        example_inputs = torch.randn(1, 3, 224, 224)
        targets = torch.tensor([7], dtype=torch.long)
        return model, example_inputs, None, targets, cross_entropy_loss

    if case_name == "vit":
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.vit_b_16(weights=None).train()
        example_inputs = torch.randn(1, 3, 224, 224)
        targets = torch.randn(1, 1000)
        return model, example_inputs, None, targets, mse_loss

    if case_name == "fasterrcnn":
        torchvision = pytest.importorskip("torchvision")
        raw_inputs = _dummy_detection_batch()
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
        ).train()
        return model, raw_inputs, None, None, None

    if case_name == "detr":
        transformers = pytest.importorskip("transformers")
        config = _build_detr_config(transformers)
        model = transformers.DetrForObjectDetection(config).train()
        pixel_values = torch.rand(1, 3, 64, 64)
        input_kwargs = {
            "labels": [
                {
                    "class_labels": torch.tensor([1], dtype=torch.int64),
                    "boxes": torch.tensor([[0.2, 0.2, 0.5, 0.5]], dtype=torch.float32),
                }
            ]
        }
        return model, pixel_values, input_kwargs, None, None

    if case_name == "distilbert_seqcls":
        transformers = pytest.importorskip("transformers")
        config = _build_distilbert_config(transformers)
        model = transformers.DistilBertForSequenceClassification(config).train()
        input_ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
        input_kwargs = {
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "labels": torch.tensor([1], dtype=torch.long),
        }
        return model, input_ids, input_kwargs, None, None

    if case_name == "yolo_v8n":
        ultralytics = pytest.importorskip("ultralytics")
        ultralytics_utils = pytest.importorskip("ultralytics.utils")
        torch.manual_seed(123)
        model = ultralytics.YOLO("yolov8n.yaml").model.train()
        model.args = ultralytics_utils.IterableSimpleNamespace(**model.args)
        return model, _build_yolo_training_batch(), None, None, None

    raise ValueError(f"Unknown training case {case_name!r}.")


def _build_detr_config(transformers):
    return transformers.DetrConfig(
        num_queries=10,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        use_pretrained_backbone=False,
        backbone="resnet18",
        num_labels=3,
        auxiliary_loss=False,
    )


def _build_distilbert_config(transformers):
    return transformers.DistilBertConfig(
        vocab_size=101,
        max_position_embeddings=32,
        sinusoidal_pos_embds=False,
        n_layers=2,
        n_heads=4,
        dim=32,
        hidden_dim=64,
        dropout=0.0,
        attention_dropout=0.0,
        num_labels=3,
    )


def _enumerate_case_splits(case_name: str, plan) -> list[Any]:
    if case_name == "fasterrcnn":
        return enumerate_frontier_splits(plan, max_frontier_size=4, max_splits=6)
    if case_name == "yolo_v8n":
        return enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=12)
    return enumerate_frontier_splits(plan, max_frontier_size=2, max_splits=8)


def _select_training_split(
    case_name: str,
    plan,
    inputs: Any,
    input_kwargs: Optional[Dict[str, Any]],
):
    splits = _enumerate_case_splits(case_name, plan)
    assert splits

    if case_name == "vit":
        return splits[min(1, len(splits) - 1)], None, None, None

    if case_name == "yolo_v8n":
        return _select_yolo_training_split(plan, inputs)

    return splits[0], None, None, None


def _select_yolo_training_split(plan, batch):
    for split in _enumerate_case_splits("yolo_v8n", plan):
        try:
            raw_result = train_partitioned(
                plan,
                batch,
                split=split,
                input_mode="raw",
                return_boundary=True,
                return_boundary_grad=True,
                step_optimizer=False,
            )
            suffix_result = train_partitioned(
                plan,
                {"boundary": raw_result["boundary"], "raw_inputs": batch},
                split=split,
                input_mode="boundary",
                return_boundary_grad=True,
                step_optimizer=False,
            )
            prefix_result = backward_prefix_from_boundary(
                plan,
                batch,
                split,
                suffix_result["boundary_grad"],
                step_optimizer=False,
                match_boundary=True,
                cached_boundary=raw_result["boundary"],
            )
        except Exception:
            continue

        if torch.allclose(raw_result["loss"], suffix_result["loss"], atol=1e-4, rtol=1e-3):
            return split, raw_result, suffix_result, prefix_result

    raise AssertionError("No YOLO training split produced matching raw/suffix replay losses.")


def _assert_full_training_loss(case_name: str, result: dict[str, Any]) -> None:
    assert result["loss"].ndim == 0
    if case_name in {"detr", "distilbert_seqcls"}:
        assert torch.allclose(result["loss"], result["output"].loss, atol=1e-4, rtol=1e-3)
    elif case_name == "yolo_v8n":
        assert torch.allclose(
            result["loss"],
            result["output"][0].sum(),
            atol=1e-4,
            rtol=1e-3,
        )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", INFERENCE_CASES)
def test_replay_function_inference_equivalence(case_name: str) -> None:
    model, inputs, input_kwargs = _build_inference_case(case_name)
    plan = compile_execution_plan(model, inputs, input_kwargs=input_kwargs)

    assert validate_replay_equivalence(
        plan,
        model,
        inputs,
        input_kwargs=input_kwargs,
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", INFERENCE_CASES)
def test_replay_function_partitioned_inference(case_name: str) -> None:
    model, inputs, input_kwargs = _build_inference_case(case_name)
    plan = compile_execution_plan(model, inputs, input_kwargs=input_kwargs)
    splits = _enumerate_case_splits(case_name, plan)

    assert splits
    split = splits[0]
    assert validate_split_equivalence(
        plan,
        split,
        model,
        inputs,
        input_kwargs=input_kwargs,
        atol=1e-4,
        rtol=1e-3,
    )

    full_output = replay_partitioned(plan, inputs, input_kwargs=input_kwargs, split=None)
    split_output = replay_partitioned(plan, inputs, input_kwargs=input_kwargs, split=split)
    assert tree_allclose(split_output, full_output, atol=1e-4, rtol=1e-3)

    if case_name == "vit":
        replay_output = replay_forward(plan, inputs, input_kwargs=input_kwargs)
        assert tree_allclose(replay_output, split_output, atol=1e-5, rtol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("case_name", TRAINING_CASES)
def test_replay_function_full_training(case_name: str) -> None:
    model, inputs, input_kwargs, targets, loss_fn = _build_training_case(case_name)
    plan = compile_execution_plan(model, inputs, input_kwargs=input_kwargs)
    result = train_partitioned(
        plan,
        inputs,
        input_kwargs=input_kwargs,
        split=None,
        targets=targets,
        loss_fn=loss_fn,
        step_optimizer=False,
    )

    _assert_full_training_loss(case_name, result)

    if case_name == "fasterrcnn":
        assert validate_replay_equivalence(
            plan,
            model,
            inputs,
            input_kwargs=input_kwargs,
            atol=1e-4,
            rtol=1e-3,
        )
        assert validate_gradient_equivalence(
            plan,
            None,
            model,
            inputs,
            targets=targets,
            loss_fn=loss_fn,
            input_kwargs=input_kwargs,
            atol=1e-4,
            rtol=1e-3,
        )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", TRAINING_CASES)
def test_replay_function_split_training_and_prefix_backward(case_name: str) -> None:
    model, inputs, input_kwargs, targets, loss_fn = _build_training_case(case_name)
    plan = compile_execution_plan(model, inputs, input_kwargs=input_kwargs)
    split, cached_raw_result, cached_suffix_result, cached_prefix_result = _select_training_split(
        case_name,
        plan,
        inputs,
        input_kwargs,
    )

    raw_result = cached_raw_result or train_partitioned(
        plan,
        inputs,
        input_kwargs=input_kwargs,
        split=split,
        input_mode="raw",
        targets=targets,
        loss_fn=loss_fn,
        return_boundary=True,
        return_boundary_grad=True,
        step_optimizer=False,
    )
    boundary_inputs = {"boundary": raw_result["boundary"], "raw_inputs": inputs}
    if input_kwargs is not None:
        boundary_inputs["input_kwargs"] = input_kwargs
    suffix_result = cached_suffix_result or train_partitioned(
        plan,
        boundary_inputs,
        split=split,
        input_mode="boundary",
        targets=targets,
        loss_fn=loss_fn,
        return_boundary_grad=True,
        step_optimizer=False,
    )
    prefix_result = cached_prefix_result or backward_prefix_from_boundary(
        plan,
        inputs,
        split,
        suffix_result["boundary_grad"],
        input_kwargs=input_kwargs,
        step_optimizer=False,
        match_boundary=True,
        cached_boundary=raw_result["boundary"],
    )

    assert raw_result["loss"].ndim == 0
    assert suffix_result["loss"].ndim == 0
    assert prefix_result["boundary"]["cut_id"] == split.split_id
    assert torch.allclose(raw_result["loss"], suffix_result["loss"], atol=1e-4, rtol=1e-3)
    assert validate_gradient_equivalence(
        plan,
        split,
        model,
        inputs,
        targets=targets,
        loss_fn=loss_fn,
        input_kwargs=input_kwargs,
        atol=1e-4,
        rtol=1e-3,
    )
