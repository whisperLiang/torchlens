"""Report successful graph-split index combinations for torchvision models.

This script runs functional replay checks for graph splitting and prints all
successful single split indices. For multi-index input, the current split
implementation uses the maximum index as the actual split boundary, so successful
multi-index combinations are determined by successful max indices.
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Any, Dict, List, Sequence, Tuple

import torch

from torchlens import log_forward_pass, replay_forward_pass, split_and_replay_graph


def _tensor_allclose(t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-4) -> bool:
    if t1.shape != t2.shape:
        return False
    nan_mask = torch.isnan(t1) == torch.isnan(t2)
    valid_mask = ~torch.isnan(t1) & ~torch.isnan(t2)
    if not torch.all(nan_mask):
        return False
    if torch.any(valid_mask):
        return bool(torch.allclose(t1[valid_mask], t2[valid_mask], atol=atol))
    return True


def _compare_outputs(output1: Any, output2: Any, atol: float = 1e-4) -> bool:
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


def _build_model_and_inputs(model_key: str) -> Tuple[torch.nn.Module, Any, Any]:
    import torchvision

    if model_key == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
        )
        original_input = [[torch.rand(3, 224, 224)]]
        new_input = torch.rand(3, 224, 224)
        return model, original_input, new_input

    if model_key == "ssdlite320_mobilenet_v3_large":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
        )
        original_input = [[torch.rand(3, 224, 224)]]
        new_input = torch.rand(3, 224, 224)
        return model, original_input, new_input

    if model_key == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
        )
        original_input = torch.rand(1, 3, 224, 224)
        new_input = torch.rand(1, 3, 224, 224)
        return model, original_input, new_input

    raise ValueError(f"Unsupported model key: {model_key}")


def _find_successful_single_indices(model_key: str, atol: float) -> Dict[str, Any]:
    model, original_input, new_input = _build_model_and_inputs(model_key)
    model.eval()

    model_log = log_forward_pass(
        model,
        original_input,
        layers_to_save="all",
        save_function_args=True,
    )
    full_output = replay_forward_pass(model_log, new_input)

    total_layers = len(model_log.layer_list)
    successful: List[int] = []
    failed: List[int] = []

    for idx in range(total_layers):
        _, split_output = split_and_replay_graph(
            model_log,
            split_layer_indices=idx,
            new_input=new_input,
        )
        if _compare_outputs(full_output, split_output, atol=atol):
            successful.append(idx)
        else:
            failed.append(idx)

    # Current split logic treats multi-index split as max(split_indices).
    # Therefore, any tuple whose max is in successful is also successful.
    max_index = total_layers - 1
    representative_pairs: List[Tuple[int, int]] = []
    for i, j in itertools.combinations(range(total_layers), 2):
        if j in successful:
            representative_pairs.append((i, j))
        if len(representative_pairs) >= 50:
            break

    return {
        "model": model_key,
        "total_layers": total_layers,
        "successful_single_indices": successful,
        "failed_single_indices": failed,
        "num_successful_single_indices": len(successful),
        "multi_index_rule": "multi-index split uses max index as boundary",
        "representative_successful_pairs": representative_pairs,
    }


def main(models: Sequence[str], atol: float, out_path: str | None = None) -> None:
    rows = [_find_successful_single_indices(model_key, atol=atol) for model_key in models]
    payload = json.dumps(rows, ensure_ascii=True, indent=2)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"Wrote report to {out_path}")
    else:
        print(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report successful split index combinations")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "ssdlite320_mobilenet_v3_large",
            "deeplabv3_resnet50",
        ],
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    main(models=args.models, atol=args.atol, out_path=args.out)
