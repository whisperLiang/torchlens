"""Verify a pinned CPU torch/vision/audio installation before running CI tests."""

from __future__ import annotations

import argparse


def main() -> None:
    """Check exact binary versions and execute torchvision's native NMS operator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torch_version")
    parser.add_argument("torchvision_version")
    args = parser.parse_args()

    import torch
    import torchaudio
    import torchvision

    for module, expected in (
        (torch, args.torch_version),
        (torchvision, args.torchvision_version),
        (torchaudio, args.torch_version),
    ):
        if module.__version__ != expected:
            raise RuntimeError(f"{module.__name__}: expected {expected}, got {module.__version__}")
        print(f"{module.__name__}=={module.__version__}")

    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    scores = torch.tensor([1.0])
    if torchvision.ops.nms(boxes, scores, 0.5).tolist() != [0]:
        raise RuntimeError("torchvision.ops.nms did not return the expected box")


if __name__ == "__main__":
    main()
