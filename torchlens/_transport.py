"""Device/layout-aware host transport for tensor digest and codec paths.

The historical spelling ``tensor.detach().cpu().contiguous()`` is a no-op on the
common already-contiguous CPU path, but for a CROSS-DEVICE source with a
dense-permuted layout (channels_last / transposed-dense) ``.cpu()`` preserves the
permuted strides (``Tensor.cpu`` defaults to ``memory_format=preserve_format``),
so the trailing ``.contiguous()`` forces a SECOND full materialization — a 2x
transient host peak on exactly the payloads GPU CNN workflows produce
(disputed-r2 b5/R35).

The naive one-line replacement ``.to("cpu", memory_format=contiguous_format)`` is
NOT behavior-identical either: on a same-device CPU noncontiguous strided view it
returns the ALIASED noncontiguous tensor (0 bytes allocated) where the old idiom
returned a contiguous copy. Hence this device/layout-aware branch instead of any
blind spelling. Real-CUDA D2H attestation for the fused single-copy claim rides
the standing GPU probe list.
"""

from __future__ import annotations

import torch


def to_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached, CPU-resident, contiguous view or copy of ``tensor``.

    Semantics match ``tensor.detach().cpu().contiguous()`` exactly — same
    values, standard-contiguous result, no autograd tape — while a
    cross-device dense-permuted source pays ONE host materialization instead
    of two.

    Parameters
    ----------
    tensor:
        Source tensor on any device, any strided layout.

    Returns
    -------
    torch.Tensor
        Detached contiguous CPU tensor. Already-contiguous CPU inputs come
        back as a zero-copy detached view of the same storage, like the
        historical idiom.
    """

    detached = tensor.detach()
    if detached.device.type == "cpu":
        # Same-device: ``.contiguous()`` is the single-copy (or no-op) path.
        # ``.to()`` must NOT be used here — on a noncontiguous CPU strided
        # view it aliases the source without contiguizing.
        return detached.contiguous()
    moved = detached.to("cpu", memory_format=torch.contiguous_format)
    if not moved.is_contiguous():
        # Fail-safe for any copy-machinery edge that ignores the requested
        # memory format: never hand back a noncontiguous transport tensor.
        moved = moved.contiguous()
    return moved
