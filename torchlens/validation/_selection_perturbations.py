"""Deterministic value probes for discrete tensor-selection operations."""

from __future__ import annotations

import torch


def nms_score_order_values(scores: torch.Tensor, saved_output: object) -> torch.Tensor | None:
    """Make a different candidate the unique highest-scoring NMS input.

    Parameters
    ----------
    scores:
        Saved one-dimensional floating scores.
    saved_output:
        Saved NMS indices, whose first element names the previous winner.

    Returns
    -------
    torch.Tensor or None
        Finite scores with a different unique winner when at least two
        candidates exist, otherwise no specialized probe. A live NMS score
        edge must change its first selected index, even for overlapping boxes.
    """
    if scores.ndim != 1 or scores.numel() < 2 or not scores.is_floating_point():
        return None
    if (
        not isinstance(saved_output, torch.Tensor)
        or saved_output.ndim != 1
        or saved_output.numel() == 0
        or saved_output.dtype != torch.long
    ):
        return None
    winner = int(saved_output[0])
    if not 0 <= winner < scores.numel():
        return None
    candidate = torch.zeros_like(scores)
    candidate[(winner + 1) % scores.numel()] = 1
    return candidate
