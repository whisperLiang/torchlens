"""Fixwave-2 FW2-POLISH pins for R19 visualization-honesty fixes.

R19-1 (b6, HIGH): the run-fold "+N more <Class>" ellipsis claims the hidden
members are interchangeable, but the uniformity fingerprint was 4 integers
(layer/param counts) — a kwargs-different conv (dilation 2) and a
tanh-for-relu block both folded under the homogeneity claim, and two
DIFFERENT models rendered byte-identical DOT. The fingerprint now folds in
the per-member op-type sequence and func_config digest.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.auto_collapse import _run_fold_hidden_members_uniform

pytestmark = pytest.mark.smoke


class _ConvBlock(nn.Module):
    """Conv+activation block whose output shape is dilation-invariant."""

    def __init__(self, dilation: int = 1, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=dilation, dilation=dilation)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


def _trace(model: nn.Module) -> tl.Trace:
    return tl.trace(model, torch.randn(1, 2, 12, 12))


def test_run_fold_rejects_kwargs_different_hidden_members() -> None:
    """A dilation-2 conv cannot hide inside a "+N more" of dilation-1 blocks.

    Parameter and layer COUNTS are identical across the run, so the old
    4-int fingerprint read the members as uniform; only the func_config
    digest tells them apart.
    """

    trace = _trace(nn.Sequential(_ConvBlock(1), _ConvBlock(1), _ConvBlock(2)))
    assert not _run_fold_hidden_members_uniform(trace, ("0", "1", "2"))


def test_run_fold_rejects_optype_different_hidden_members() -> None:
    """A Tanh block cannot hide inside a "+N more" of ReLU blocks.

    Activations carry zero parameters, so every count in the old
    fingerprint matched; only the op-type sequence tells them apart.
    """

    trace = _trace(
        nn.Sequential(
            _ConvBlock(1),
            _ConvBlock(1),
            _ConvBlock(1, activation=nn.Tanh),
        )
    )
    assert not _run_fold_hidden_members_uniform(trace, ("0", "1", "2"))


def test_run_fold_still_accepts_genuinely_uniform_hidden_members() -> None:
    """Identical hidden members keep folding (no over-rejection).

    The representative (first address) may differ structurally — its stats
    stay visible — so a changed first block with a uniform hidden plateau
    stays foldable.
    """

    trace = _trace(nn.Sequential(_ConvBlock(1), _ConvBlock(1), _ConvBlock(1)))
    assert _run_fold_hidden_members_uniform(trace, ("0", "1", "2"))

    representative_differs = _trace(nn.Sequential(_ConvBlock(2), _ConvBlock(1), _ConvBlock(1)))
    assert _run_fold_hidden_members_uniform(representative_differs, ("0", "1", "2"))


def test_fold_honesty_dot_sources_differ(tmp_path) -> None:
    """Two structurally different models never render byte-identical DOT.

    The b6 probe's failure shape: with the count-only fingerprint, the
    dilated-conv model and the plain model folded to the SAME rendered
    graph. Renders are save-only DOT so the pin is a pure byte comparison.
    """

    sources: list[str] = []
    for variant in ("plain", "dilated"):
        blocks = [_ConvBlock(1), _ConvBlock(1), _ConvBlock(2 if variant == "dilated" else 1)]
        trace = _trace(nn.Sequential(*blocks))
        outpath = tmp_path / f"fold_{variant}"
        trace.draw(
            collapse="max",
            fold_repeats=True,
            vis_save_only=True,
            vis_fileformat="dot",
            vis_outpath=str(outpath),
        )
        sources.append((tmp_path / f"fold_{variant}.dot").read_text())
    assert sources[0] != sources[1], (
        "structurally different models rendered byte-identical DOT; the fold "
        "is hiding a non-uniform member"
    )
