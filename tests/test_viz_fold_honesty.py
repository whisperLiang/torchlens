"""Fixwave-2 FW2-POLISH pins for R19 visualization-honesty fixes.

R19-1 (b6, HIGH): the run-fold "+N more <Class>" ellipsis claims the hidden
members are interchangeable, but the uniformity fingerprint was 4 integers
(layer/param counts) — a kwargs-different conv (dilation 2) and a
tanh-for-relu block both folded under the homogeneity claim, and two
DIFFERENT models rendered byte-identical DOT. The fingerprint now folds in
the per-member op-type sequence and func_config digest.

T9 (grind-p3, HIGH): the fingerprint was compared only across the HIDDEN
members (``addresses[1:]``), so a plateau uniformly different from its own
visible representative still folded — an all-Tanh plateau behind a ReLU
representative rendered byte-identical DOT to the all-ReLU model. The
uniformity check now spans EVERY member including the representative; a
run whose representative differs splits, so the plateau re-folds from its
own structurally-matching representative instead of hiding the change.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.auto_collapse import _run_fold_members_uniform

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
    assert not _run_fold_members_uniform(trace, ("0", "1", "2"))


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
    assert not _run_fold_members_uniform(trace, ("0", "1", "2"))


def test_run_fold_still_accepts_genuinely_uniform_members() -> None:
    """Identical members keep folding (no over-rejection)."""

    trace = _trace(nn.Sequential(_ConvBlock(1), _ConvBlock(1), _ConvBlock(1)))
    assert _run_fold_members_uniform(trace, ("0", "1", "2"))


def test_run_fold_rejects_representative_unlike_hidden_members() -> None:
    """A run whose representative differs from its plateau must not fold whole.

    REVIEWED rebaseline (T9, grind-p3): this exact shape was previously
    pinned as foldable on the theory that the representative's own stats
    stay visible. The theory was wrong in the other direction — the HIDDEN
    plateau's structure appears nowhere, so a plateau uniformly different
    from the representative folded invisibly (see
    ``test_fold_honesty_uniform_plateau_dot_differs``). The run must split
    so the plateau folds from its own structurally-matching representative.
    """

    representative_differs = _trace(nn.Sequential(_ConvBlock(2), _ConvBlock(1), _ConvBlock(1)))
    assert not _run_fold_members_uniform(representative_differs, ("0", "1", "2"))


def test_run_fold_rejects_plateau_unlike_representative() -> None:
    """A uniform Tanh plateau cannot fold behind a ReLU representative.

    T9 (grind-p3, HIGH) red pin: the hidden members agree with each other,
    so the old hidden-only comparison accepted the fold; only comparing the
    representative too tells the run apart.
    """

    trace = _trace(
        nn.Sequential(
            _ConvBlock(1),
            _ConvBlock(1, activation=nn.Tanh),
            _ConvBlock(1, activation=nn.Tanh),
        )
    )
    assert not _run_fold_members_uniform(trace, ("0", "1", "2"))


class _ResidualBlock(nn.Module):
    """Linear+activation residual block for run-fold ellipsis renders."""

    def __init__(self, width: int = 8, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.lin(x))


class _ResidualStack(nn.Module):
    """Repeated residual blocks that auto-collapse folds into one ellipsis."""

    def __init__(self, activations: list[type[nn.Module]]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ResidualBlock(activation=act) for act in activations])
        self.out = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def test_fold_honesty_uniform_plateau_dot_differs(tmp_path) -> None:
    """A uniformly-changed hidden plateau never renders byte-identical DOT.

    T9 (grind-p3, HIGH) red pin: with the hidden-only comparison, a model
    whose blocks 1..7 all swap ReLU for Tanh folded to "+7 more" behind the
    unchanged ReLU representative and rendered the SAME bytes as the
    all-ReLU model. Post-fix the changed model splits (representative box
    plus a plateau fold of its own), so the sources differ.
    """

    torch.manual_seed(0)
    sources: list[str] = []
    for variant in ("plain", "tanh_plateau"):
        activations: list[type[nn.Module]] = [nn.ReLU] * 8
        if variant == "tanh_plateau":
            activations = [nn.ReLU] + [nn.Tanh] * 7
        trace = tl.trace(_ResidualStack(activations), torch.randn(2, 8))
        outpath = tmp_path / f"plateau_{variant}"
        trace.draw(
            collapse="auto",
            fold_repeats=True,
            vis_save_only=True,
            vis_fileformat="dot",
            vis_outpath=str(outpath),
        )
        sources.append((tmp_path / f"plateau_{variant}.dot").read_text())
    assert sources[0] != sources[1], (
        "a uniformly-changed hidden plateau rendered byte-identical DOT; the "
        "fold is claiming sameness with a representative it never checked"
    )


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
