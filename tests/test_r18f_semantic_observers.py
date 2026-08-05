"""R18F regression tests: residual recipe, patching state safety, observer labels/spans.

Covers:
- H3 residual recipe only fires on genuine attn+mlp transformer blocks; resid_mid is
  the real post-attention add, never the degenerate first-add-equals-output fallback.
- H8 activation/attribution patching snapshots and restores model state + global RNG
  around every counterfactual run, so the caller's model is not mutated and each
  counterfactual starts from identical state.
- M7 forward TapRecord.site_label resolves to the PUBLIC trace label, not the raw
  internal ``*_raw`` label.
- M9 (obs-local) observer span direction is enforced: a forward record only carries
  forward/both spans; a backward record only carries backward/both spans.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


# --------------------------------------------------------------------------------------
# H3 - residual recipe structural gate
# --------------------------------------------------------------------------------------


class _ScaleBlock(nn.Module):
    """A ``*Block*``-named module that is NOT a transformer block (no attn/mlp)."""

    def __init__(self) -> None:
        """Initialize a single scaling child."""

        super().__init__()
        self.scale = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a plain residual scaling update, not an attention/MLP block."""

        return x + self.scale(x)


class _GenuineBlock(nn.Module):
    """A genuine transformer block with attention and MLP residual updates."""

    def __init__(self) -> None:
        """Initialize attention and MLP children."""

        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention and MLP residual updates."""

        resid_mid = x + self.attn(x)
        return resid_mid + self.mlp(resid_mid)


def test_residual_recipe_rejects_non_transformer_named_block() -> None:
    """A ``*Block*``-named module without attn+mlp children exposes no residual facets."""

    torch.manual_seed(0)
    log = tl.trace(_ScaleBlock(), torch.randn(2, 4), layers_to_save="all")
    facets = log.modules["self"].facets

    assert not facets.has("resid_pre")
    assert not facets.has("resid_mid")
    assert not facets.has("resid_post")


def test_residual_recipe_marks_genuine_transformer_block() -> None:
    """A genuine attn+mlp block exposes resid_pre/mid/post; mid is not degenerate."""

    torch.manual_seed(0)
    log = tl.trace(_GenuineBlock(), torch.randn(2, 4), layers_to_save="all")
    facets = log.modules["self"].facets

    assert facets.has("resid_pre")
    assert facets.has("resid_mid")
    assert facets.has("resid_post")
    # resid_mid is the post-attention add, not the first-add fallback that collapses
    # onto the block output (resid_post).
    assert not torch.equal(facets.resid_mid.value, facets.resid_post.value)
