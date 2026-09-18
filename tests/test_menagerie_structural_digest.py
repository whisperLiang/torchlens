"""Tests for menagerie structural digest entry points."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from menagerie.catalog import CatalogRow
from menagerie.recipe import build_model_and_input
from menagerie.structural_digest import (
    architecture_distinctness_hash,
    structural_fingerprint,
)
from support.menagerie_catalog import menagerie_rows as menagerie_rows


# Display ordinals move whenever the catalog grows: the old sample ordinal 182
# now names a billion-parameter AIMv2. Pin bounded, real CPU recipes by natural
# identity and retain convolution, recurrence, FFT, recursion and indexing coverage.
SAMPLE_CLASSIC_NAMES = (
    "LeNet-4 / pre-LeNet-5 CNN",
    "Clockwork RNN",
    "Compact Bilinear Pooling",
    "Williams-Zipser fully-recurrent net",
    "Original LSTM (1997, no forget gate)",
    "RAAM (Recursive Auto-Associative Memory)",
    "Pi-Sigma network",
    "Sigma-Pi / higher-order unit",
    "CMAC (Albus)",
    "Classic Adaptive Mixture-of-Experts (dense)",
)


class WidthOnlyModel(nn.Module):
    """Single-layer model whose only architectural difference is output width."""

    def __init__(self, output_width: int) -> None:
        """Initialize the width-only model.

        Parameters
        ----------
        output_width:
            Output feature width for the linear layer.
        """

        super().__init__()
        self.proj = nn.Linear(4, output_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the projection.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Projected tensor.
        """

        return self.proj(x)


def _build_sample(row: CatalogRow) -> tuple[Any, Any]:
    """Build a deterministic model/input sample for one menagerie row.

    Parameters
    ----------
    row:
        Menagerie catalog row.

    Returns
    -------
    tuple[Any, Any]
        Model and example input.
    """

    torch.manual_seed(0)
    return build_model_and_input(row)


# The full-catalog fixture costs 7-9 s in the unified environment, independent
# of which sample receives its first setup. Keep every recipe in the full backstop.
@pytest.mark.heavy
@pytest.mark.parametrize("name", SAMPLE_CLASSIC_NAMES)
def test_structural_fingerprint_is_deterministic_for_menagerie_sample(
    name: str, menagerie_rows: tuple[CatalogRow, ...]
) -> None:
    """Structural fingerprints are deterministic across repeated calls.

    Parameters
    ----------
    name:
        Canonical classic name to build and trace.
    menagerie_rows:
        Current catalog loaded from a session-private SQLite artifact.
    """

    matches = [row for row in menagerie_rows if row.name == name and row.source == "classics"]
    assert len(matches) == 1
    row = matches[0]
    model, example_input = _build_sample(row)

    first = structural_fingerprint(model, example_input)
    second = structural_fingerprint(model, example_input)

    assert first == second
    assert len(first) == 64


def test_structural_fingerprint_distinguishes_layer_width_change() -> None:
    """Shape-aware structural fingerprints catch layer-width corruption."""

    input_value = torch.randn(2, 4)
    width_8 = structural_fingerprint(WidthOnlyModel(output_width=8), input_value)
    width_16 = structural_fingerprint(WidthOnlyModel(output_width=16), input_value)

    assert width_8 != width_16


def test_architecture_distinctness_hash_matches_for_layer_width_change() -> None:
    """Shape-blind distinctness hash groups same-topology width variants."""

    input_value = torch.randn(2, 4)
    width_8 = architecture_distinctness_hash(WidthOnlyModel(output_width=8), input_value)
    width_16 = architecture_distinctness_hash(WidthOnlyModel(output_width=16), input_value)

    assert width_8 == width_16
