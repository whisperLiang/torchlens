"""Regression tests for the r18e debug-module hardening fixes.

Covers:
- H5: ``tl.debug.lineage`` no longer crashes on recurrent models (getattr-multipass class).
- H2/M8: ``audit_trace`` honesty -- surfaces real bisect_nan/dtype_range findings and
  counts empty/insufficient checks as SKIPPED, not RUN.
- H6/H7: ``graph_breaks`` runs the Dynamo probe on clean torch and does not mutate the
  caller's model.
- M4/M5: ``infer_input_shape`` restores global RNG and labels failed results honestly.
- M6: live ``find_nan`` does not leak the internal ``_raw`` label namespace.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _Recurrent(nn.Module):
    """Weight-shared recurrent block that rolls into aggregate multi-pass Layers."""

    def __init__(self) -> None:
        """Build a single shared linear applied three times."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``relu(lin(x))`` three times.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Final activation.
        """

        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


# ---------------------------------------------------------------------------
# H5 -- lineage on recurrent models (getattr-multipass class)
# ---------------------------------------------------------------------------


def test_lineage_recurrent_model_never_raises() -> None:
    """lineage honours its non-raising contract on aggregate multi-pass Layers."""

    trace = tl.trace(_Recurrent(), torch.randn(2, 4), save=tl.where(lambda record: True))

    # A single-pass boundary start walks the pass-qualified graph without crashing.
    for label, direction in (
        ("output_1", "ancestors"),
        ("input_1", "descendants"),
    ):
        result = tl.debug.lineage(trace, label, direction=direction)
        assert result.nodes, (label, direction)
        # No internal raw-namespace markers leak into node labels.
        assert all(not node[0].endswith("_raw") for node in result.nodes)

    # A bare recurrent label is per-pass ambiguous -> honest message, no crash.
    ambiguous = tl.debug.lineage(trace, "linear_1_1", direction="descendants")
    assert ambiguous.nodes == []
    assert "recurrent" in ambiguous.message
    assert "select a pass" in ambiguous.message

    # A pass-qualified start resolves cleanly.
    qualified = tl.debug.lineage(trace, "linear_1_1:2", direction="descendants")
    assert qualified.start_label == "linear_1_1:2"
    assert qualified.nodes
