"""Regression tests for r18a compat/_report.py detection-heuristic tightening.

Each detection row must anchor its structural marker on real module identity /
AST references / typed attributes, mirroring ``_is_quantized_module``. These tests
kill the false-positive and false-negative detection classes flagged in round 18
(A3-02, A3-03, A3-08, A3-09, A3-10, A3-11 + DDP/DeepSpeed siblings, A3-12, LOW-8).
"""

from __future__ import annotations

import torch
from torch import nn

from torchlens.compat import report


# ---------------------------------------------------------------------------
# A3-08 — HF Transformers detection must key on real transformers namespace,
# not on the mere presence of a ``.config`` attribute.
# ---------------------------------------------------------------------------


class OrdinaryConfiguredModel(nn.Module):
    """Plain module that happens to carry an application ``config``."""

    def __init__(self) -> None:
        """Attach a non-HF config object."""

        super().__init__()
        self.config = {"application": "not-huggingface"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


class TransformersNamespaceModel(nn.Module):
    """Model advertising a real ``transformers`` module namespace."""

    __module__ = "transformers.modeling_utils"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


def test_hf_row_ignores_plain_config_attribute() -> None:
    """A generic ``.config`` attribute must not label a module as HF Transformers."""

    row = report(OrdinaryConfiguredModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is False
    assert row.status == "pass"
    assert row.severity == "ok"


def test_hf_row_detects_real_transformers_namespace() -> None:
    """A class defined under the ``transformers`` namespace stays detected."""

    row = report(TransformersNamespaceModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is True
    assert row.status == "pass"
    assert row.severity == "info"


def test_hf_row_detects_transformers_subclass_by_mro() -> None:
    """A user subclass of a transformers-namespace base is detected via its MRO."""

    class _FakePreTrainedBase(nn.Module):
        __module__ = "transformers.modeling_utils"

    class UserModel(_FakePreTrainedBase):
        __module__ = "my_project.models"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return input unchanged."""

            return x

    row = report(UserModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is True
