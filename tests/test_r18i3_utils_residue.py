"""Regression tests for r18i3 utils/__init__.py residue findings M12/M13/M14.

Each finding is a diagnostic/state-hygiene bug that r18i-2 diagnosed but could
not reach from its options.py lease:

* M12 -- ``_log_ops_for_mode`` restored training modes via a recursive
  ``model.train(root_mode)`` that clobbered mixed child eval/train state, and it
  ran unconditionally even for ``mode="current"`` (which must not mutate the
  model at all).
* M13 -- ``synthetic_input`` skipped only VAR_POSITIONAL/VAR_KEYWORD params, so a
  keyword-only tensor parameter was appended to the positional return tuple and
  raised ``TypeError`` when splatted into ``forward``.
* M14 -- ``_probe_torch_capabilities`` hardcoded ``"PASS"`` even when the probed
  snapshot reported missing capabilities (a diagnostic-honesty lie).
"""

from __future__ import annotations

from typing import Annotated

import pytest
import torch
from torch import nn

import torchlens.utils as tl_utils


# --------------------------------------------------------------------------- #
# M12 -- mixed child training-mode preservation in _log_ops_for_mode/list_ops
# --------------------------------------------------------------------------- #
class _MixedModeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.lin(x))


def _mode_map(model: nn.Module) -> dict[str, bool]:
    return {name or "self": module.training for name, module in model.named_modules()}


def test_m12_current_mode_preserves_mixed_child_states() -> None:
    """mode="current" must leave every submodule's training flag exactly as found."""

    model = _MixedModeNet()
    model.train()
    model.bn.eval()  # frozen child under a training root -> MIXED state
    x = torch.randn(8, 4)

    before = _mode_map(model)
    assert before["bn"] is False and before["self"] is True  # precondition: mixed

    tl_utils.list_ops(model, x, mode="current")

    after = _mode_map(model)
    assert after == before, f"mixed child modes clobbered: {before} -> {after}"
    # The specific regression: the frozen BN child must stay in eval.
    assert model.bn.training is False


def test_m12_eval_and_train_modes_restore_mixed_states() -> None:
    """eval/train captures must restore each submodule's original flag, not the root's."""

    model = _MixedModeNet()
    model.train()
    model.bn.eval()
    x = torch.randn(8, 4)
    before = _mode_map(model)

    tl_utils.list_ops(model, x, mode="eval")
    assert _mode_map(model) == before, "eval capture failed to restore mixed states"

    tl_utils.list_ops(model, x, mode="train")
    assert _mode_map(model) == before, "train capture failed to restore mixed states"


def test_m12_both_mode_restores_mixed_states() -> None:
    model = _MixedModeNet()
    model.train()
    model.bn.eval()
    x = torch.randn(8, 4)
    before = _mode_map(model)

    result = tl_utils.list_ops(model, x, mode="both")
    assert set(result) == {"eval", "train"}
    assert _mode_map(model) == before, "both-mode capture failed to restore mixed states"


# --------------------------------------------------------------------------- #
# M13 -- synthetic_input keyword-only parameter handling
# --------------------------------------------------------------------------- #
class _RequiredKwOnly(nn.Module):
    def forward(
        self,
        x: Annotated[torch.Tensor, (2, 3)],
        *,
        y: Annotated[torch.Tensor, (2, 3)],
    ) -> torch.Tensor:
        return x + y


class _OptionalKwOnly(nn.Module):
    def forward(
        self,
        x: Annotated[torch.Tensor, (2, 3)],
        *,
        y: torch.Tensor = torch.zeros(2, 3),
    ) -> torch.Tensor:
        return x + y


class _AllPositional(nn.Module):
    def forward(
        self,
        x: Annotated[torch.Tensor, (2, 3)],
        z: Annotated[torch.Tensor, (2, 3)],
    ) -> torch.Tensor:
        return x + z


def test_m13_required_keyword_only_raises_clear_error() -> None:
    """A required keyword-only tensor param cannot ride in the positional tuple."""

    model = _RequiredKwOnly()
    with pytest.raises(ValueError, match=r"keyword-only forward parameter 'y'"):
        tl_utils.synthetic_input(model)


def test_m13_optional_keyword_only_is_omitted_and_forward_runs() -> None:
    """An optional keyword-only param is dropped; forward uses its default."""

    model = _OptionalKwOnly()
    si = tl_utils.synthetic_input(model)
    # Only the single positional param survives -> a bare tensor, not a tuple.
    assert isinstance(si, torch.Tensor)
    # The generated input must be usable positionally without a TypeError.
    model.forward(si)


def test_m13_positional_params_still_form_a_tuple() -> None:
    model = _AllPositional()
    si = tl_utils.synthetic_input(model)
    assert isinstance(si, tuple) and len(si) == 2
    model.forward(*si)
