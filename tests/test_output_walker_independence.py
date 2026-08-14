"""Shared-root plant for ground-truth output enumeration (b9 R74/75-1).

Sol's live probe: a planted drop-the-last-leaf mutation in the ONE backends
output walker made capture AND the validation oracle omit output #2 of a
2-tuple model, and ``validate_forward_pass`` returned True -- a dropped
capture output survived the tripwire because both sides resolved through the
same callable. The fix is the validation-owned independent traversal
(``torchlens.validation._output_walk``) cross-checked against the adapter's
enumeration inside ``validate_forward_pass``.

This file is the handoff contract: the plant re-applies sol's exact
mutation shape and validation must now FAIL; the structural test keeps the
two roots from ever re-merging.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.backends.torch.ops as torch_ops
from torchlens.validation import _output_walk

pytestmark = pytest.mark.smoke


class _TupleOut(nn.Module):
    """Model returning a 2-tuple, the shape of sol's original plant."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two derived tensors.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two independently derived leaves.
        """

        y = torch.relu(x)
        return y, y + 1


def test_validation_survives_pristine_tuple_output():
    """Control: the 2-tuple model validates before any plant."""

    assert tl.validate_forward_pass(_TupleOut(), [torch.randn(3)], input_kwargs={})


def test_planted_walker_leaf_drop_fails_validation(monkeypatch: pytest.MonkeyPatch):
    """Sol's shared-root plant now goes RED instead of validating True.

    The plant wraps the real backends walker and drops the LAST yielded
    leaf -- the exact mutation that previously made both capture and the
    oracle agree on the truncated enumeration.
    """

    real_walker = torch_ops._walk_output_tensors_with_paths

    def _dropping_walker(output):
        """Yield the real walk minus its final leaf.

        Parameters
        ----------
        output:
            Model output tree.

        Returns
        -------
        list
            Truncated walk results.
        """

        rows = list(real_walker(output))
        return rows[:-1]

    monkeypatch.setattr(torch_ops, "_walk_output_tensors_with_paths", _dropping_walker)
    with pytest.warns(RuntimeWarning, match="output-enumeration defect"):
        result = tl.validate_forward_pass(_TupleOut(), [torch.randn(3)], input_kwargs={})
    assert result is False, (
        "a dropped output leaf in the shared walker still validated True: "
        "the independent cross-check is disarmed"
    )


def test_independent_walker_never_imports_the_capture_adapter():
    """The two enumeration roots must stay distinct callables and modules.

    If both sides ever resolve to the same callable or the independent
    module grows a ``torchlens.backends`` import, the R74/75-1 defect class
    (one defect silencing both sides) is structurally possible again.
    """

    assert _output_walk.independent_output_tensor_ids is not (
        torch_ops._walk_output_tensors_with_paths
    )
    tree = ast.parse(inspect.getsource(_output_walk))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "backends" not in name, (
                f"validation/_output_walk.py imports {name!r}: the independent "
                "traversal must never share the capture adapter's root"
            )
