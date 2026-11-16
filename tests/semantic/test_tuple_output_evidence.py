"""Tuple facets follow captured container paths, never chronological output positions."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._ir.container import TupleIndex
from torchlens._semantic.facets import AbsenceReason, FacetSpec
from torchlens._semantic.recipes._helpers import module_output_spec


class _ReorderedPair(nn.Module):
    """Return tensors in the opposite order from their computation."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the auxiliary result before the primary result."""

        auxiliary = x + 1
        primary = auxiliary * 3
        return primary, auxiliary


def test_tuple_facet_uses_exit_evidence_not_output_inventory_order() -> None:
    """An index-zero facet follows the return path despite chronological ordering."""

    trace = tl.trace(nn.Sequential(_ReorderedPair()), torch.ones(2))
    module = trace.modules["0"]
    spec = module_output_spec(module, "test")
    assert isinstance(spec, FacetSpec)
    torch.testing.assert_close(spec.read(), torch.full((2,), 6.0))
    assert spec.home_label == module.calls[0].output_ops[1]


@pytest.mark.parametrize(
    "corruption", ["absent", "duplicate", "incomplete", "wrong_path", "foreign"]
)
def test_tuple_facet_refuses_unproven_paths(corruption: str) -> None:
    """Missing or incoherent path and role evidence cannot choose any available tensor instead."""

    trace = tl.trace(nn.Sequential(_ReorderedPair()), torch.ones(2))
    module = trace.modules["0"]
    call = module.calls[0]
    primary = next(label for label in call.output_ops if trace.ops[label].multi_output_name == "0")
    if corruption == "absent":
        call.output_paths = ()
    elif corruption == "duplicate":
        for label in call.output_ops:
            trace.ops[label].multi_output_name = "0"
    elif corruption == "incomplete":
        call.output_paths = ((),)
    elif corruption == "wrong_path":
        call.output_paths = ((TupleIndex(2),), (TupleIndex(3),))
    else:
        call.output_ops = [
            trace.input_ops[0].label if label == primary else label for label in call.output_ops
        ]
    assert isinstance(module_output_spec(module, "test"), AbsenceReason)


def test_tuple_facet_does_not_read_an_unsaved_primary() -> None:
    """A retained auxiliary output cannot substitute for an unsaved primary."""

    trace = tl.trace(nn.Sequential(_ReorderedPair()), torch.ones(2), save=tl.func("add"))
    assert isinstance(module_output_spec(trace.modules["0"], "test"), AbsenceReason)
