"""Tuple facets follow call-exit paths, never chronological output positions."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.container import TupleIndex
from torchlens.semantic.facets import AbsenceReason, FacetSpec
from torchlens.semantic.recipes._helpers import module_output_spec


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
    spec = module_output_spec(module, "test", tuple_index=0)
    assert isinstance(spec, FacetSpec)
    torch.testing.assert_close(spec.read(), torch.full((2,), 6.0))
    assert spec.home_label == module.calls[0].output_ops[1]
    assert isinstance(module_output_spec(module, "test"), AbsenceReason)


@pytest.mark.parametrize(
    "corruption", ["absent", "duplicate", "incomplete", "wrong_path", "foreign"]
)
def test_tuple_facet_refuses_unproven_paths(corruption: str) -> None:
    """Missing or incoherent evidence cannot choose any available tensor instead."""

    trace = tl.trace(nn.Sequential(_ReorderedPair()), torch.ones(2))
    module = trace.modules["0"]
    events = trace.event_stream
    assert events is not None
    event = next(event for event in events.module_exit_events if event.call_label == "0:1")
    events.module_exit_events[:] = [event]
    if corruption == "absent":
        events.module_exit_events.clear()
    elif corruption == "duplicate":
        events.module_exit_events.append(event)
    elif corruption == "incomplete":
        events.module_exit_events[0] = replace(event, output_paths=())
    elif corruption == "wrong_path":
        events.module_exit_events[0] = replace(
            event, output_paths=((TupleIndex(2),), (TupleIndex(3),))
        )
    else:
        trace._raw_to_final_op_labels[event.output_tensor_labels_raw[0]] = trace.input_ops[0].label
    assert isinstance(module_output_spec(module, "test", tuple_index=0), AbsenceReason)


def test_tuple_facet_does_not_read_an_unsaved_primary() -> None:
    """A retained auxiliary output cannot substitute for an unsaved primary."""

    trace = tl.trace(nn.Sequential(_ReorderedPair()), torch.ones(2), save=tl.func("add"))
    assert isinstance(module_output_spec(trace.modules["0"], "test", tuple_index=0), AbsenceReason)
