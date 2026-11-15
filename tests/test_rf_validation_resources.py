"""Geometry validation keeps its full verdict without retaining explanation histories."""

from __future__ import annotations

import gc
import tracemalloc
import weakref
from dataclasses import replace

import pytest
import torch
from example_models import RandomGraphModel
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._errors import ReceptiveFieldError
from torchlens.receptive_field._types import ReceptiveFieldStatus
from torchlens.receptive_field._validation import check_geometric_metadata_invariants

pytestmark = pytest.mark.heavy


def test_geometry_validation_has_linear_temporary_storage() -> None:
    """A complete graph check must not retain every prefix of diagnostic notes."""
    with torch.no_grad():
        warm = tl.trace(nn.ReLU(), torch.randn(2, 4))
        check_geometric_metadata_invariants(warm)
        warm.cleanup()
        trace = tl.trace(RandomGraphModel(target_nodes=1200, seed=42), torch.randn(2, 64))
    count = len(trace.layer_list)
    gc.collect()
    already_tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    tracemalloc.reset_peak()
    before, _ = tracemalloc.get_traced_memory()
    try:
        assert check_geometric_metadata_invariants(trace)
        _, peak = tracemalloc.get_traced_memory()
        assert peak - before < count * 8192, (count, peak - before)
    finally:
        if not already_tracing:
            tracemalloc.stop()
        trace.cleanup()


@pytest.mark.parametrize("direction", ["receptive", "projective"])
def test_geometry_validation_preserves_geometry_notes_and_cached_tripwires(direction: str) -> None:
    """Transient checks match detailed geometry and still reject cached corruption."""
    with torch.no_grad():
        trace = tl.trace(
            nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)), torch.randn(2, 4)
        )
    try:
        if direction == "receptive":
            compact = _engine.solve(trace, metadata_only=True)
            detailed = _engine.solve(trace)
            assert _engine.solve(trace) is detailed
        else:
            compact = solve_projective(trace, trace.output_ops, metadata_only=True)
            detailed = solve_projective(trace, trace.output_ops)
            assert solve_projective(trace, trace.output_ops) is detailed
        assert compact is not detailed
        assert compact.descriptors.keys() == detailed.descriptors.keys()
        assert any(descriptor.notes for descriptor in detailed.descriptors.values())
        for key, descriptor in detailed.descriptors.items():
            assert replace(compact.descriptors[key], notes=()) == replace(descriptor, notes=())
        descriptor = next(row for row in detailed.descriptors.values() if row.axes is not None)
        status = descriptor.status
        object.__setattr__(descriptor, "status", ReceptiveFieldStatus.DATA_DEPENDENT)
        try:
            with pytest.raises(ReceptiveFieldError, match="status monotonicity"):
                check_geometric_metadata_invariants(trace)
        finally:
            object.__setattr__(descriptor, "status", status)
    finally:
        trace.cleanup()


def test_cleanup_releases_both_directional_geometry_caches() -> None:
    """Keeping a husked Trace must not retain source/target geometry solutions."""
    with torch.no_grad():
        trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    source = _engine.solve_from(trace, trace.input_ops[0])
    target = solve_projective(trace, trace.output_ops)
    source_ref, target_ref = weakref.ref(source), weakref.ref(target)
    del source, target
    trace.cleanup()
    gc.collect()
    assert source_ref() is None
    assert target_ref() is None
