"""Resource and ordering contracts for complete shared-parameter pass groups."""

from __future__ import annotations

import torch
from example_models import RandomGraphModel

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.postprocess.finalization import _build_layer_logs
from torchlens.validation.invariants import check_metadata_invariants


def test_shared_pass_aggregation_has_linear_ordering_work() -> None:
    """Building pass accessors must not repeatedly sort every earlier pass."""

    class CountedPass(int):
        """An integer pass index that accounts for ordering comparisons."""

        comparisons = 0

        def __lt__(self, other: int) -> bool:
            """Count ordering work without changing integer comparison semantics."""
            type(self).comparisons += 1
            return super().__lt__(other)

    model = RandomGraphModel(target_nodes=1000, seed=42, share_blocks=True, functional_blocks=True)
    with torch.no_grad():
        trace = tl.trace(
            model, torch.randn(2, 64), capture=CaptureOptions(recurrence_detection=False)
        )
    try:
        for op in trace.layer_list:
            op._internal_set("pass_index", CountedPass(op.pass_index))
        _build_layer_logs(trace)
        assert CountedPass.comparisons < 4 * len(trace.layer_list), CountedPass.comparisons
        for layer in trace.layers:
            expected = sorted(layer.ops.values(), key=lambda op: op.pass_index)
            assert [layer.ops[index] for index in range(len(expected))] == expected
            assert all(layer.ops.get(op.pass_index) is op for op in expected)
        assert check_metadata_invariants(trace)
    finally:
        trace.cleanup()
