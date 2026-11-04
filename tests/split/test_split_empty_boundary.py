"""Empty non-batch dimensions stay concrete across boundary cache round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from dag_helpers import call_ids, close_call_ids

from torchlens.split import prepare_graph
from torchlens.split.graph import _portable_dim
from torchlens.split.shape_program import DimExpr


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_empty_boundary_roundtrip(batch: int, tmp_path: Path) -> None:
    """A solved (B, 0) frontier remains loadable at captured and extrapolated batches."""

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Produce an empty feature axis, not an empty batch axis."""

            return torch.relu(x[:, :0])

    graph = prepare_graph(Model(), torch.ones(4, 4))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"relu"}), ops=frozenset({"relu"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    assert runtime.batch_validation["status"] == "passed"
    boundary = runtime.run_prefix(torch.ones(batch, 4))
    for schema in boundary.spec.values():
        assert schema.shape.as_tuple() == ("B", 0)
    runtime.save_boundary(boundary, tmp_path)
    loaded = runtime.load_boundary(tmp_path)
    assert runtime.run_suffix(loaded).shape == (batch, 0)


def test_portable_shape_folds_zero_times_batch_algebraically() -> None:
    """Projection must not turn a constant-zero expression into a positive B axis."""

    batch = DimExpr.symbol("B")
    zero_product = DimExpr("mul", args=(DimExpr.const(0), batch))
    assert _portable_dim(zero_product, "B") == 0
    assert _portable_dim(DimExpr("add", args=(zero_product, DimExpr.const(4))), "B") == 4
    assert _portable_dim(DimExpr("mul", args=(DimExpr.const(2), batch)), "B") == "B"
