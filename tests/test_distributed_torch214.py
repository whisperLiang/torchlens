"""Torch 2.14 collective API census: capture renamed sites, refuse new geometry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.distributed import UncapturedCollectiveOpError, _lifecycle as lifecycle

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
        reason="torch.distributed gloo unavailable",
    ),
]


@pytest.fixture()
def gloo_world(tmp_path: Path) -> Iterator[Any]:
    """Provide a clean single-rank gloo world and restore distributed state."""

    dist = torch.distributed
    lifecycle.disarm()
    if dist.is_initialized():
        dist.destroy_process_group()
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield dist
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    ("attr", "kind", "op_type"),
    [
        ("all_gather_single", "all_gather_into_tensor", "allgatherintotensor"),
        ("reduce_scatter_single", "reduce_scatter_tensor", "reducescattertensor"),
    ],
)
def test_renamed_collective_has_one_boundary_and_matches_bare_execution(
    gloo_world: Any, attr: str, kind: str, op_type: str
) -> None:
    """The new canonical spelling retains output, provenance, and one issue tick."""

    if not hasattr(gloo_world, attr):
        pytest.skip(f"this torch has no {attr} collective")

    class Collective(nn.Module):
        """Issue one renamed collective between ordinary tensor operations."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a consumer of the collective's destination tensor."""

            contribution = x * 2
            output = torch.empty_like(contribution)
            getattr(gloo_world, attr)(output, contribution)
            return output + 1

    x = torch.arange(4, dtype=torch.float32)
    model = Collective()
    expected = model(x)
    lifecycle.arm()
    log = tl.trace(model, x)
    boundaries = [op for op in log.ops if op.type == op_type]
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert any(parent.startswith("mul_") for parent in boundary.parents)
    assert any(boundary.layer_label in op.parents for op in log.ops if op.type == "add")
    info = boundary.annotations["collective"]
    assert info["kind"] == kind
    assert info["correlation"]["seq"] == 0
    assert [role["role"] for role in info["roles"]] == ["contribution", "destination"]
    assert len(log.annotations["distributed"]["boundaries"]) == 1
    assert torch.equal(log.output_ops[0].out, expected)


@pytest.mark.parametrize("attr", ["gather_single", "gather_into_tensor"])
@pytest.mark.parametrize("capture", [tl.trace, tl.record], ids=["trace", "record"])
def test_single_tensor_gather_refuses_before_destination_mutation(
    gloo_world: Any, attr: str, capture: Any
) -> None:
    """New root-buffer geometry never silently disappears from either capture path."""

    if not hasattr(gloo_world, attr):
        pytest.skip(f"this torch has no {attr} collective")
    destination = torch.full((4,), -37.0)

    class Gather(nn.Module):
        """Issue an unsupported collective against a visible sentinel buffer."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Attempt the new single-buffer gather."""

            getattr(gloo_world, attr)(x, destination, dst=0)
            return destination

    lifecycle.arm()
    with pytest.raises(UncapturedCollectiveOpError) as excinfo:
        capture(Gather(), torch.arange(4, dtype=torch.float32), save=tl.func("add"))
    assert excinfo.value.fields["kind"] == "uncaptured_collective_op"
    assert excinfo.value.fields["layer"] == 3
    assert excinfo.value.fields["reason"] == "single_tensor_gather_unsupported"
    assert excinfo.value.fields["func"] == f"torch.distributed.{attr}"
    assert torch.equal(destination, torch.full((4,), -37.0))
