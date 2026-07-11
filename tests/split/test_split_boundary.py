"""ReplayBoundary ABI tests."""

from __future__ import annotations

import pytest
import torch

from torchlens.split.boundary import ReplayBoundary
from torchlens.split.errors import SplitBoundaryError
from torchlens.split.shape import SymbolicShape
from torchlens.split import BoundarySchema


def _spec(label: str = "relu_1_1") -> dict[str, BoundarySchema]:
    """Return a one-tensor boundary spec."""

    return {
        "h": BoundarySchema(
            value_id="h",
            container_path=(),
            role="primary",
            alias_group=None,
            source_kind="boundary",
            label=label,
            backend="torch",
            module_path="relu",
            op_type="relu",
            shape=SymbolicShape(("B", 3)),
            dtype="torch.float32",
            requires_grad=False,
        )
    }


def _boundary(
    tensor: torch.Tensor | None = None,
    *,
    split_id: str = "s1",
    spec: dict[str, BoundarySchema] | None = None,
) -> ReplayBoundary:
    """Return a simple replay boundary."""

    return ReplayBoundary(
        backend="torch",
        tensors={"h": torch.ones(2, 3) if tensor is None else tensor},
        spec=_spec() if spec is None else spec,
        metadata={"split_id": split_id, "batch_symbol": "B", "dynamic_batch": (1, 4)},
    )


def test_boundary_validate_and_transforms() -> None:
    """Boundary methods should be adapter-delegated and immutable."""

    boundary = _boundary()

    boundary.validate(split_id="s1")
    detached = boundary.detach()
    cloned = boundary.clone()
    cpu = boundary.cpu()

    assert detached is not boundary
    assert cloned.tensors["h"] is not boundary.tensors["h"]
    assert cpu.tensors["h"].device.type == "cpu"


def test_boundary_collate() -> None:
    """Collation stacks matching boundary tensors."""

    collated = ReplayBoundary.collate([_boundary(torch.ones(3)), _boundary(torch.zeros(3))])

    assert collated.tensors["h"].shape == (2, 3)
    assert collated.metadata["collated"] is True


def test_boundary_collate_rejects_mismatched_abi() -> None:
    """Collation rejects boundaries from different split ABIs."""

    with pytest.raises(SplitBoundaryError):
        ReplayBoundary.collate([_boundary(), _boundary(split_id="s2")])
    with pytest.raises(SplitBoundaryError):
        ReplayBoundary.collate([_boundary(), _boundary(spec=_spec(label="other"))])


def test_boundary_validation_mismatch_errors() -> None:
    """Boundary ABI mismatches raise structured errors."""

    with pytest.raises(SplitBoundaryError):
        _boundary(torch.ones(5, 3)).validate(split_id="s1")
    with pytest.raises(SplitBoundaryError):
        _boundary().validate(split_id="other")
    bad = ReplayBoundary(
        backend="torch",
        tensors={"other": torch.ones(2, 3)},
        spec=_spec(),
        metadata={"split_id": "s1"},
    )
    with pytest.raises(SplitBoundaryError):
        bad.validate(split_id="s1")
