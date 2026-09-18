"""Non-tensor graph boundaries retain uncertainty without inventing scalar grids."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._errors import (
    ReceptiveFieldError,
    ReceptiveFieldUnavailableError,
)
from torchlens.receptive_field._types import ReceptiveFieldStatus
from torchlens.receptive_field._validation import check_geometric_metadata_invariants

pytestmark = pytest.mark.smoke


class _TwoStage(nn.Module):
    """Exercise pointwise and whole-extent propagation after a missing grid."""

    def __init__(self, reduction: bool = False) -> None:
        """Choose whether the second stage reduces all input axes."""

        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a pointwise or reduced result after the first ReLU."""

        intermediate = torch.relu(x)
        return intermediate.sum() if self.reduction else torch.relu(intermediate)


@pytest.mark.parametrize("reduction", [False, True])
def test_missing_grid_taints_both_directions_without_dropping_paths(reduction: bool) -> None:
    """Keep tensor-to-tensor dependencies uncertain across a gridless boundary."""

    trace = tl.trace(_TwoStage(reduction), torch.ones((2, 3)))
    boundary = next(op for op in trace.layer_list if op.func_name == "relu")
    original_shape = boundary.shape
    boundary.shape = None

    receptive = _engine.solve(trace)
    projective = solve_projective(trace, trace.output_ops)

    assert boundary.label not in receptive.per_op
    assert boundary.label not in projective.per_op
    for solution, endpoint in (
        (receptive, trace.output_ops[0]),
        (projective, trace.input_ops[0]),
    ):
        descriptors = tuple(solution.per_op[endpoint.label].values())
        assert descriptors, "Dropping the boundary must not erase the influence path."
        assert all(item.status is ReceptiveFieldStatus.UNKNOWN for item in descriptors)
        assert all(item.axes is None for item in descriptors)
        assert all(
            any("no tensor output grid" in note for note in item.notes) for item in descriptors
        )
    assert check_geometric_metadata_invariants(trace)

    boundary.shape = original_shape
    assert _engine.solve(trace) is not receptive
    assert solve_projective(trace, trace.output_ops) is not projective
    assert check_geometric_metadata_invariants(trace)


def test_missing_grid_is_not_the_scalar_grid_in_cache_identity() -> None:
    """A boundary becoming a true scalar must invalidate both cached solutions."""

    trace = tl.trace(_TwoStage(), torch.tensor(1.0))
    boundary = next(op for op in trace.layer_list if op.func_name == "relu")
    assert boundary.shape == ()
    scalar_revision = _engine._graph_revision(trace)
    scalar_solution = _engine.solve(trace)
    scalar_projective = solve_projective(trace, trace.output_ops)
    assert boundary.label in scalar_solution.per_op

    boundary.shape = None
    assert _engine._graph_revision(trace) != scalar_revision
    assert _engine.solve(trace) is not scalar_solution
    assert solve_projective(trace, trace.output_ops) is not scalar_projective


@pytest.mark.parametrize("direction", ["receptive", "projective"])
def test_gridless_explicit_endpoint_refuses_typed(direction: str) -> None:
    """An explicit boundary endpoint is unavailable, not an empty scalar grid."""

    trace = tl.trace(_TwoStage(), torch.ones((2, 3)))
    boundary = next(op for op in trace.layer_list if op.func_name == "relu")
    boundary.shape = None

    with pytest.raises(ReceptiveFieldUnavailableError, match="no tensor output grid"):
        if direction == "receptive":
            _engine.solve_from(trace, boundary)
        else:
            solve_projective(trace, [boundary])


def test_gridless_boundary_does_not_disable_tensor_metadata_tripwire() -> None:
    """The ordinary tensor descriptor invariants still reject a forged exact claim."""

    trace = tl.trace(_TwoStage(), torch.ones((2, 3)))
    boundary = next(op for op in trace.layer_list if op.func_name == "relu")
    boundary.shape = None
    solution = _engine.solve(trace)
    descriptor = next(iter(solution.per_op[trace.output_ops[0].label].values()))
    object.__setattr__(descriptor, "status", ReceptiveFieldStatus.EXACT)

    with pytest.raises(ReceptiveFieldError, match="taint propagation failed"):
        check_geometric_metadata_invariants(trace)


@pytest.mark.backend_jax
@pytest.mark.parametrize("kind", ["while", "custom_vjp"])
def test_jax_nontensor_region_boundaries_pass_geometric_metadata(kind: str) -> None:
    """Real JAX region/control nodes have no grid and never publish scalar geometry."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    @jax.custom_vjp
    def square(x: Any) -> Any:
        """Define an opaque custom-VJP region."""

        return x * x

    def forward(x: Any) -> tuple[Any, Any]:
        """Return the square and its residual."""

        return x * x, x

    def backward(residual: Any, grad: Any) -> tuple[Any]:
        """Return the square's custom gradient."""

        return (2 * residual * grad,)

    square.defvjp(forward, backward)

    def model(params: dict[str, Any], x: Any) -> Any:
        """Capture the chosen boundary with a small tensor input and output."""

        del params
        if kind == "custom_vjp":
            return square(x)
        return jax.lax.while_loop(
            lambda state: state[0] < 2,
            lambda state: (state[0] + 1, state[1] + 1),
            (0, x),
        )[1]

    trace = tl.trace(model, ({}, jnp.ones((2, 3))), backend="jax")
    boundaries = [op for op in trace.layer_list if op.shape is None]
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.parents and boundary.children
    assert check_geometric_metadata_invariants(trace)
    for solution in (_engine.solve(trace), solve_projective(trace, trace.output_ops)):
        assert boundary.label not in solution.per_op
        assert all(label != boundary.label for label, _ in solution.descriptors)
    if kind == "custom_vjp":
        assert trace.validate_forward_pass([]).state == "unverified"
    else:
        assert trace.validate_forward_pass([]) is True
