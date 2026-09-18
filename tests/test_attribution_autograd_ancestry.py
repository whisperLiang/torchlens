"""Autograd ancestry remains complete across transient Python node wrappers."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor

from torchlens.attribution._layer import _autograd_leaf_variable_ids

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("depth", [1, 4, 16])
def test_leaf_ancestry_keeps_every_branch_in_deep_graphs(depth: int) -> None:
    """Report structural ancestors, including zero-derivative and shared branches."""

    first = torch.ones(2, requires_grad=True)
    second = torch.ones(2, requires_grad=True)
    decoy = torch.ones(2, requires_grad=True)
    value = first * 0.0 + second
    for _ in range(depth):
        value = value.sin() + value.cos()

    reachable = _autograd_leaf_variable_ids((value,))

    assert id(first) in reachable
    assert id(second) in reachable
    assert id(decoy) not in reachable
    assert id(value) in reachable
    gradients = torch.autograd.grad(value.sum(), (first, second))
    torch.testing.assert_close(gradients[0], torch.zeros_like(first))


def test_ancestry_retains_wrappers_only_until_walk_finishes() -> None:
    """Prevent id reuse during traversal without retaining wrappers afterward."""

    leaf = torch.ones(1, requires_grad=True)
    references: list[weakref.ReferenceType[EphemeralNode]] = []

    class EphemeralNode:
        """Expose a new Python wrapper for each ancestor, as autograd can do."""

        def __init__(self, depth: int) -> None:
            """Record a weak lifetime witness without retaining this wrapper."""

            self.depth = depth
            self.variable = leaf if depth == 0 else None
            references.append(weakref.ref(self))

        @property
        def next_functions(self) -> tuple[tuple[EphemeralNode, int], ...]:
            """Assert prior wrappers remain alive while their ids are in use."""

            assert all(reference() is not None for reference in references)
            return ((EphemeralNode(self.depth - 1), 0),) if self.depth else ()

    activation = cast(Tensor, SimpleNamespace(grad_fn=EphemeralNode(4)))

    assert _autograd_leaf_variable_ids((activation,)) == {id(activation), id(leaf)}
    assert len(references) == 5
    assert references[0]() is activation.grad_fn
    assert all(reference() is None for reference in references[1:])
