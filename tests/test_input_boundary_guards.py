"""r-b4 R27-1: capture-entry input walkers are depth-bounded and cycle-guarded.

The four input-boundary walkers (``walk_input_boundary``, ``snapshot_input_boundary``,
``backends.default_specs._simple_leaves``, ``utils.arg_handling.copy_arg_tree``) share
ONE nesting ceiling (``INPUT_TREE_MAX_DEPTH``) and refuse deep or self-referential
input trees TYPED (``input_tree_depth_exceeded`` / ``input_tree_cycle``) instead of
dying in a raw ``RecursionError`` (probe: ~350 user levels crossed the interpreter
limit; a self-referential list crashed every walker without a cycle guard).

DAG-shaped (shared, acyclic) inputs remain fully walked: the cycle guard is
path-scoped, never global, because every occurrence of a shared container must be
witnessed under its own path.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._input_walk import (
    INPUT_TREE_MAX_DEPTH,
    raw_mapping_key_component,
    snapshot_input_boundary,
    walk_input_boundary,
)
from torchlens.backends.default_specs import _simple_leaves
from torchlens.utils.arg_handling import copy_arg_tree

pytestmark = pytest.mark.smoke


def _deep_list(depth: int, leaf: object) -> object:
    """Build one ``depth``-level nested list around ``leaf``."""

    value = leaf
    for _ in range(depth):
        value = [value]
    return value


def _cyclic_list() -> list[object]:
    """Build one self-referential list holding a tensor leaf."""

    value: list[object] = [torch.ones(1)]
    value.append(value)
    return value


def test_trace_refuses_overdeep_input_typed() -> None:
    """A too-deep input tree refuses typed at capture entry, never RecursionError."""

    deep = _deep_list(INPUT_TREE_MAX_DEPTH + 50, torch.ones(1))
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(nn.Identity(), deep)
    assert excinfo.value.fields["code"] == "input_tree_depth_exceeded"


def test_trace_refuses_cyclic_input_typed() -> None:
    """A self-referential input container refuses typed at capture entry."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(nn.Identity(), _cyclic_list())
    assert excinfo.value.fields["code"] == "input_tree_cycle"


def test_trace_recovers_and_accepts_shared_substructure_after_refusal() -> None:
    """State recovers after a refusal, and DAG-shaped inputs still trace."""

    with pytest.raises(InvalidArgumentError):
        tl.trace(nn.Identity(), _cyclic_list())
    shared = [torch.ones(1), torch.ones(1)]
    log = tl.trace(nn.Sequential(nn.Identity()), [shared, shared])
    assert len(log) > 0


def test_walk_input_boundary_depth_and_cycle_ceiling_the_subtree() -> None:
    """The normative walker ceilings over-deep and cyclic subtrees, never crashes.

    Post d95ca11f (input-walk union) the WALKER routes a depth/cycle violation
    to ``on_opaque_key_subtree`` and skips the subtree — the typed refusal
    guarantee lives at capture entry (``test_trace_refuses_*_typed`` above) and
    in the snapshot refusals ledger, not here.
    """

    deep = _deep_list(INPUT_TREE_MAX_DEPTH + 10, torch.ones(1))
    deep_opaque: list[tuple[Any, ...]] = []
    walk_input_boundary(
        deep,
        key_component=raw_mapping_key_component,
        on_opaque_key_subtree=lambda _child, path: deep_opaque.append(path),
    )
    assert len(deep_opaque) == 1, "over-deep subtree must ceiling exactly once"

    cyclic_opaque: list[tuple[Any, ...]] = []
    walk_input_boundary(
        _cyclic_list(),
        key_component=raw_mapping_key_component,
        on_opaque_key_subtree=lambda _child, path: cyclic_opaque.append(path),
    )
    assert len(cyclic_opaque) == 1, "cyclic subtree must ceiling exactly once"


def test_walk_input_boundary_walks_every_shared_occurrence() -> None:
    """The cycle guard is PATH-scoped: a shared container is walked per occurrence."""

    leaf = torch.ones(1)
    shared = [leaf]
    seen: list[tuple[object, ...]] = []
    walk_input_boundary(
        [shared, shared],
        key_component=raw_mapping_key_component,
        on_tensor=lambda _tensor, path: seen.append(path),
    )
    assert seen == [(0, 0), (1, 0)]


def test_snapshot_input_boundary_is_total_and_refuses_in_ledger() -> None:
    """The runnable structure snapshot stays TOTAL: violations join the refusals."""

    cyclic_snapshot = snapshot_input_boundary(_cyclic_list())
    assert "input_container_cycle" in {r["reason"] for r in cyclic_snapshot["refusals"]}

    deep_snapshot = snapshot_input_boundary(_deep_list(INPUT_TREE_MAX_DEPTH + 10, 1))
    assert "input_container_too_deep" in {r["reason"] for r in deep_snapshot["refusals"]}


def test_snapshot_input_boundary_clean_input_has_no_guard_refusals() -> None:
    """Ordinary nested inputs snapshot with zero guard refusals (no false refusal)."""

    snapshot = snapshot_input_boundary({"a": [torch.ones(1), {"b": (1, 2.5)}]})
    reasons = {r["reason"] for r in snapshot["refusals"]}
    assert "input_tree_cycle" not in reasons
    assert "input_tree_depth_exceeded" not in reasons


def test_simple_leaves_depth_and_cycle_refuse_typed() -> None:
    """Backend-resolution leaf sniffing shares the same typed guards."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        _simple_leaves(_deep_list(INPUT_TREE_MAX_DEPTH + 10, torch.ones(1)))
    assert excinfo.value.fields["code"] == "input_tree_depth_exceeded"

    with pytest.raises(InvalidArgumentError) as excinfo:
        _simple_leaves(_cyclic_list())
    assert excinfo.value.fields["code"] == "input_tree_cycle"

    shared = [torch.ones(1)]
    assert len(_simple_leaves([shared, shared])) == 2


def test_copy_arg_tree_depth_refuses_typed_and_cycles_still_reproduce() -> None:
    """The canonical input copier is depth-bounded; cycle reproduction is unchanged."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        copy_arg_tree(_deep_list(INPUT_TREE_MAX_DEPTH + 10, torch.ones(1)))
    assert excinfo.value.fields["code"] == "input_tree_depth_exceeded"

    cyclic = _cyclic_list()
    copied = copy_arg_tree(cyclic)
    assert copied[1] is copied  # the cycle is reproduced in the copy
    assert torch.equal(copied[0], cyclic[0])


def test_moderate_nesting_still_traces() -> None:
    """Inputs well under the ceiling keep working end to end."""

    log = tl.trace(nn.Identity(), _deep_list(30, torch.ones(1)))
    assert len(log) > 0


def test_copy_arg_tree_dag_is_linear_and_preserves_aliasing() -> None:
    """r-b4 R29-3: a DAG-shaped input copies O(nodes), not O(paths).

    The historical path-scoped memo copied a shared sub-container once per PATH
    (x2 per shared-substructure level; depth 25 hung capture entry ~4 minutes).
    The call-scoped memo copies it once and PRESERVES the aliasing topology the
    model itself would have seen.
    """

    import time

    node: object = [torch.ones(1)]
    for _ in range(60):  # 2**60 paths under the old memo: only a linear memo finishes
        node = [node, node]
    start = time.perf_counter()
    copied = copy_arg_tree(node)
    assert time.perf_counter() - start < 5.0
    assert copied[0] is copied[1]  # shared substructure stays aliased in the copy
    assert copied[0] is not node[0]  # ...but is a genuine copy


def test_copy_arg_tree_distinct_containers_stay_distinct() -> None:
    """Equal-valued but DISTINCT containers still copy to distinct objects."""

    left = [torch.ones(1)]
    right = [torch.ones(1)]
    copied = copy_arg_tree([left, right])
    assert copied[0] is not copied[1]
