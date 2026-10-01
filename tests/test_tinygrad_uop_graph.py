"""Shared tinygrad UOp DAG traversal and signature contracts."""

from __future__ import annotations

from typing import Any

import pytest

from torchlens.backends.tinygrad import _uop_graph


class FakeUOp:
    """Minimal UOp stand-in with identity-based graph nodes."""

    def __init__(self, op: str, *src: FakeUOp) -> None:
        """Create one operation with shared source references."""

        self.op = op
        self.dtype = "float32"
        self.arg = None
        self.src = src


class Output:
    """Minimal output tensor carrying a UOp root."""

    def __init__(self, uop: FakeUOp) -> None:
        """Store the output graph root."""

        self.uop = uop


def _old_signature(node: FakeUOp) -> str:
    """Compute the prior structural signature without memoization."""

    children = ",".join(_old_signature(child) for child in node.src)
    return f"{node.op}:{node.dtype}:{node.arg}[{children}]"


def test_shared_dag_traversal_is_unique_and_topological() -> None:
    """Shared operands and output subgraphs appear once in stable order."""

    leaf = FakeUOp("BUFFER")
    shared = FakeUOp("MUL", leaf, leaf)
    left = FakeUOp("ADD", shared, leaf)
    right = FakeUOp("SUB", shared, leaf)
    outputs = [Output(left), Output(right), Output(shared)]
    order = _uop_graph._unique_uops(outputs)
    assert order == (leaf, shared, left, right)
    assert _uop_graph._unique_uops(outputs) == order
    assert len({id(node) for node in order}) == len(order)


@pytest.mark.parametrize("kind", ["FUNCTION", "CALL"])
def test_function_call_traverses_shared_arguments_only(kind: str) -> None:
    """The unbound function body is excluded even when arguments are shared."""

    body = FakeUOp("UNBOUND")
    argument = FakeUOp("BUFFER")
    called = FakeUOp(kind, body, argument, argument)
    assert _uop_graph._unique_uops([Output(called)]) == (argument, called)


def test_shared_signature_matches_old_format_and_visits_each_node_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memoization changes traversal cost without changing signature bytes."""

    leaf = FakeUOp("BUFFER")
    shared = leaf
    for _ in range(9):
        shared = FakeUOp("ADD", shared, shared)
    second = FakeUOp("MUL", shared, leaf)
    expected = _old_signature(second)
    visited: list[int] = []
    original = _uop_graph._uop_name

    def counted(node: Any) -> str:
        """Record each structural signature visit."""

        visited.append(id(node))
        return original(node)

    monkeypatch.setattr(_uop_graph, "_uop_name", counted)
    memo: dict[int, str] = {}
    assert _uop_graph._uop_signature(shared, memo) == _old_signature(shared)
    assert _uop_graph._uop_signature(second, memo) == expected
    assert len(visited) == len(set(visited)) == len(memo) == 11
