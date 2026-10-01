"""Shared tinygrad UOp DAG traversal and signature contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.backends.tinygrad import _uop_graph
from torchlens.split.adapters import tinygrad as split_tinygrad


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


def test_shared_signature_is_bounded_stable_and_visits_each_node_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared DAG stays compact while retaining structural matching."""

    leaf = FakeUOp("BUFFER")
    shared = leaf
    for _ in range(15):
        shared = FakeUOp("ADD", shared, shared)
    second = FakeUOp("MUL", shared, leaf)
    visited: list[int] = []
    original = _uop_graph._uop_name

    def counted(node: Any) -> str:
        """Record each structural signature visit."""

        visited.append(id(node))
        return original(node)

    monkeypatch.setattr(_uop_graph, "_uop_name", counted)
    memo: dict[int, str] = {}
    signature = _uop_graph._uop_signature(second, memo)
    assert len(signature) == 71
    assert len(visited) == len(set(visited)) == len(memo) == 17
    clone = FakeUOp("BUFFER")
    rebuilt = clone
    for _ in range(15):
        rebuilt = FakeUOp("ADD", rebuilt, rebuilt)
    assert _uop_graph._uop_signature(FakeUOp("MUL", rebuilt, clone)) == signature
    assert _uop_graph._uop_signature(FakeUOp("SUB", rebuilt, clone)) != signature


def test_device_rewrite_visits_shared_uops_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One DEVICE edit preserves DAG sharing without exponential visits."""

    monkeypatch.setattr(split_tinygrad, "_tinygrad_ops", lambda: SimpleNamespace(DEVICE="DEVICE"))

    class DeviceUOp:
        """Count source reads on an identity-based UOp graph."""

        def __init__(self, op: str, src: tuple[DeviceUOp, ...] = (), arg: str = "") -> None:
            """Store one graph node."""

            self.op = op
            self._src = src
            self.arg = arg
            self.reads = 0

        @property
        def src(self) -> tuple[DeviceUOp, ...]:
            """Return sources while counting traversals."""

            self.reads += 1
            return self._src

        def replace(self, **changes: Any) -> DeviceUOp:
            """Make an immutable-style replacement."""

            return DeviceUOp(
                changes.get("op", self.op),
                changes.get("src", self._src),
                changes.get("arg", self.arg),
            )

    leaf = DeviceUOp("DEVICE", arg="CPU")
    nodes = [leaf]
    root = leaf
    for _ in range(11):
        root = DeviceUOp("ADD", (root, root))
        nodes.append(root)
    rewritten = split_tinygrad._rewrite_tinygrad_uop_device(root, "GPU")
    assert rewritten.src[0] is rewritten.src[1]
    while rewritten.op != "DEVICE":
        rewritten = rewritten.src[0]
    assert rewritten.arg == "GPU"
    assert sum(node.reads for node in nodes) <= 3 * len(nodes)
