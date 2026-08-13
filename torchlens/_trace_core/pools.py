"""Per-trace exact-type intern pools.

The pool key contract is absorbed VERBATIM from the shipped Op metadata
compaction (``op.py`` ``_pool_key``): keys carry the exact type, so ``True``,
``1``, ``1.0``, and ``Bytes(1)`` can never collapse into one entry. Pools are
per ``TraceCore`` — NEVER process-global — and only immutable value objects
enter them; payload tensors, callables, live handles, and mutable user
objects never do.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any


class InternPool:
    """Interning table mapping values to dense int ids and back."""

    __slots__ = ("_by_key", "_values")

    def __init__(self) -> None:
        """Create an empty pool."""

        self._by_key: dict[Hashable, int] = {}
        self._values: list[Any] = []

    def __len__(self) -> int:
        """Return the number of distinct interned values."""

        return len(self._values)

    @staticmethod
    def key_for(value: Any) -> Hashable:
        """Return the exact-type injective pool key for a value."""

        return (type(value), value)

    def intern(self, value: Any) -> int:
        """Return the pool id for ``value``, interning it if new."""

        key = self.key_for(value)
        pool_id = self._by_key.get(key)
        if pool_id is None:
            pool_id = len(self._values)
            self._by_key[key] = pool_id
            self._values.append(value)
        return pool_id

    def value(self, pool_id: int) -> Any:
        """Return the interned value for a pool id."""

        return self._values[pool_id]


class ClosurePool:
    """Interned ancestor-closure pool: one entry per distinct closure.

    The column contract is a pool id; the encoding (frozenset here; a dense
    bitset over a trace-local label table is an equivalent internal choice)
    is invisible to consumers, which always receive a fresh mutable ``set``
    on materialization.
    """

    __slots__ = ("_by_closure", "_closures")

    def __init__(self) -> None:
        """Create an empty closure pool."""

        self._by_closure: dict[frozenset, int] = {}
        self._closures: list[frozenset] = []

    def __len__(self) -> int:
        """Return the number of distinct closures."""

        return len(self._closures)

    def intern(self, members: Any) -> int:
        """Intern one closure (any iterable of labels) and return its id."""

        closure = frozenset(members)
        closure_id = self._by_closure.get(closure)
        if closure_id is None:
            closure_id = len(self._closures)
            self._by_closure[closure] = closure_id
            self._closures.append(closure)
        return closure_id

    def materialize(self, closure_id: int) -> set:
        """Return a FRESH mutable set for one closure id."""

        return set(self._closures[closure_id])
