"""The canonical edge-occurrence table with CSR indexes.

Every variable-cardinality relation stores each edge OCCURRENCE once —
(source row, target row, use kind, argument position/path, stable sequence)
— so parallel edges and per-occurrence metadata can never collapse. Dense
single-target relations use plain foreign-key columns instead and do not
live here. Parent/child CSR indexes hold EDGE ids (not target ids), so the
occurrence metadata rides along every traversal.
"""

from __future__ import annotations

from typing import Any, Iterator, NamedTuple

import numpy as np


class Edge(NamedTuple):
    """One edge occurrence."""

    edge_id: int
    source: int
    target: int
    use_kind: int
    arg_position: Any
    seq: int


class _CsrIndex:
    """Frozen CSR mapping row -> edge-id slice."""

    __slots__ = ("_offsets", "_edge_ids")

    def __init__(self, offsets: np.ndarray, edge_ids: np.ndarray) -> None:
        """Bind offset and edge-id arrays."""

        self._offsets = offsets
        self._edge_ids = edge_ids

    def edge_ids(self, row: int) -> np.ndarray:
        """Return the edge ids incident to ``row`` in insertion order."""

        return self._edge_ids[self._offsets[row] : self._offsets[row + 1]]


class EdgeTable:
    """Chunked builder + frozen store for one edge family."""

    __slots__ = (
        "_sources",
        "_targets",
        "_use_kinds",
        "_arg_positions",
        "_seqs",
        "_frozen",
        "_by_source",
        "_by_target",
    )

    def __init__(self) -> None:
        """Create an empty edge table."""

        self._sources: list[int] = []
        self._targets: list[int] = []
        self._use_kinds: list[int] = []
        self._arg_positions: list[Any] = []
        self._seqs: list[int] = []
        self._frozen = False
        self._by_source: _CsrIndex | None = None
        self._by_target: _CsrIndex | None = None

    def __len__(self) -> int:
        """Return the number of edge occurrences."""

        return len(self._sources)

    def add(
        self,
        source: int,
        target: int,
        *,
        use_kind: int = 0,
        arg_position: Any = None,
        seq: int | None = None,
    ) -> int:
        """Append one edge occurrence and return its edge id.

        Parallel edges are first-class: repeated (source, target) pairs
        stay distinct occurrences.
        """

        if self._frozen:
            raise RuntimeError("edge table is frozen")
        edge_id = len(self._sources)
        self._sources.append(source)
        self._targets.append(target)
        self._use_kinds.append(use_kind)
        self._arg_positions.append(arg_position)
        self._seqs.append(seq if seq is not None else edge_id)
        return edge_id

    def edge(self, edge_id: int) -> Edge:
        """Return one edge occurrence by id."""

        return Edge(
            edge_id,
            self._sources[edge_id],
            self._targets[edge_id],
            self._use_kinds[edge_id],
            self._arg_positions[edge_id],
            self._seqs[edge_id],
        )

    def freeze(self, n_source_rows: int, n_target_rows: int) -> None:
        """Build both CSR indexes; the table becomes immutable."""

        if self._frozen:
            return
        self._by_source = self._build_csr(self._sources, n_source_rows)
        self._by_target = self._build_csr(self._targets, n_target_rows)
        self._frozen = True

    def _build_csr(self, keys: list[int], n_rows: int) -> _CsrIndex:
        """Build one CSR index over ``keys`` preserving insertion order."""

        counts = np.zeros(n_rows + 1, dtype=np.int64)
        for key in keys:
            counts[key + 1] += 1
        offsets = np.cumsum(counts)
        edge_ids = np.empty(len(keys), dtype=np.int64)
        cursor = offsets[:-1].copy()
        for edge_id, key in enumerate(keys):
            edge_ids[cursor[key]] = edge_id
            cursor[key] += 1
        return _CsrIndex(offsets, edge_ids)

    def out_edges(self, source: int) -> Iterator[Edge]:
        """Iterate edges whose source is ``source``, in insertion order."""

        if not self._frozen or self._by_source is None:
            raise RuntimeError("freeze() before CSR traversal")
        for edge_id in self._by_source.edge_ids(source):
            yield self.edge(int(edge_id))

    def in_edges(self, target: int) -> Iterator[Edge]:
        """Iterate edges whose target is ``target``, in insertion order."""

        if not self._frozen or self._by_target is None:
            raise RuntimeError("freeze() before CSR traversal")
        for edge_id in self._by_target.edge_ids(target):
            yield self.edge(int(edge_id))

    def targets_of(self, source: int) -> list[int]:
        """Return target rows for ``source`` in insertion order."""

        return [edge.target for edge in self.out_edges(source)]

    def sources_of(self, target: int) -> list[int]:
        """Return source rows for ``target`` in insertion order."""

        return [edge.source for edge in self.in_edges(target)]
