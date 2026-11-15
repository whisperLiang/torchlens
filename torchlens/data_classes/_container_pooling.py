"""Container-cell compaction without materializing a second graph of candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from .._trace_core.op_store import PooledCell
from .op import _container_pool_key, _count_container_ids

_CELL_CLASSES = (dict, list, set, defaultdict)
_CENSUS_CLASSES = (*_CELL_CLASSES, tuple)
_STREAM_POOL_MIN_ROWS = 512


def _container_cells(
    stores: list[tuple[Any, Any]],
) -> Iterator[tuple[Any, int, int, list[Any], Any]]:
    """Iterate sanctioned mutable cells without retaining a candidate inventory."""
    for store, fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        fields = range(store.layout.n_fields) if fids is None else fids
        for row_index, cells in enumerate(rows):
            for fid in fields:
                value = cells[fid]
                if value.__class__ in _CELL_CLASSES:
                    yield store, row_index, fid, cells, value


def _alias_counts(stores: list[tuple[Any, Any]]) -> dict[int, int]:
    """Census all cells, including fields excluded from pooling, before any write."""
    counts: dict[int, int] = {}
    expanded: set[int] = set()
    for store, _fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        for cells in rows:
            for value in cells:
                if value.__class__ in _CENSUS_CLASSES:
                    _count_container_ids(value, counts, expanded, 0)
    return counts


def _candidate_key(value: Any, alias_counts: dict[int, int]) -> Any:
    """Key an unaliased cell; shared nested containers remain coupled."""
    visited: list[int] = []
    key = _container_pool_key(value, 0, visited)
    if key is None or any(alias_counts[oid] > 1 for oid in visited):
        return None
    return key


def _pool_cell(value: Any, key: Any, counts: dict[Any, int], pool: dict[Any, PooledCell]) -> Any:
    """Apply the exact duplicate threshold and preserve unpoolable cells."""
    if key is None or (value and counts.get(key, 0) < 3):
        return value
    cell = pool.get(key)
    if cell is None:
        cell = pool[key] = PooledCell.from_value(value)
    return cell


def pool_container_cells(
    stores: list[tuple[Any, Any]], container_pool: dict[Any, PooledCell]
) -> None:
    """Pool exact duplicates with streaming passes and an alias-preserving census.

    Parameters
    ----------
    stores:
        Building stores and their sanctioned field ids; None means all fields.
    container_pool:
        Trace-local exact-content pool shared across the stores.
    """
    stores = [(store, None if fids is None else tuple(frozenset(fids))) for store, fids in stores]
    if sum(len(store.rows_building() or ()) for store, _fids in stores) < _STREAM_POOL_MIN_ROWS:
        _pool_small_tables(stores, container_pool)
        return
    aliases = _alias_counts(stores)
    fingerprints: dict[int, int] = {}
    for _store, _row, _fid, _cells, value in _container_cells(stores):
        key = _candidate_key(value, aliases)
        if key is not None:
            fingerprint = hash(key)
            fingerprints[fingerprint] = min(3, fingerprints.get(fingerprint, 0) + 1)

    # A fingerprint is only a cheap rejection filter. Count full keys again
    # for possible duplicates so hash collisions cannot change the >=3 rule.
    repeated_keys: dict[Any, int] = {}
    for _store, _row, _fid, _cells, value in _container_cells(stores):
        key = _candidate_key(value, aliases)
        if key is not None and fingerprints[hash(key)] >= 3:
            repeated_keys[key] = min(3, repeated_keys.get(key, 0) + 1)
    del fingerprints

    for _store, _row, fid, cells, value in _container_cells(stores):
        key = _candidate_key(value, aliases)
        cells[fid] = _pool_cell(value, key, repeated_keys, container_pool)
    del repeated_keys
    _compact_singletons(stores, aliases)


def _compact_singletons(stores: list[tuple[Any, Any]], aliases: dict[int, int]) -> None:
    """Compact unaliased singleton string lists in kind tables only."""
    kind_stores = [(store, None) for store, fids in stores if fids is None]
    for store, row, fid, cells, value in _container_cells(kind_stores):
        if (
            value.__class__ is list
            and len(value) == 1
            and value[0].__class__ is str
            and aliases[id(value)] == 1
        ):
            element = value[0]
            cells[fid] = element
            store.register_compacted_singleton(row, fid, element)


def _pool_small_tables(
    stores: list[tuple[Any, Any]], container_pool: dict[Any, PooledCell]
) -> None:
    """Pool duplicate/empty immutable-content container cells across stores.

    Parameters
    ----------
    stores:
        ``(store, fids)`` pairs; ``fids`` is an iterable of sanctioned field
        ids or ``None`` for every field (kind tables). Only building-phase
        row stores participate (``rows_building()`` returning ``None`` skips
        the store) -- pooling always precedes the physical freeze.
    container_pool:
        Pass-local ``content key -> PooledCell`` table shared across stores,
        so equal content pools trace-wide.
    """

    # ONE scan over every cell: the alias census recurses only into cells
    # whose class can hold a container (everything else is skipped by an
    # inline class test — the vast majority of cells are scalars/None), and
    # the same scan collects the pool candidates (container-classed cells at
    # sanctioned fids) and the singleton-compaction sites (kind-table list
    # cells), so neither later pass re-walks rows x fields.
    id_counts: dict[int, int] = {}
    expanded_ids: set[int] = set()
    cell_sites: list[tuple[Any, int, Any]] = []
    singleton_sites: list[tuple[Any, int, int, Any, Any]] = []
    any_swept = False
    for store, fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        any_swept = True
        fid_set = None if fids is None else frozenset(fids)
        for row_idx, row_cells in enumerate(rows):
            for fid, value in enumerate(row_cells):
                cls = value.__class__
                if cls is tuple:
                    _count_container_ids(value, id_counts, expanded_ids, 0)
                    continue
                if not (cls is dict or cls is list or cls is set or cls is defaultdict):
                    continue
                _count_container_ids(value, id_counts, expanded_ids, 0)
                if fid_set is None or fid in fid_set:
                    cell_sites.append((row_cells, fid, value))
                if fid_set is None and cls is list:
                    singleton_sites.append((store, row_idx, fid, row_cells, value))
    if not any_swept:
        return
    candidates: list[tuple[Any, int, Any, Any]] = []
    key_counts: dict[Any, int] = {}
    for row_cells, fid, value in cell_sites:
        visited_ids: list[int] = []
        key = _container_pool_key(value, 0, visited_ids)
        if key is None:
            continue
        if any(id_counts[oid] > 1 for oid in visited_ids):
            continue
        candidates.append((row_cells, fid, value, key))
        key_counts[key] = key_counts.get(key, 0) + 1
    for row_cells, fid, value, key in candidates:
        # Empty containers always pool (all empties of a class share ONE
        # cell+prototype). Non-empty content needs >= 3 occurrences: a pooled
        # key retains 2 objects (cell + detached prototype), so pooling a
        # pair is object-neutral before any read and negative after.
        row_cells[fid] = _pool_cell(value, key, key_counts, container_pool)
    # Singleton-label compaction (M14 slice 2, kind tables only): a
    # one-element list holding exactly one str (per-record label/address
    # lists — distinct content per row, so PooledCell's >= 3 threshold never
    # reaches them) stores the bare element, registered on the store so
    # decode fires only for the exact registered object. Same alias census
    # as pooling: a list reachable from more than one swept cell never
    # compacts (breaking its mutation coupling is not sanctioned).
    for store, row_idx, fid, row_cells, value in singleton_sites:
        if (
            row_cells[fid] is value  # not already pooled above
            and len(value) == 1
            and value[0].__class__ is str
            and id_counts[id(value)] == 1
        ):
            element = value[0]
            row_cells[fid] = element
            store.register_compacted_singleton(row_idx, fid, element)
