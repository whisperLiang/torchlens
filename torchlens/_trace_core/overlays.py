"""Sparse versioned row overlays and transactions.

The frozen base is physically immutable; every public write after freeze
lands here as a ``(row, field) -> value`` entry. Reads check the overlay
first, then the base. Transactions checkpoint the overlay (plus backward
epochs) and roll back atomically — the COW-fork and intervention-rollback
substrate.
"""

from __future__ import annotations

from typing import Any, Hashable

#: Sentinel distinct from None.
_MISSING = object()


class RowOverlay:
    """Sparse per-(row, field) write overlay with version stamps."""

    __slots__ = ("_entries", "_generation")

    def __init__(self) -> None:
        """Create an empty overlay at generation 0."""

        self._entries: dict[tuple[int, str], Any] = {}
        self._generation = 0

    def __len__(self) -> int:
        """Return the number of overlaid cells."""

        return len(self._entries)

    @property
    def generation(self) -> int:
        """Monotonic write-generation stamp."""

        return self._generation

    def write(self, row: int, field: str, value: Any) -> None:
        """Overlay one cell and bump the generation."""

        self._entries[(row, field)] = value
        self._generation += 1

    def read(self, row: int, field: str, default: Any = _MISSING) -> Any:
        """Return the overlaid value, or ``default`` when absent."""

        return self._entries.get((row, field), default)

    def has(self, row: int, field: str) -> bool:
        """Return whether a cell is overlaid."""

        return (row, field) in self._entries

    def snapshot(self) -> dict[tuple[int, str], Any]:
        """Return a checkpoint of the overlay contents."""

        return dict(self._entries)

    def restore(self, checkpoint: dict[tuple[int, str], Any]) -> None:
        """Atomically restore a checkpoint (rollback)."""

        self._entries = dict(checkpoint)
        self._generation += 1


#: Module-level export of the missing sentinel for overlay readers.
MISSING = _MISSING


class Transaction:
    """Checkpoint/rollback bracket over an overlay set.

    Parameters
    ----------
    overlays:
        Mapping of name to overlay participating in the transaction.
    """

    __slots__ = ("_overlays", "_checkpoints", "_extra_state", "_extra_restore")

    def __init__(self, overlays: dict[Hashable, RowOverlay]) -> None:
        """Checkpoint every overlay immediately."""

        self._overlays = overlays
        self._checkpoints = {
            name: overlay.snapshot() for name, overlay in overlays.items()
        }
        self._extra_state: dict[Hashable, Any] = {}
        self._extra_restore: dict[Hashable, Any] = {}

    def stash(self, key: Hashable, snapshot: Any, restore: Any) -> None:
        """Register extra state (epoch lists, caches) with a restore callable."""

        self._extra_state[key] = snapshot
        self._extra_restore[key] = restore

    def rollback(self) -> None:
        """Atomically restore every participant to its checkpoint."""

        for name, overlay in self._overlays.items():
            overlay.restore(self._checkpoints[name])
        for key, restore in self._extra_restore.items():
            restore(self._extra_state[key])
