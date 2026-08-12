"""Identity-preserving payload arena.

The arena preserves IDENTITY, not equality: handing it the same live tensor
object twice returns the same handle; equal-but-distinct tensors keep
distinct handles; there is NO value-based dedup. In-place mutation stays
visible through every handle that aliases the same tensor, because the
arena stores the object itself, never a copy. Replacement writes (fork
overlays, interventions) allocate new handles and never disturb aliases of
the original.
"""

from __future__ import annotations

from typing import Any


class PayloadArena:
    """Per-core payload registry mapping live objects to dense handles."""

    __slots__ = ("_by_identity", "_values")

    def __init__(self) -> None:
        """Create an empty arena."""

        self._by_identity: dict[int, int] = {}
        self._values: list[Any] = []

    def __len__(self) -> int:
        """Return the number of registered payloads."""

        return len(self._values)

    def register(self, value: Any) -> int:
        """Return the handle for ``value``, registering it if new.

        Identity-keyed: the SAME object always maps to the same handle;
        equal-but-distinct objects get distinct handles.
        """

        handle = self._by_identity.get(id(value))
        if handle is not None and self._values[handle] is value:
            return handle
        handle = len(self._values)
        self._by_identity[id(value)] = handle
        self._values.append(value)
        return handle

    def value(self, handle: int) -> Any:
        """Return the payload object for a handle (the object ITSELF)."""

        return self._values[handle]

    def replace(self, handle: int, value: Any) -> int:
        """Allocate a NEW handle for a replacement value.

        The original handle keeps its object so existing aliases are never
        disturbed; callers store the returned handle in their overlay.
        """

        del handle  # replacement never mutates the original slot
        return self.register(value)

    def aliases(self, value: Any) -> list[int]:
        """Return every handle whose stored object IS ``value``."""

        return [
            handle
            for handle, stored in enumerate(self._values)
            if stored is value
        ]
