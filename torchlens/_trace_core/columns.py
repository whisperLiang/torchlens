"""Typed chunked column builders and frozen columns.

Builders accept appends and random-access writes during the BUILDING window
(postprocess steps 0-20); ``freeze()`` allocates each final array exactly
once. Dense numeric/bool/id columns are numpy-backed; interned and object
values are pool-id int columns or (transitionally) object lists — never
numpy object arrays, which would reintroduce the per-cell Python object the
store deletes.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

#: Sentinel distinct from None (None is a real storable value).
_MISSING = object()

#: numpy dtypes by declared codec name.
_CODEC_DTYPES = {
    "bool": np.bool_,
    "uint8": np.uint8,
    "int32": np.int32,
    "int64": np.int64,
    "float64": np.float64,
}


class FrozenColumn:
    """Immutable typed column with a validity bitmap.

    Parameters
    ----------
    values:
        Backing numpy array (dense codecs) or tuple (object codec).
    validity:
        Boolean numpy array; ``False`` rows read as ``None``.
    """

    __slots__ = ("_values", "_validity")

    def __init__(self, values: Any, validity: np.ndarray) -> None:
        """Bind backing storage; internal, produced by ``ColumnBuilder``."""

        self._values = values
        self._validity = validity

    def __len__(self) -> int:
        """Return the row count."""

        return len(self._validity)

    def get(self, row: int) -> Any:
        """Return the value at ``row``, or ``None`` when invalid."""

        if not self._validity[row]:
            return None
        value = self._values[row]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def __iter__(self) -> Iterator[Any]:
        """Iterate values in row order."""

        for row in range(len(self)):
            yield self.get(row)

    @property
    def nbytes(self) -> int:
        """Approximate retained bytes of the backing storage."""

        backing = (
            self._values.nbytes
            if isinstance(self._values, np.ndarray)
            else sum(0 for _ in self._values)
        )
        return int(backing) + int(self._validity.nbytes)


class ColumnBuilder:
    """Mutable chunked builder for one column.

    Parameters
    ----------
    codec:
        One of ``bool/uint8/int32/int64/float64`` for dense numpy backing,
        or ``object`` for a transitional Python-object column (used while a
        field family has not yet columnarized its value encoding).
    """

    __slots__ = ("codec", "_values", "_frozen")

    def __init__(self, codec: str) -> None:
        """Create an empty builder for ``codec``."""

        if codec != "object" and codec not in _CODEC_DTYPES:
            raise ValueError(f"unknown column codec: {codec}")
        self.codec = codec
        self._values: list[Any] = []
        self._frozen: FrozenColumn | None = None

    def __len__(self) -> int:
        """Return the current row count."""

        return len(self._values)

    def append(self, value: Any) -> int:
        """Append one value (or ``None``) and return its row index."""

        self._require_building()
        self._values.append(_MISSING if value is None else value)
        return len(self._values) - 1

    def append_missing(self) -> int:
        """Append an explicitly missing cell and return its row index."""

        self._require_building()
        self._values.append(_MISSING)
        return len(self._values) - 1

    def set(self, row: int, value: Any) -> None:
        """Write one cell during the BUILDING window."""

        self._require_building()
        self._values[row] = _MISSING if value is None else value

    def get(self, row: int) -> Any:
        """Read one cell (builder or frozen)."""

        if self._frozen is not None:
            return self._frozen.get(row)
        value = self._values[row]
        return None if value is _MISSING else value

    def freeze(self) -> FrozenColumn:
        """Allocate the final arrays once and freeze this column."""

        if self._frozen is not None:
            return self._frozen
        validity = np.array(
            [value is not _MISSING for value in self._values], dtype=np.bool_
        )
        if self.codec == "object":
            values: Any = tuple(
                None if value is _MISSING else value for value in self._values
            )
        else:
            dtype = _CODEC_DTYPES[self.codec]
            fill = False if dtype is np.bool_ else 0
            values = np.array(
                [fill if value is _MISSING else value for value in self._values],
                dtype=dtype,
            )
        self._frozen = FrozenColumn(values, validity)
        self._values = []
        return self._frozen

    @property
    def frozen(self) -> bool:
        """Return whether this column has been frozen."""

        return self._frozen is not None

    def _require_building(self) -> None:
        """Raise if the column is already frozen."""

        if self._frozen is not None:
            raise RuntimeError("column is frozen; writes go to the overlay")
