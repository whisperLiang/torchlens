"""Errors raised by TorchLens split replay and split training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SplitError(RuntimeError):
    """Base class for split runtime errors."""


class SplitSpecError(ValueError):
    """Raised when a split specification is invalid."""


class SplitBoundaryError(ValueError):
    """Raised when a boundary payload is incompatible with a split runtime."""


class SplitUnsupportedError(SplitError):
    """Raised when TorchLens cannot lower an observed operation into split runtime."""


@dataclass(frozen=True)
class SplitErrorContext:
    """Debug context included in split errors.

    Parameters
    ----------
    split_point:
        User-requested split point.
    module_path:
        TorchLens module path for the failing operation.
    op_type:
        Operation type/name.
    layer_label:
        TorchLens layer label.
    traced_shape:
        Shape observed during trace.
    runtime_shape:
        Shape observed at runtime, when available.
    dtype:
        Tensor dtype.
    reason:
        Human-readable failure reason.
    """

    split_point: str | None = None
    module_path: str | None = None
    op_type: str | None = None
    layer_label: str | None = None
    traced_shape: Any = None
    runtime_shape: Any = None
    dtype: Any = None
    reason: str = ""

    def format(self) -> str:
        """Return a compact context string.

        Returns
        -------
        str
            Human-readable error context.
        """

        parts = [
            f"split_point={self.split_point!r}",
            f"module_path={self.module_path!r}",
            f"op_type={self.op_type!r}",
            f"layer_label={self.layer_label!r}",
            f"traced_shape={self.traced_shape!r}",
            f"runtime_shape={self.runtime_shape!r}",
            f"dtype={self.dtype!r}",
            f"reason={self.reason}",
        ]
        return ", ".join(parts)


def unsupported(context: SplitErrorContext) -> SplitUnsupportedError:
    """Build a standardized unsupported-lowering error.

    Parameters
    ----------
    context:
        Context describing the unsupported operation.

    Returns
    -------
    SplitUnsupportedError
        Error ready to raise.
    """

    return SplitUnsupportedError(f"Unsupported split lowering: {context.format()}")


__all__ = [
    "SplitBoundaryError",
    "SplitError",
    "SplitErrorContext",
    "SplitSpecError",
    "SplitUnsupportedError",
    "unsupported",
]
