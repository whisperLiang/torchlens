"""Structured errors for TorchLens split replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SplitErrorContext:
    """Actionable context attached to split runtime errors.

    Parameters
    ----------
    backend:
        Backend name involved in the failing split operation.
    split_point:
        User-facing split point string or resolved target label.
    module_path:
        Module address associated with the failing operation, if available.
    op_type:
        Operation type associated with the failing operation, if available.
    layer_label:
        Layer label associated with the failing operation, if available.
    reason:
        Human-readable reason for the failure.
    traced_shape:
        Shape observed during tracing.
    runtime_shape:
        Shape observed during split replay.
    dtype:
        Runtime dtype name, if relevant.
    """

    backend: str
    split_point: str
    module_path: str | None
    op_type: str | None
    layer_label: str | None
    reason: str
    traced_shape: Any | None = None
    runtime_shape: Any | None = None
    dtype: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return this context as a plain dictionary."""

        return asdict(self)


class SplitError(Exception):
    """Base class for split runtime failures."""

    code = "split_error"

    def __init__(
        self,
        message: str,
        *,
        context: SplitErrorContext | None = None,
    ) -> None:
        """Create a split error.

        Parameters
        ----------
        message:
            Human-readable message.
        context:
            Optional structured context.
        """

        super().__init__(message)
        self.context = context

    def __str__(self) -> str:
        """Return the message plus compact context when present."""

        base = super().__str__()
        if self.context is None:
            return base
        return f"{base} ({self.context.reason}; backend={self.context.backend!r})"


class SplitSpecError(SplitError, ValueError):
    """Raised when a split specification cannot be resolved."""

    code = "split_spec_error"


class SplitBoundaryError(SplitError, ValueError):
    """Raised when a replay boundary violates the split ABI."""

    code = "split_boundary_error"


class SplitUnsupportedError(SplitError, NotImplementedError):
    """Raised for unsupported split capabilities or replay constructs."""

    code = "split_unsupported"


__all__ = [
    "SplitBoundaryError",
    "SplitError",
    "SplitErrorContext",
    "SplitSpecError",
    "SplitUnsupportedError",
]
