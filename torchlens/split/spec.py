"""Public split specifications and boundary tensor ABI records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .errors import SplitSpecError
from .shape import SymbolicShape


def _validate_percent_boundary(boundary: str) -> None:
    """Validate percent boundary syntax when used.

    Parameters
    ----------
    boundary:
        Boundary string from :class:`SplitSpec`.
    """

    percent_text: str | None = None
    if boundary.startswith("percent:"):
        percent_text = boundary.split(":", 1)[1]
    elif boundary.endswith("%"):
        percent_text = boundary[:-1]
    if percent_text is None:
        return
    try:
        percent = float(percent_text)
    except ValueError as exc:
        raise SplitSpecError(f"Invalid percent split {boundary!r}.") from exc
    if not 0 < percent < 100:
        raise SplitSpecError("Percent split must be strictly between 0 and 100.")


@dataclass(frozen=True)
class SplitSpec:
    """User-facing split runtime specification.

    Parameters
    ----------
    boundary:
        Split boundary string, such as ``"after:relu_1_1"`` or ``"50%"``.
    batch_symbol:
        Symbol used for dynamic leading batch dimensions.
    dynamic_batch:
        Optional inclusive runtime batch range.
    trainable:
        Whether to prepare split training support.
    mode:
        Runtime mode. ``"compiled"`` is accepted but unsupported in v1.
    backend:
        Optional explicit TorchLens backend name.
    device_policy:
        Device handling policy. Only ``"runtime"`` is supported in v1.
    use_live_param_sources:
        Whether replay should rebind params/buffers from the live model.
    """

    boundary: str
    batch_symbol: str = "B"
    dynamic_batch: tuple[int, int] | None = None
    trainable: bool = False
    mode: Literal["generated_eager", "compiled"] = "generated_eager"
    backend: str | None = None
    device_policy: Literal["runtime"] = "runtime"
    use_live_param_sources: bool | None = None

    def __post_init__(self) -> None:
        """Validate local split-spec invariants."""

        if not isinstance(self.boundary, str) or not self.boundary.strip():
            raise SplitSpecError("SplitSpec.boundary must be a non-empty string.")
        if self.dynamic_batch is not None:
            low, high = self.dynamic_batch
            if low <= 0 or high <= 0 or low > high:
                raise SplitSpecError(
                    "SplitSpec.dynamic_batch must be a positive inclusive (low, high) range."
                )
        if self.mode not in {"generated_eager", "compiled"}:
            raise SplitSpecError("SplitSpec.mode must be 'generated_eager' or 'compiled'.")
        if self.device_policy != "runtime":
            raise SplitSpecError("SplitSpec.device_policy must be 'runtime'.")
        _validate_percent_boundary(self.boundary)


BoundaryRole = Literal[
    "primary",
    "skip",
    "passthrough",
    "multi_scale_feature",
    "index",
    "shape_value",
]


@dataclass(frozen=True)
class BoundaryTensorSpec:
    """Backend-neutral metadata for one replay boundary tensor."""

    canonical_id: str
    label: str
    backend: str
    module_path: str | None
    op_type: str
    shape: SymbolicShape | None
    dtype: str | None
    requires_grad: bool | None
    role: BoundaryRole
    output_index: int | None = None
    container_path: tuple[Any, ...] = ()
    device_policy: str = "runtime"


__all__ = ["BoundaryRole", "BoundaryTensorSpec", "SplitSpec"]
