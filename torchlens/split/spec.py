"""Public split specification objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import SplitSpecError


TraceBatchMode = Literal["batch_1", "batch_gt1"]
DevicePolicy = Literal["runtime"]
SplitMode = Literal["generated_eager", "compiled"]


@dataclass(frozen=True)
class SplitSpec:
    """Configuration for preparing a TorchLens split runtime.

    Parameters
    ----------
    boundary:
        Split point, such as ``"after:model.23"``, ``"before:relu_1_2"``,
        ``"percent:50"``, or ``"50%"``.
    batch_symbol:
        Symbol used for dynamic batch dimensions.
    dynamic_batch:
        Optional inclusive ``(low, high)`` runtime batch range.
    trainable:
        Whether the runtime will be used for split training.
    trace_batch_mode:
        Policy hint for trace batch size.
    device_policy:
        Device placement policy. Only runtime placement is supported; device is
        deliberately not part of ABI compatibility.
    mode:
        Segment execution mode. ``compiled`` is reserved for a later wrapper over
        generated eager segments.
    """

    boundary: str
    batch_symbol: str = "B"
    dynamic_batch: tuple[int, int] | None = None
    trainable: bool = False
    trace_batch_mode: TraceBatchMode = "batch_gt1"
    device_policy: DevicePolicy = "runtime"
    mode: SplitMode = "generated_eager"

    def __post_init__(self) -> None:
        """Validate split spec fields."""

        if not self.boundary or not isinstance(self.boundary, str):
            raise SplitSpecError("SplitSpec.boundary must be a non-empty string.")
        if not self.batch_symbol:
            raise SplitSpecError("SplitSpec.batch_symbol must be non-empty.")
        if self.dynamic_batch is not None:
            low, high = self.dynamic_batch
            if low <= 0 or high < low:
                raise SplitSpecError("SplitSpec.dynamic_batch must be a positive inclusive range.")
        if self.device_policy != "runtime":
            raise SplitSpecError("SplitSpec.device_policy must be 'runtime'.")
        if self.mode not in {"generated_eager", "compiled"}:
            raise SplitSpecError("SplitSpec.mode must be 'generated_eager' or 'compiled'.")


__all__ = ["DevicePolicy", "SplitMode", "SplitSpec", "TraceBatchMode"]
