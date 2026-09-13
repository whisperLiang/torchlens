"""Backend-neutral prefix/suffix device placement.

A split runtime executes a prefix and a suffix that may live on different
devices.  This module carries the *declared* placement; adapters own the
backend-specific movement of state, inputs, boundaries, and gradients.
TorchLens never performs network transport: a higher-level system may
serialize a boundary and ship it elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import SplitErrorContext, SplitUnsupportedError

SegmentName = Literal["prefix", "suffix"]


@dataclass(frozen=True)
class DevicePlacement:
    """Requested device for one split segment.

    ``device is None`` means "wherever the capture already lives" — no state
    movement is performed and the segment behaves exactly as an unplaced
    runtime.
    """

    device: Any = None

    @property
    def is_explicit(self) -> bool:
        """Return whether a concrete device was requested."""

        return self.device is not None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like placement metadata."""

        return {"device": None if self.device is None else str(self.device)}


@dataclass(frozen=True)
class PlacementPlan:
    """Declared placement for both split segments."""

    prefix: DevicePlacement = field(default_factory=DevicePlacement)
    suffix: DevicePlacement = field(default_factory=DevicePlacement)

    @classmethod
    def unplaced(cls) -> PlacementPlan:
        """Return the default plan that moves nothing."""

        return cls()

    @classmethod
    def on(cls, device: Any) -> PlacementPlan:
        """Return a plan that places both segments on one device."""

        placement = DevicePlacement(device)
        return cls(prefix=placement, suffix=placement)

    @classmethod
    def across(cls, prefix: Any, suffix: Any) -> PlacementPlan:
        """Return a heterogeneous plan with explicit prefix/suffix devices."""

        return cls(prefix=DevicePlacement(prefix), suffix=DevicePlacement(suffix))

    @property
    def is_explicit(self) -> bool:
        """Return whether either segment declared a device."""

        return self.prefix.is_explicit or self.suffix.is_explicit

    @property
    def is_heterogeneous(self) -> bool:
        """Return whether prefix and suffix declare different devices."""

        if not (self.prefix.is_explicit and self.suffix.is_explicit):
            return self.prefix.is_explicit != self.suffix.is_explicit
        return str(self.prefix.device) != str(self.suffix.device)

    def for_segment(self, segment: SegmentName) -> DevicePlacement:
        """Return the placement for one segment."""

        return self.prefix if segment == "prefix" else self.suffix

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like plan metadata."""

        return {"prefix": self.prefix.as_dict(), "suffix": self.suffix.as_dict()}


def require_placement_support(
    adapter: Any,
    plan: PlacementPlan,
    *,
    split_point: str,
) -> None:
    """Reject an explicit placement on a backend that cannot honor it."""

    if not plan.is_explicit:
        return
    if bool(getattr(adapter, "supports_state_placement", False)):
        return
    backend = str(getattr(adapter, "name", "unknown"))
    raise SplitUnsupportedError(
        f"backend={backend!r} cannot place split segment state on an explicit device; "
        "move the model yourself and prepare an unplaced runtime, or transport the "
        "boundary with ReplayBoundary.to(...).",
        context=SplitErrorContext(
            backend=backend,
            split_point=split_point,
            reason="unsupported segment state placement",
        ),
    )


def move_value(adapter: Any, value: Any, placement: DevicePlacement) -> Any:
    """Move one tensor-like value to a placement, leaving others untouched."""

    if not placement.is_explicit:
        return value
    if not adapter.is_tensor(value):
        return value
    return adapter.to_device(value, placement.device)


def move_tree(adapter: Any, value: Any, placement: DevicePlacement) -> Any:
    """Move every tensor leaf of a nested container to a placement."""

    if not placement.is_explicit:
        return value
    if adapter.is_tensor(value):
        return adapter.to_device(value, placement.device)
    if isinstance(value, dict):
        return type(value)(
            {key: move_tree(adapter, item, placement) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        items = [move_tree(adapter, item, placement) for item in value]
        if hasattr(type(value), "_fields"):
            return type(value)(*items)
        return type(value)(items)
    return value


__all__ = [
    "DevicePlacement",
    "PlacementPlan",
    "SegmentName",
    "move_tree",
    "move_value",
    "require_placement_support",
]
