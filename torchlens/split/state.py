"""Separable prefix/suffix segment state.

A heterogeneous split runtime cannot depend on mutating one monolithic model
between devices, so each segment resolves the state it needs through a
:class:`SegmentState` binding.  A binding either REFERENCES live backend state
(same device as the capture) or OWNS a device-local replica.  Shared and tied
state keeps one identity per binding: two references to the same live object
resolve to the same replica, so tied weights are never silently duplicated
into two independent tensors.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import SplitErrorContext, SplitUnsupportedError
from .placement import DevicePlacement

StateOwnership = Literal["referenced", "owned"]


@dataclass(frozen=True)
class StateEntry:
    """One resolved state value for a segment."""

    ownership: StateOwnership
    value: Any = field(repr=False, compare=False)
    source_id: int = 0
    trainable: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like state metadata without the tensor payload."""

        return {
            "ownership": self.ownership,
            "source_id": self.source_id,
            "trainable": self.trainable,
        }


class SegmentState:
    """Per-segment state binding with placement and alias awareness.

    The binding is lazy: a live handle is resolved on first use and cached by
    source-object identity, so a tied parameter used by several ops resolves
    to exactly one value.
    """

    def __init__(
        self,
        *,
        adapter: Any,
        placement: DevicePlacement,
    ) -> None:
        """Create an empty segment state binding."""

        self.adapter = adapter
        self.placement = placement
        self._entries: dict[int, StateEntry] = {}
        self._inherited: dict[int, tuple[StateEntry, ...]] = {}

    def inherit_from(self, *bindings: SegmentState) -> None:
        """Stage effective source values from an existing runtime for lazy rebinding.

        Values keep their original source identities. They only become entries
        in this binding when consumed, so a recut cannot expose parameters from
        the other segment to its optimizer.
        """

        self.inherit_entries(entry for binding in bindings for entry in binding.entries())

    def inherit_entries(self, entries: Iterable[StateEntry]) -> None:
        """Stage only the prior entries used by a newly selected segment's nodes."""

        inherited: dict[int, list[StateEntry]] = {}
        for entry in entries:
            candidates = inherited.setdefault(entry.source_id, [])
            if not any(candidate.value is entry.value for candidate in candidates):
                candidates.append(entry)
        self._inherited = {key: tuple(values) for key, values in inherited.items()}

    def entries(self) -> tuple[StateEntry, ...]:
        """Return every effective source binding, including live references."""

        return tuple(self._entries.values())

    @property
    def owns_state(self) -> bool:
        """Return whether this binding owns any device-local replica."""

        return any(entry.ownership == "owned" for entry in self._entries.values())

    def resolve(self, value: Any, *, trainable: bool | None = None) -> Any:
        """Return the segment-local value for one live or captured tensor.

        Parameters
        ----------
        value:
            Live parameter/buffer handle or captured literal tensor.
        trainable:
            Whether the resolved replica should carry gradients.  Defaults to
            the source value's own autograd flag.
        """

        if not self.adapter.is_tensor(value):
            return value
        source_id = id(value)
        cached = self._entries.get(source_id)
        if cached is not None:
            return cached.value
        wants_grad = (
            bool(self.adapter.requires_grad(value)) if trainable is None else bool(trainable)
        )
        inherited = self._inherited.get(source_id, ())
        if inherited:
            entry = self._rebind(value, inherited, trainable=wants_grad)
        elif not self.placement.is_explicit or self._already_placed(value):
            entry = StateEntry(
                ownership="referenced",
                value=value,
                source_id=source_id,
                trainable=wants_grad,
            )
        else:
            replica = self._replicate(value, trainable=wants_grad)
            entry = StateEntry(
                ownership="owned",
                value=replica,
                source_id=source_id,
                trainable=wants_grad,
            )
        self._entries[source_id] = entry
        return entry.value

    def _rebind(
        self, source: Any, candidates: tuple[StateEntry, ...], *, trainable: bool
    ) -> StateEntry:
        """Keep an effective value or copy it to a new device without source writes."""

        selected = candidates[0]
        for candidate in candidates[1:]:
            if not self._same_value(selected.value, candidate.value):
                raise SplitUnsupportedError(
                    "Cannot recut divergent replicas of a shared state value; synchronize "
                    "the segment replicas before changing the split point.",
                    context=SplitErrorContext(
                        backend=self.adapter.name,
                        split_point="",
                        reason="divergent segment state replicas",
                    ),
                )
            if candidate.ownership == "owned":
                selected = candidate
        device_of = getattr(self.adapter, "device_of", None)
        target = self.placement.device
        if target is None and callable(device_of):
            target = device_of(source)
        current = device_of(selected.value) if callable(device_of) else None
        if target is None or (current is not None and _devices_match(current, target)):
            return StateEntry(
                ownership=selected.ownership,
                value=selected.value,
                source_id=id(source),
                trainable=trainable,
            )
        replicate = getattr(self.adapter, "replicate_state", None)
        replica = (
            replicate(selected.value, target, trainable=trainable)
            if callable(replicate)
            else self.adapter.to_device(selected.value, target)
        )
        return StateEntry(
            ownership="owned", value=replica, source_id=id(source), trainable=trainable
        )

    def _same_value(self, left: Any, right: Any) -> bool:
        """Compare competing replicas exactly before coalescing a tied source."""

        if left is right:
            return True
        if self.adapter.shape(left) != self.adapter.shape(right):
            return False
        if self.adapter.dtype_name(left) != self.adapter.dtype_name(right):
            return False
        device_of = getattr(self.adapter, "device_of", None)
        if callable(device_of):
            right = self.adapter.to_device(right, device_of(left))
        return bool(self.adapter.allclose(left, right, atol=0, rtol=0))

    def _already_placed(self, value: Any) -> bool:
        """Return whether ``value`` already lives on the requested device."""

        device_of = getattr(self.adapter, "device_of", None)
        if not callable(device_of):
            return False
        current = device_of(value)
        if current is None:
            return False
        return _devices_match(current, self.placement.device)

    def _replicate(self, value: Any, *, trainable: bool) -> Any:
        """Create a device-local replica that can own its own gradients."""

        replicate = getattr(self.adapter, "replicate_state", None)
        if callable(replicate):
            return replicate(value, self.placement.device, trainable=trainable)
        return self.adapter.to_device(value, self.placement.device)

    def owned_entries(self) -> tuple[StateEntry, ...]:
        """Return the device-local replicas this binding owns."""

        return tuple(entry for entry in self._entries.values() if entry.ownership == "owned")

    def trainable_values(self) -> list[Any]:
        """Return owned replicas that carry gradients, in resolution order."""

        return [
            entry.value
            for entry in self._entries.values()
            if entry.ownership == "owned" and entry.trainable
        ]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like binding diagnostics."""

        entries = list(self._entries.values())
        return {
            "placement": self.placement.as_dict(),
            "entries": len(entries),
            "owned": sum(1 for entry in entries if entry.ownership == "owned"),
            "referenced": sum(1 for entry in entries if entry.ownership == "referenced"),
            "trainable_owned": sum(
                1 for entry in entries if entry.ownership == "owned" and entry.trainable
            ),
        }


def _devices_match(left: Any, right: Any) -> bool:
    """Compare two backend device descriptors by normalized text."""

    left_text = str(left)
    right_text = str(right)
    if left_text == right_text:
        return True
    return _normalize_device_text(left_text) == _normalize_device_text(right_text)


def _normalize_device_text(text: str) -> str:
    """Normalize a device string so ``cuda`` and ``cuda:0`` compare equal."""

    lowered = text.strip().lower()
    if lowered.endswith(":0"):
        return lowered[:-2]
    return lowered


__all__ = ["SegmentState", "StateEntry", "StateOwnership"]
