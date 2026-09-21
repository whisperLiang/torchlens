"""Separable prefix/suffix segment state.

A heterogeneous split runtime cannot depend on mutating one monolithic model
between devices, so each segment resolves the state it needs through a
:class:`SegmentState` binding.  A binding either REFERENCES live backend state
(same device as the capture) or OWNS a device-local replica.  Shared and tied
state keeps one identity per binding: two references to the same live object
resolve to the same replica, so tied weights are never silently duplicated
into two independent tensors.

Generated Torch segments may also share a runtime-scoped replica pool. Only
explicitly shareable, non-trainable copies are pooled across distinct bindings;
mutable buffers and independently owned trainable replicas remain separate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import SplitErrorContext, SplitUnsupportedError
from .placement import DevicePlacement

StateOwnership = Literal["referenced", "owned"]


class _StateReplicaPool:
    """Reuse explicitly immutable copies within one runtime's segment bundle.

    Retaining the effective source alongside its replica makes identity keys
    safe against object-ID reuse. Nothing outside the runtime owns this pool,
    so its sources and replicas are collectible with the runtime.
    """

    def __init__(self) -> None:
        """Create an empty pool with no process-global references."""

        self._replicas: dict[tuple[str, int, str], tuple[Any, Any]] = {}

    def replicate(
        self, adapter: Any, source: Any, device: Any, *, trainable: bool, shareable: bool
    ) -> Any:
        """Copy state with separate ownership unless sharing is explicitly safe."""

        key = (
            str(getattr(adapter, "name", type(adapter).__name__)),
            id(source),
            _device_key(adapter, device),
        )
        share = shareable and not trainable
        if share and key in self._replicas:
            return self._replicas[key][1]
        replicate = getattr(adapter, "replicate_state", None)
        value = (
            replicate(source, device, trainable=trainable)
            if callable(replicate)
            else adapter.to_device(source, device)
        )
        if share:
            self._replicas[key] = (source, value)
        return value


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
        replica_pool: _StateReplicaPool | None = None,
    ) -> None:
        """Create an empty segment state binding."""

        self.adapter = adapter
        self.placement = placement
        self._replica_pool = replica_pool
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

    def resolve(
        self,
        value: Any,
        *,
        trainable: bool | None = None,
        shareable: bool = False,
        source_id: int | None = None,
    ) -> Any:
        """Return the segment-local value for one live or captured tensor.

        Parameters
        ----------
        value:
            Live parameter/buffer handle or captured literal tensor.
        trainable:
            Whether the resolved replica should carry gradients.  Defaults to
            the source value's own autograd flag.
        shareable:
            Whether this source is immutable during replay and safe to share
            across segment bindings. Non-trainability alone is insufficient:
            mutable buffers still need independent segment ownership. Trainable
            replicas are always excluded from cross-binding pooling.
        source_id:
            Stable identity of a replaceable live source, such as a registered
            Torch buffer. Referenced entries follow its current value; owned
            replicas remain independent when the live model moves devices.
        """

        if not self.adapter.is_tensor(value):
            return value
        source_id = id(value) if source_id is None else source_id
        cached = self._entries.get(source_id)
        if cached is not None:
            if cached.ownership == "owned":
                return cached.value
            if cached.value is value and (
                not self.placement.is_explicit or self._already_placed(value)
            ):
                return cached.value
        wants_grad = (
            bool(self.adapter.requires_grad(value)) if trainable is None else bool(trainable)
        )
        inherited = self._inherited.get(source_id, ()) if cached is None else ()
        if inherited:
            entry = self._rebind(
                value, inherited, trainable=wants_grad, shareable=shareable, source_id=source_id
            )
        elif not self.placement.is_explicit or self._already_placed(value):
            entry = StateEntry(
                ownership="referenced",
                value=value,
                source_id=source_id,
                trainable=wants_grad,
            )
        else:
            replica = self._replicate(value, trainable=wants_grad, shareable=shareable)
            entry = StateEntry(
                ownership="owned",
                value=replica,
                source_id=source_id,
                trainable=wants_grad,
            )
        self._entries[source_id] = entry
        return entry.value

    def _rebind(
        self,
        source: Any,
        candidates: tuple[StateEntry, ...],
        *,
        trainable: bool,
        shareable: bool,
        source_id: int,
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
        if target is None or (current is not None and self._devices_match(current, target)):
            return StateEntry(
                ownership=selected.ownership,
                value=selected.value,
                source_id=source_id,
                trainable=trainable,
            )
        # Pool the inherited effective value, never its original source handle:
        # an earlier runtime may have updated the owned replica independently.
        replica = self._replicate(
            selected.value, trainable=trainable, shareable=shareable, device=target
        )
        return StateEntry(
            ownership="owned", value=replica, source_id=source_id, trainable=trainable
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
        return self._devices_match(current, self.placement.device)

    def _devices_match(self, left: Any, right: Any) -> bool:
        """Compare concrete devices using backend normalization when available."""

        return _device_key(self.adapter, left) == _device_key(self.adapter, right)

    def _replicate(
        self, value: Any, *, trainable: bool, shareable: bool, device: Any = None
    ) -> Any:
        """Create a device-local replica that can own its own gradients."""

        target = self.placement.device if device is None else device
        if self._replica_pool is not None:
            return self._replica_pool.replicate(
                self.adapter, value, target, trainable=trainable, shareable=shareable
            )
        replicate = getattr(self.adapter, "replicate_state", None)
        if callable(replicate):
            return replicate(value, target, trainable=trainable)
        return self.adapter.to_device(value, target)

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


def _device_key(adapter: Any, device: Any) -> str:
    """Resolve backend device aliases before looking up a state replica."""

    normalize = getattr(adapter, "normalize_device", None)
    if callable(normalize):
        return str(normalize(device))
    return _normalize_device_text(str(device))


def _devices_match(left: Any, right: Any) -> bool:
    """Compare backend device descriptors by normalized text.

    Kept as a small compatibility helper for adapters that do not expose a
    device normalizer; ``SegmentState`` uses the adapter-aware variant above.
    """

    return _normalize_device_text(str(left)) == _normalize_device_text(str(right))


def _normalize_device_text(text: str) -> str:
    """Normalize a device string so ``cuda`` and ``cuda:0`` compare equal."""

    lowered = text.strip().lower()
    if lowered.endswith(":0"):
        return lowered[:-2]
    return lowered


__all__ = ["SegmentState", "StateEntry", "StateOwnership"]
