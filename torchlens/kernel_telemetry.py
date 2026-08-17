"""Detachable CUDA kernel telemetry for the gated ATen execution profile.

This module is intentionally not imported by the TorchLens package or capture
core. Importing it installs the documented-unstable ``gpu_kernels``
descriptors, while the private profiling adapter instruments the existing ATen
observer only for its own scope. Removing this module therefore leaves the ATen
profile and every non-telemetry capture path untouched.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import torch

from ._io import FieldPolicy, _json
from ._io.prerelease import (
    register_prerelease_annotations_key,
    register_prerelease_field,
)
from .data_classes.aten_op import AtenOp
from .data_classes.op import Op

_ANNOTATION_KEY = "_kernel_telemetry"
_MARKER_PREFIX = "torchlens::aten::"
_INSTRUMENTATION_LOCK = threading.RLock()
_ATEN_KERNELS: weakref.WeakKeyDictionary[AtenOp, tuple[KernelLaunch, ...]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelLaunch:
    """One CUDA kernel or memory-copy event reported by Kineto.

    All fields are measured observations from one profiler session. ``None``
    values are used only by the single ``unavailable`` disclosure row; they
    never stand in for an observed launch.
    """

    launch_name: str | None
    device: str | None
    stream: int | str | None
    duration: float | None
    runtime_correlation: int | str | None
    attribution_status: str

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "launch_name": FieldPolicy.DROP,
        "device": FieldPolicy.DROP,
        "stream": FieldPolicy.DROP,
        "duration": FieldPolicy.DROP,
        "runtime_correlation": FieldPolicy.DROP,
        "attribution_status": FieldPolicy.DROP,
    }


@dataclass(frozen=True, slots=True)
class _MarkerSpan:
    """One unique TorchLens ATen marker in Chrome-trace coordinates."""

    name: str
    sequence: int
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _TelemetryPayload:
    """Private DROP-gated annotation payload and primitive-row relation."""

    _available: bool
    _launches: tuple[KernelLaunch, ...]
    _relations: tuple[tuple[int, int], ...]

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "_available": FieldPolicy.DROP,
        "_launches": FieldPolicy.DROP,
        "_relations": FieldPolicy.DROP,
    }


_UNAVAILABLE = KernelLaunch(
    launch_name=None,
    device=None,
    stream=None,
    duration=None,
    runtime_correlation=None,
    attribution_status="unavailable",
)


def _event_args(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an event's argument mapping, or an empty mapping.

    Parameters
    ----------
    event
        Chrome-trace event mapping.

    Returns
    -------
    Mapping[str, Any]
        Normalized argument mapping.
    """

    args = event.get("args", {})
    return args if isinstance(args, Mapping) else {}


def _category_tokens(event: Mapping[str, Any]) -> frozenset[str]:
    """Return exact lower-case category tokens for one Chrome event.

    Parameters
    ----------
    event
        Chrome-trace event mapping.

    Returns
    -------
    frozenset[str]
        Comma-separated category tokens without name heuristics.
    """

    category = event.get("cat", "")
    if not isinstance(category, str):
        return frozenset()
    return frozenset(part.strip().lower() for part in category.split(",") if part.strip())


def _number(value: Any) -> float | None:
    """Coerce a finite profiler coordinate to ``float`` when possible.

    Parameters
    ----------
    value
        Profiler coordinate.

    Returns
    -------
    float | None
        Numeric value, or ``None`` for an unusable coordinate.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        result = float(value)
        if result == result and result not in {float("inf"), float("-inf")}:
            return result
    return None


def _correlation_id(event: Mapping[str, Any]) -> int | str | None:
    """Read a Kineto runtime correlation identifier without name matching.

    Parameters
    ----------
    event
        Chrome-trace event mapping.

    Returns
    -------
    int | str | None
        Correlation identifier when Kineto emitted one.
    """

    args = _event_args(event)
    for key in ("correlation", "Correlation id", "correlation_id"):
        value = args.get(key)
        if isinstance(value, int | str) and not isinstance(value, bool):
            return value
    return None


def _marker_spans(
    events: Sequence[Mapping[str, Any]], marker_sequences: Mapping[str, int]
) -> tuple[_MarkerSpan, ...]:
    """Resolve complete-duration and begin/end marker events into spans.

    Parameters
    ----------
    events
        Chrome-trace event sequence.
    marker_sequences
        Unique marker name to primitive-row sequence mapping.

    Returns
    -------
    tuple[_MarkerSpan, ...]
        Valid marker intervals ordered by primitive sequence.
    """

    spans: list[_MarkerSpan] = []
    open_markers: dict[tuple[str, int, int], float] = {}
    for event in events:
        name = event.get("name")
        if not isinstance(name, str) or name not in marker_sequences:
            continue
        phase = event.get("ph")
        start = _number(event.get("ts"))
        if start is None:
            continue
        process_id = int(event.get("pid", 0)) if isinstance(event.get("pid", 0), int) else 0
        thread_id = int(event.get("tid", 0)) if isinstance(event.get("tid", 0), int) else 0
        key = (name, process_id, thread_id)
        if phase == "X":
            duration = _number(event.get("dur"))
            if duration is not None and duration >= 0:
                spans.append(_MarkerSpan(name, marker_sequences[name], start, start + duration))
        elif phase == "B":
            open_markers[key] = start
        elif phase == "E" and key in open_markers:
            marker_start = open_markers.pop(key)
            if start >= marker_start:
                spans.append(_MarkerSpan(name, marker_sequences[name], marker_start, start))
    return tuple(sorted(spans, key=lambda item: (item.sequence, item.start)))


def _is_runtime_event(event: Mapping[str, Any]) -> bool:
    """Return whether an event belongs to an exact CUDA runtime category.

    Parameters
    ----------
    event
        Chrome-trace event mapping.

    Returns
    -------
    bool
        ``True`` for Kineto CUDA runtime/driver API events.
    """

    return bool(_category_tokens(event) & {"cuda_runtime", "cuda_driver"})


def _is_device_event(event: Mapping[str, Any]) -> bool:
    """Return whether an event is a CUDA kernel or memory-copy row.

    Parameters
    ----------
    event
        Chrome-trace event mapping.

    Returns
    -------
    bool
        ``True`` for exact Kineto device-event categories.
    """

    return bool(
        _category_tokens(event) & {"kernel", "gpu_memcpy", "gpu_memset", "cuda_kernel", "memcpy"}
    )


def _device_label(event: Mapping[str, Any]) -> str | None:
    """Return a stable device disclosure from a Kineto device event.

    Parameters
    ----------
    event
        Chrome-trace device event.

    Returns
    -------
    str | None
        Device identifier when present.
    """

    args = _event_args(event)
    for key in ("device", "Device", "device_id"):
        value = args.get(key)
        if isinstance(value, int | str) and not isinstance(value, bool):
            return str(value)
    process_id = event.get("pid")
    return str(process_id) if isinstance(process_id, int | str) else None


def _stream_id(event: Mapping[str, Any]) -> int | str | None:
    """Return a CUDA stream identifier from a Kineto device event.

    Parameters
    ----------
    event
        Chrome-trace device event.

    Returns
    -------
    int | str | None
        Stream identifier when present.
    """

    args = _event_args(event)
    for key in ("stream", "Stream", "stream_id"):
        value = args.get(key)
        if isinstance(value, int | str) and not isinstance(value, bool):
            return value
    return None


def _payload_from_chrome_events(
    events: Sequence[Mapping[str, Any]],
    marker_sequences: Mapping[str, int],
    *,
    telemetry_available: bool,
) -> _TelemetryPayload:
    """Join unique ATen markers to CUDA device events by runtime correlation.

    Marker containment identifies runtime API calls made by one redispatched
    ATen operation. Kineto's runtime correlation identifier then follows
    asynchronous work onto device events, including events whose timestamps
    fall outside the CPU marker. No operator or launch-name substring is used.

    Parameters
    ----------
    events
        Chrome-trace events exported by ``torch.profiler``.
    marker_sequences
        Unique marker name to primitive-row sequence mapping.
    telemetry_available
        Whether the CUDA/CUPTI profiler session initialized successfully.

    Returns
    -------
    _TelemetryPayload
        Launch rows plus a many-to-many primitive-sequence relation.
    """

    if not telemetry_available:
        return _TelemetryPayload(
            _available=False,
            _launches=(_UNAVAILABLE,),
            _relations=tuple((sequence, 0) for sequence in sorted(set(marker_sequences.values()))),
        )

    spans = _marker_spans(events, marker_sequences)
    sequences_by_correlation: dict[int | str, set[int]] = {}
    for event in events:
        if not _is_runtime_event(event):
            continue
        timestamp = _number(event.get("ts"))
        correlation = _correlation_id(event)
        if timestamp is None or correlation is None:
            continue
        owners = {span.sequence for span in spans if span.start <= timestamp <= span.end}
        if owners:
            sequences_by_correlation.setdefault(correlation, set()).update(owners)

    launches: list[KernelLaunch] = []
    relations: list[tuple[int, int]] = []
    for event in events:
        if not _is_device_event(event):
            continue
        correlation = _correlation_id(event)
        owners = set() if correlation is None else sequences_by_correlation.get(correlation, set())
        status = "attributed" if owners else "unattributed"
        duration = _number(event.get("dur"))
        name = event.get("name")
        launch = KernelLaunch(
            launch_name=name if isinstance(name, str) else None,
            device=_device_label(event),
            stream=_stream_id(event),
            duration=duration,
            runtime_correlation=correlation,
            attribution_status=status,
        )
        launch_index = len(launches)
        launches.append(launch)
        relations.extend((sequence, launch_index) for sequence in sorted(owners))
    return _TelemetryPayload(
        _available=True,
        _launches=tuple(launches),
        _relations=tuple(relations),
    )


def _coerce_launch(value: Any) -> KernelLaunch:
    """Coerce a loaded mapping back to a ``KernelLaunch`` facade.

    Parameters
    ----------
    value
        Live or rehydrated launch value.

    Returns
    -------
    KernelLaunch
        Typed launch row.

    Raises
    ------
    TypeError
        If the loaded value has no recognized launch representation.
    """

    if isinstance(value, KernelLaunch):
        return value
    if isinstance(value, Mapping):
        return KernelLaunch(**{name: value[name] for name in KernelLaunch.PORTABLE_STATE_SPEC})
    raise TypeError("kernel telemetry launch row has an invalid representation")


def _coerce_payload(value: Any) -> _TelemetryPayload | None:
    """Coerce a live or loaded private annotation payload.

    Parameters
    ----------
    value
        Trace annotation value.

    Returns
    -------
    _TelemetryPayload | None
        Typed payload, or ``None`` when telemetry is absent.
    """

    if isinstance(value, _TelemetryPayload):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        raw_relations = value["_relations"]
        if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, str | bytes):
            return None
        relations: list[tuple[int, int]] = []
        for item in raw_relations:
            if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
                return None
            relations.append((int(item[0]), int(item[1])))
        return _TelemetryPayload(
            _available=bool(value["_available"]),
            _launches=tuple(_coerce_launch(item) for item in value["_launches"]),
            _relations=tuple(relations),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _bind_trace_telemetry(trace: Any) -> None:
    """Bind one trace's relation rows to its live ``AtenOp`` facades.

    Parameters
    ----------
    trace
        Trace carrying the primitive profile and optional telemetry annotation.
    """

    profile = getattr(trace, "_primitive_op_profile", None)
    if profile is None:
        return
    annotations = getattr(trace, "annotations", {})
    payload = (
        _coerce_payload(annotations.get(_ANNOTATION_KEY))
        if isinstance(annotations, Mapping)
        else None
    )
    rows = tuple(getattr(profile, "primitive_ops", ()))
    if payload is None:
        for row in rows:
            _ATEN_KERNELS.pop(row, None)
        return
    indices_by_sequence: dict[int, list[int]] = {}
    for sequence, launch_index in payload._relations:
        indices_by_sequence.setdefault(sequence, []).append(launch_index)
    for row in rows:
        indices = indices_by_sequence.get(row.sequence, [])
        _ATEN_KERNELS[row] = tuple(
            payload._launches[index] for index in indices if 0 <= index < len(payload._launches)
        )


def _aten_gpu_kernels(row: AtenOp) -> tuple[KernelLaunch, ...]:
    """Return device events correlated to one primitive row.

    Parameters
    ----------
    row
        Primitive ATen row.

    Returns
    -------
    tuple[KernelLaunch, ...]
        Correlated rows, or one typed ``unavailable`` disclosure when no
        telemetry session has been bound.
    """

    return _ATEN_KERNELS.get(row, (_UNAVAILABLE,))


def _op_gpu_kernels(op: Op) -> tuple[KernelLaunch, ...]:
    """Return the deduplicated union of device events for one user Op.

    Parameters
    ----------
    op
        User-level Op facade.

    Returns
    -------
    tuple[KernelLaunch, ...]
        Correlated device rows. A single ``unavailable`` disclosure is
        returned when no telemetry session exists.
    """

    trace = op._source_trace_or_none()
    if trace is None:
        return (_UNAVAILABLE,)
    _bind_trace_telemetry(trace)
    profile = getattr(trace, "_primitive_op_profile", None)
    annotations = getattr(trace, "annotations", {})
    payload = (
        _coerce_payload(annotations.get(_ANNOTATION_KEY))
        if isinstance(annotations, Mapping)
        else None
    )
    if profile is None or payload is None or not payload._available:
        return (_UNAVAILABLE,)
    row_index = object.__getattribute__(op, "_row")
    launches: list[KernelLaunch] = []
    seen: set[int] = set()
    for aten_row in profile.primitive_ops:
        if not any(ref.op_row_index == row_index for ref in aten_row.parent_op_refs):
            continue
        for launch in _ATEN_KERNELS.get(aten_row, ()):
            identity = id(launch)
            if identity not in seen:
                seen.add(identity)
                launches.append(launch)
    return tuple(launches)


def _install_gpu_kernel_properties() -> None:
    """Install the two documented-unstable descriptors exactly once."""

    for owner, getter in ((AtenOp, _aten_gpu_kernels), (Op, _op_gpu_kernels)):
        existing = vars(owner).get("gpu_kernels")
        if existing is None:
            owner.gpu_kernels = property(getter)  # type: ignore[union-attr]
        elif not isinstance(existing, property) or existing.fget is not getter:
            raise RuntimeError(f"{owner.__name__}.gpu_kernels is already owned by another lane")


@contextmanager
def _instrument_aten_markers() -> Iterator[dict[str, int]]:
    """Temporarily bracket redispatched ATen calls with unique profiler markers.

    Yields
    ------
    dict[str, int]
        Mutable marker-name to finalized primitive-sequence relation.
    """

    from .backends.torch import _aten_capture

    with _INSTRUMENTATION_LOCK:
        original_prepare = _aten_capture._prepare_aten_call
        original_finish = _aten_capture._finish_aten_call
        pending_markers: dict[int, tuple[str, Any]] = {}
        marker_sequences: dict[str, int] = {}
        counter = 0

        def prepare(*args: Any, **kwargs: Any) -> Any:
            """Enter one marker after the core has prepared an ATen call."""

            nonlocal counter
            pending = original_prepare(*args, **kwargs)
            counter += 1
            name = f"{_MARKER_PREFIX}{counter}"
            marker = torch.profiler.record_function(name)
            marker.__enter__()
            pending_markers[id(pending)] = (name, marker)
            return pending

        def finish(state: Any, pending: Any, **kwargs: Any) -> None:
            """Exit the marker before materializing the primitive event."""

            marker_entry = pending_markers.pop(id(pending), None)
            if marker_entry is not None:
                marker_entry[1].__exit__(None, None, None)
            original_finish(state, pending, **kwargs)
            if marker_entry is not None and state.aten_events.aten_events:
                marker_sequences[marker_entry[0]] = state.aten_events.aten_events[-1].seq

        _aten_capture._prepare_aten_call = prepare
        _aten_capture._finish_aten_call = finish
        try:
            yield marker_sequences
        finally:
            _aten_capture._prepare_aten_call = original_prepare
            _aten_capture._finish_aten_call = original_finish
            for _, marker in pending_markers.values():
                marker.__exit__(None, None, None)


def _attach_payload(trace: Any, payload: _TelemetryPayload) -> None:
    """Attach a DROP-gated telemetry annotation and bind its live views.

    Parameters
    ----------
    trace
        Trace receiving the telemetry result.
    payload
        Parsed telemetry rows and relations.
    """

    trace.annotations[_ANNOTATION_KEY] = {
        "_available": payload._available,
        "_launches": tuple(
            {
                field_name: getattr(launch, field_name)
                for field_name in KernelLaunch.PORTABLE_STATE_SPEC
            }
            for launch in payload._launches
        ),
        "_relations": payload._relations,
    }
    _bind_trace_telemetry(trace)


def _profile_trace_with_cuda_kernels(factory: Callable[[], Any]) -> Any:
    """Run a private ATen-recording trace factory under CUDA kernel profiling.

    This is the non-public activation seam while ``record_aten=`` remains
    naming-gated. On hosts without CUDA, the factory still runs once and every
    primitive row receives a typed ``unavailable`` disclosure; no launch is
    fabricated.

    Parameters
    ----------
    factory
        Zero-argument callable returning one Trace.

    Returns
    -------
    Any
        The factory's Trace with a gated telemetry annotation.
    """

    from .backends.torch._aten_capture import _activate_aten_recording_for_tests

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        with _activate_aten_recording_for_tests(), _instrument_aten_markers() as marker_sequences:
            trace = factory()
        payload = _payload_from_chrome_events((), marker_sequences, telemetry_available=False)
        _attach_payload(trace, payload)
        return trace

    with TemporaryDirectory(prefix="torchlens-kernel-telemetry-") as temp_dir:
        trace_path = Path(temp_dir) / "kineto.json"
        with _activate_aten_recording_for_tests(), _instrument_aten_markers() as marker_sequences:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
            ) as profiler:
                trace = factory()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(trace_path))
        raw = _json.read_bounded(trace_path)
    raw_events = raw.get("traceEvents", ()) if isinstance(raw, Mapping) else ()
    events = tuple(event for event in raw_events if isinstance(event, Mapping))
    payload = _payload_from_chrome_events(events, marker_sequences, telemetry_available=True)
    _attach_payload(trace, payload)
    return trace


def _register_prerelease_rows() -> None:
    """Register telemetry rows and their private annotation section with S3."""

    for owner in (KernelLaunch, _TelemetryPayload):
        for field_name in owner.PORTABLE_STATE_SPEC:
            register_prerelease_field(owner, field_name, persisted_policy=FieldPolicy.KEEP)
    register_prerelease_annotations_key(_ANNOTATION_KEY, owner="L3 kernel telemetry")


_register_prerelease_rows()
_install_gpu_kernel_properties()

__all__ = ["KernelLaunch"]
