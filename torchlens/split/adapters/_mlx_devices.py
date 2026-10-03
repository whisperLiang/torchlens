"""MLX device validation and scoped execution placement."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from ..errors import SplitErrorContext, SplitUnsupportedError


def _device_from_string(mx: Any, device: str) -> Any:
    """Parse a native device spelling, rejecting missing backends and CPU indexes."""

    match = re.fullmatch(r"(cpu|gpu|metal|cuda)(?::(\d+))?", device.lower())
    if match is None:
        raise ValueError(f"Unknown MLX device {device!r}.")
    kind, index_text = match.groups()
    index = int(index_text or 0)
    if kind in {"metal", "cuda"}:
        native = getattr(mx, kind, None)
        available = getattr(native, "is_available", None)
        if not callable(available) or not available():
            raise ValueError(f"MLX {kind} backend is unavailable.")
    if kind == "cpu" and index != 0:
        raise ValueError("MLX supports only CPU device 0.")
    return mx.Device(mx.cpu if kind == "cpu" else mx.gpu, index)


def resolve_mlx_device(device: Any, *, split_point: str = "") -> Any:
    """Resolve a device or stream and refuse unavailable native backends.

    Parameters
    ----------
    device
        MLX device, stream, device type, or indexed CPU/GPU string.
    split_point
        Boundary spelling for refusal diagnostics.
    """

    import mlx.core as mx

    try:
        if isinstance(device, mx.Stream):
            resolved = device.device
        elif isinstance(device, mx.Device):
            resolved = device
        elif isinstance(device, mx.DeviceType):
            resolved = mx.Device(device)
        elif isinstance(device, str):
            resolved = _device_from_string(mx, device)
        else:
            raise ValueError(f"Invalid MLX device {device!r}.")
        # This probes the native backend without allocating a new stream or
        # changing a process-global default. CPU-only builds reject GPU here.
        if mx.default_stream(resolved).device != resolved:
            raise ValueError(f"MLX cannot address device {resolved!r}.")
        return resolved
    except (ValueError, RuntimeError, TypeError, IndexError) as exc:
        raise SplitUnsupportedError(
            f"Cannot use MLX split device {device!r}: {exc}",
            context=SplitErrorContext(
                backend="mlx",
                split_point=split_point,
                reason="invalid or unavailable MLX device",
            ),
        ) from exc


def mlx_execution_context(device: Any, *, split_point: str = "") -> Any:
    """Scope MLX operations to one requested stream without changing ambient state."""

    if device is None:
        return nullcontext()
    import mlx.core as mx

    stream = (
        device
        if isinstance(device, mx.Stream)
        else resolve_mlx_device(device, split_point=split_point)
    )
    return _preserved_stream(mx, stream)


@contextmanager
def _preserved_stream(mx: Any, stream: Any) -> Iterator[None]:
    """Restore the target device's stream even when the ambient device differs."""

    previous_device = mx.default_device()
    target_device = stream.device if isinstance(stream, mx.Stream) else stream
    previous_stream = mx.default_stream(target_device)
    try:
        with mx.stream(stream):
            yield
    finally:
        mx.set_default_stream(previous_stream)
        mx.set_default_device(previous_device)


__all__ = ["mlx_execution_context", "resolve_mlx_device"]
