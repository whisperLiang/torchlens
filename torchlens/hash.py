"""Provisional public structural-hash helpers.

The names in this namespace are provisional and may become ``fingerprint``-style
names after public API review. The digest is address-free and describes the
captured operation graph; it intentionally does not encode parameter values,
tensor shapes, or dtypes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import torch

from .options import CaptureOptions
from .utils.hashing import compute_graph_shape_hash

__all__ = ["StructuralHashMismatchError", "assert_unchanged", "content", "model", "trace"]


class StructuralHashMismatchError(AssertionError):
    """Raised when a model's provisional structural hash differs from its pin."""


def content(value: Any) -> str:
    """Return a deterministic SHA-256 digest of nested tensor content.

    Tensor dtype, shape, and exact logical bytes are included. Lists, tuples,
    and mappings are framed recursively so distinct input structures cannot
    collide merely because their tensor bytes concatenate to the same stream.

    Parameters
    ----------
    value:
        Tensor or nested tensor-containing value to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """

    digest = sha256()
    _update_content_digest(digest, value)
    return digest.hexdigest()


def _update_content_digest(digest: Any, value: Any) -> None:
    """Add one nested value to a framed content digest.

    Parameters
    ----------
    digest:
        Hashlib-compatible mutable digest.
    value:
        Value to frame and hash.
    """

    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            digest.update(b"tensor\0meta\0")
            digest.update(str(value.dtype).encode("utf-8") + b"\0")
            digest.update(repr(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(str(value.layout).encode("utf-8") + b"\0")
            return
        tensor = value.detach().cpu()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("utf-8") + b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(str(tensor.layout).encode("utf-8") + b"\0")
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        if tensor.is_quantized:
            _update_content_digest(digest, tensor.int_repr())
            digest.update(repr(tensor.qscheme()).encode("ascii") + b"\0")
            return
        # Flatten to 1-D before the uint8 reinterpret: ``view(torch.uint8)``
        # refuses a 0-dim (scalar) tensor ("self.dim() cannot be 0 to view Float
        # as Byte"), which otherwise crashes ``content()`` on a scalar and, via a
        # bare-except in the runnable-bundle path, silently drops the manifest
        # ``input_hash`` for scalar inputs (an attestation gap). ``reshape(-1)`` on
        # a contiguous tensor is a contiguous view and is byte-identical to the
        # prior expression for every >=1-D tensor, so pinned hashes are unchanged.
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        items = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        digest.update(len(items).to_bytes(8, "big"))
        for key, item_value in items:
            _update_content_digest(digest, key)
            _update_content_digest(digest, item_value)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        digest.update(type(value).__name__.encode("utf-8") + b"\0")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_content_digest(digest, item)
        return
    if isinstance(value, bytes | bytearray):
        raw_bytes = bytes(value)
        digest.update(b"bytes\0" + len(raw_bytes).to_bytes(8, "big") + raw_bytes)
        return
    digest.update(type(value).__module__.encode("utf-8") + b".")
    digest.update(type(value).__qualname__.encode("utf-8") + b"\0")
    digest.update(repr(value).encode("utf-8") + b"\0")


def trace(captured_trace: Any) -> str:
    """Return an address-free structural hash for an existing trace.

    This provisional name may change after public API review. The digest uses
    TorchLens' established graph-shape hasher with module addresses excluded.

    Parameters
    ----------
    captured_trace:
        Completed TorchLens trace to hash.

    Returns
    -------
    str
        SHA-256 structural hash for the trace.
    """

    return compute_graph_shape_hash(captured_trace, include_module_address=False)


def _model_hash(captured_model: Any, example_input: Any) -> str:
    """Capture a model with the canonical metadata-only hash options.

    Parameters
    ----------
    captured_model:
        Model or callable accepted by :func:`torchlens.trace`.
    example_input:
        Example positional input accepted by :func:`torchlens.trace`.

    Returns
    -------
    str
        SHA-256 structural hash for the captured trace.
    """

    from .user_funcs import trace as capture_trace

    captured_trace = capture_trace(
        captured_model,
        example_input,
        capture=CaptureOptions(layers_to_save=None, inference_only=True),
    )
    return trace(captured_trace)


def model(model: Any, example_input: Any) -> str:
    """Capture a model cheaply and return its address-free structural hash.

    This provisional name may change after public API review. The metadata-only
    capture deliberately uses ``layers_to_save=None`` so buffer events are
    represented identically to the corresponding metadata-only trace passed to
    :func:`trace`.

    Parameters
    ----------
    model:
        Model or callable accepted by :func:`torchlens.trace`.
    example_input:
        Example positional input accepted by :func:`torchlens.trace`.

    Returns
    -------
    str
        SHA-256 structural hash for the captured trace.
    """

    return _model_hash(model, example_input)


def assert_unchanged(model: Any, example_input: Any, expected: str | None) -> str:
    """Assert that a model still has a pinned address-free structural hash.

    This provisional name may change after public API review. Pass ``None`` to
    print and return a hash suitable for adding as a CI pin.

    Parameters
    ----------
    model:
        Model or callable accepted by :func:`torchlens.trace`.
    example_input:
        Example positional input accepted by :func:`torchlens.trace`.
    expected:
        Previously pinned structural hash, or ``None`` to bootstrap one.

    Returns
    -------
    str
        The actual structural hash.

    Raises
    ------
    StructuralHashMismatchError
        If ``expected`` differs from the actual hash.
    TypeError
        If ``expected`` is neither a string nor ``None``.
    """

    actual = _model_hash(model, example_input)
    if expected is None:
        print(f"TorchLens structural hash: {actual}")
        return actual
    if not isinstance(expected, str):
        raise TypeError("expected must be a structural hash string or None")
    if expected != actual:
        raise StructuralHashMismatchError(
            "TorchLens structural hash changed: "
            f"expected {expected}, got {actual}. "
            "Inspect the captured traces to find the structural divergence."
        )
    return actual
