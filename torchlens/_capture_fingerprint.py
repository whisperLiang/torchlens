"""Capture-cache content fingerprints and instance-attribute key fragments.

Split out of ``_capture_state_helpers.py`` under the R43 file-size ratchet:
the content-hash / code-digest / attribute-fragment family that feeds the
capture-cache key. The historical import surface
(``torchlens._capture_state_helpers``) re-exports every name here.
"""

import hashlib
import itertools
import os
import types
from typing import Any

import torch
from torch import nn

from . import _state
from ._transport import to_cpu_contiguous


def _iter_tensor_inputs(obj: Any) -> list[torch.Tensor]:
    """Collect tensor leaves from a nested input object.

    Parameters
    ----------
    obj:
        Arbitrary nested input object.

    Returns
    -------
    list[torch.Tensor]
        Tensor leaves in traversal order.
    """

    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=repr):
            tensors.extend(_iter_tensor_inputs(obj[key]))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_iter_tensor_inputs(item))
    return tensors


def _hash_tensor_content(tensor: torch.Tensor) -> str:
    """Return a content hash for a tensor.

    Parameters
    ----------
    tensor:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest over tensor metadata and CPU bytes. Metadata includes
        the ORIGINAL tensor's device and ``requires_grad`` flag: a CPU->CUDA
        move or a freeze between ``cache=True`` runs changes ``device_ref``,
        timing/memory, and grad_fn metadata on the capture, so it must be a
        cache miss even though the bytes match.

        The digest frames the LOGICAL dtype, captured before the
        bf16 -> float32 transport upcast numpy requires: framing the
        post-upcast dtype made a bfloat16 tensor collide with the float32
        tensor of the same values, so ``cache=True`` could serve the WRONG
        trace across dtypes (same fix as ``op.py::_tensor_content_hash``).
    """

    with _state.pause_logging():
        cpu = to_cpu_contiguous(tensor)
        # Frame the LOGICAL dtype captured BEFORE the numpy-transport
        # bf16->float32 upcast (b5-opus-R35-1 twin; same rule as the op.py
        # dedup digest): framing the post-upcast dtype made a bfloat16 input
        # hash identically to the float32 tensor of the same values, so a
        # dtype change was a capture-cache HIT serving the wrong trace.
        logical_dtype = str(cpu.dtype)
        if cpu.dtype is torch.bfloat16:
            cpu = cpu.to(torch.float32)
        # Byte-reinterpreting uint8 view: covers dtypes numpy cannot
        # transport directly (float8 and friends), so content-bearing
        # exotic-dtype state hashes by CONTENT instead of falling back to
        # a content-blind fragment.
        payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(
        repr(
            (
                tuple(cpu.shape),
                logical_dtype,
                str(tensor.device),
                bool(tensor.requires_grad),
            )
        ).encode("utf-8")
    )
    hasher.update(payload)
    return hasher.hexdigest()


def _hash_nested_tensor_content(value: Any) -> str:
    """Return a deterministic content hash for nested tensor inputs.

    Parameters
    ----------
    value:
        Nested tensor container.

    Returns
    -------
    str
        SHA-256 digest.
    """

    tensors = _iter_tensor_inputs(value)
    entries = [_hash_tensor_content(tensor) for tensor in tensors]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _fingerprint_model_content(model: nn.Module) -> str:
    """Fingerprint model tensor contents for the capture cache.

    Parameters
    ----------
    model:
        Model to fingerprint.

    Returns
    -------
    str
        SHA-256 digest.
    """

    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode("utf-8"))
        hasher.update(_hash_tensor_content(tensor).encode("utf-8"))
    # ``state_dict()`` detaches, so a live parameter's requires_grad flag never
    # reaches the tensor hash: fold the flags explicitly (freezing params
    # changes captured grad metadata and must be a cache miss).
    for name, parameter in model.named_parameters():
        hasher.update(repr((name, bool(parameter.requires_grad))).encode("utf-8"))
    for module_name, module in model.named_modules():
        hasher.update(repr((module_name, bool(module.training))).encode("utf-8"))
        for buffer_name in sorted(module._non_persistent_buffers_set):
            buffer = module._buffers.get(buffer_name)
            hasher.update(repr((module_name, buffer_name)).encode("utf-8"))
            if isinstance(buffer, torch.Tensor):
                hasher.update(_hash_tensor_content(buffer).encode("utf-8"))
            else:
                hasher.update(repr(buffer).encode("utf-8"))
    return hasher.hexdigest()


def _hash_code_object_into(hasher: Any, code: types.CodeType, depth: int = 0) -> None:
    """Fold one code object's behavioral surface into ``hasher``.

    Covers the compiled bytecode, referenced names, and constants (recursing
    into nested code objects such as comprehensions and local closures), which
    is what changes when a ``forward`` implementation is edited. Line-number
    tables and file paths are deliberately excluded so moving a file or adding
    a comment does not invalidate the cache.
    """

    if depth > 8:
        hasher.update(b"<code-depth-ceiling>")
        return
    hasher.update(code.co_code)
    hasher.update(repr(code.co_names).encode("utf-8"))
    hasher.update(repr(code.co_varnames).encode("utf-8"))
    hasher.update(repr(code.co_freevars).encode("utf-8"))
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _hash_code_object_into(hasher, const, depth + 1)
        else:
            hasher.update(repr(const).encode("utf-8"))


def _callable_code_digest(func: Any) -> str:
    """Digest one callable's implementation for the capture-cache key.

    Plain functions (and bound/unbound methods) hash their code object plus
    default-argument reprs. Callables without a Python code object (C
    builtins, scripted callables) fall back to a stable module/qualname token
    -- NEVER ``repr(func)``, whose memory address would break cross-process
    key stability.
    """

    hasher = hashlib.sha256()
    target = getattr(func, "__func__", func)
    code = getattr(target, "__code__", None)
    if code is not None and isinstance(code, types.CodeType):
        _hash_code_object_into(hasher, code)
        for attribute in ("__defaults__", "__kwdefaults__"):
            try:
                hasher.update(repr(getattr(target, attribute, None)).encode("utf-8"))
            except Exception:
                hasher.update(f"<unreprable-{attribute}>".encode())
    else:
        module = getattr(target, "__module__", None) or type(target).__module__
        qualname = getattr(target, "__qualname__", None) or type(target).__qualname__
        hasher.update(f"<no-code:{module}.{qualname}>".encode())
    return hasher.hexdigest()


_ATTRIBUTE_FRAGMENT_DEPTH_CEILING = 4
_ATTRIBUTE_FRAGMENT_ITEM_CEILING = 256

# Per-process salt + counter minting NEVER-MATCHING fragments for values the
# key cannot soundly cover: tensor attributes whose content cannot be read,
# and containers whose size exceeds the item ceiling (truncating them would
# make two containers differing only past the cut key identically). A stable
# content-blind fragment in either case would false-HIT on changed values,
# which the key contract forbids; the salt keeps the token unique across
# processes (and across pid reuse), the counter within this process.
_ATTRIBUTE_MISS_SALT = os.urandom(8).hex()
_ATTRIBUTE_MISS_COUNTER = itertools.count()


def _never_matching_fragment(kind: str) -> object:
    """Mint a fragment that can never equal any other fragment (always-miss)."""

    return (kind, _ATTRIBUTE_MISS_SALT, next(_ATTRIBUTE_MISS_COUNTER))


def _attribute_state_fragment(value: Any, depth: int = 0) -> object:
    """Return a bounded, address-free key fragment for one instance attribute.

    Plain instance attributes routinely determine the traced program
    (``self.num_layers``, ``self.scale``, ``self.use_checkpoint``), so they
    must participate in the capture-cache key. Values are reduced to stable
    primitives: scalars by value, tensors/arrays by content hash, callables by
    code digest, containers element-wise under the depth ceiling (containers
    larger than the item ceiling mint a never-matching token -- always-miss,
    never a truncated fragment that could false-HIT on a tail difference),
    and any other object by TYPE identity only -- an opaque object's internal
    state is a documented boundary of the signature (changing it without
    changing type keeps the key; conservative for false hits on the covered
    kinds, never address-churning).
    """

    if depth > _ATTRIBUTE_FRAGMENT_DEPTH_CEILING:
        # Same rule as the item ceiling: a STABLE ceiling token would make two
        # attributes identical down to the ceiling but differing BELOW it key
        # identically (false HIT). Never-match instead.
        return _never_matching_fragment("<attr-depth-ceiling>")
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if isinstance(value, torch.Tensor):
        if value.is_meta:
            # Meta tensors carry NO bytes: shape/dtype/device metadata IS
            # their entire observable content, so a stable metadata fragment
            # cannot be content-blind for them.
            return (
                "tensor-meta",
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        try:
            return ("tensor", _hash_tensor_content(value))
        except Exception:
            # Content-unreadable tensor (sparse/exotic layout or backend):
            # degrading to a stable shape/dtype fragment made two
            # DIFFERENT-content tensors key identically -- a false cache HIT,
            # inverting this signature's "false hits never" contract. Mint a
            # never-matching token instead: this capture can never hit any
            # other entry (conservative always-miss, cache utility traded for
            # correctness).
            return _never_matching_fragment("tensor-unhashable")
    if isinstance(value, (torch.dtype, torch.device, torch.Size)):
        return ("torch-value", str(value))
    if isinstance(value, nn.Module):
        cls = type(value)
        return ("module", f"{cls.__module__}.{cls.__qualname__}")
    if isinstance(value, dict):
        # Truncating an over-ceiling container would leave items past the cut
        # OUT of the key, so two dicts differing only there would key
        # identically and serve each other's cached trace (false HIT, proven:
        # a 300-key config dict differing at insertion position 258 hit the
        # stale entry). Over-ceiling containers therefore always miss.
        if len(value) > _ATTRIBUTE_FRAGMENT_ITEM_CEILING:
            return _never_matching_fragment("dict-over-item-ceiling")
        return (
            "dict",
            len(value),
            tuple(
                sorted(
                    (
                        repr(_attribute_state_fragment(key, depth + 1)),
                        repr(_attribute_state_fragment(item, depth + 1)),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        if len(value) > _ATTRIBUTE_FRAGMENT_ITEM_CEILING:
            return _never_matching_fragment("sequence-over-item-ceiling")
        return (
            "sequence",
            len(value),
            tuple(_attribute_state_fragment(item, depth + 1) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        if len(value) > _ATTRIBUTE_FRAGMENT_ITEM_CEILING:
            return _never_matching_fragment("set-over-item-ceiling")
        member_reprs = sorted(repr(_attribute_state_fragment(item, depth + 1)) for item in value)
        return ("set", len(value), tuple(member_reprs))
    if type(value).__module__ == "numpy" and hasattr(value, "tobytes"):
        try:
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            return ("ndarray", tuple(getattr(value, "shape", ())), str(value.dtype), digest)
        except Exception:
            pass
    if callable(value):
        return ("callable", _callable_code_digest(value))
    cls = type(value)
    return ("object", f"{cls.__module__}.{cls.__qualname__}")


# Hook dicts that fire during (or around) the captured forward/backward and
# therefore change what a capture observes. State-dict/load hooks are excluded:
# they cannot affect the traced program. The ``*_with_kwargs`` /
# ``*_always_called`` companions are flag dicts keyed by handle id; ids come
# from a process-global counter and are NOT stable across processes, so only
# their VALUES are folded, aligned by registration order.
