"""Tensor-variant detection and pre-flight guards for ``trace``.

TorchLens was designed around standard dense ``torch.Tensor`` /
``torch.nn.Parameter`` objects on real (CPU/CUDA/MPS) devices.  A number of
tensor variants break the logging pipeline in different ways:

================  =================================================  =================
Variant           Why TorchLens cannot handle it today                Detection outcome
================  =================================================  =================
Meta tensor       No storage, so out saving returns garbage;  raise RuntimeError
                  ``.clone()`` yields another meta tensor, etc.
Sparse tensor     ``safe_copy``/print-override paths assume dense    raise RuntimeError
                  layouts; postprocess indexing uses ``.numel()``
                  which double-counts sparse entries.
Symbolic shape   Dimensions that are ``torch.SymInt`` /              raise RuntimeError
                  ``torch.SymFloat`` break shape-dependent metadata
                  (flops, tensor memory, counter alignment).
Tracing tensor    ``FakeTensor`` / ``FunctionalTensor`` carry no      raise RuntimeError
                  data, so every value-reading step (``safe_copy``,
                  ``torch.equal``, ``.item()``, ``data_ptr()``) is
                  meaningless; torch's own fake machinery aborts
                  mid-forward with a bare ``AssertionError``.
Quantized model   Partial support: logging works but FLOPs are        warn (keep going)
                  computed as zero/wrong for quantized ops.
================  =================================================  =================

This module centralises detection.  Callers (``trace``,
``log_model_metadata``, ``validate_forward_pass``) invoke
:func:`check_model_and_input_variants` near entry, *before* decoration or
session setup, so failures happen up front with a clear error message
instead of partway through an 18-step pipeline.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from ._distributed import check_distributed_capture
from .errors._base import CompatibilityError, TorchLensWarning
from .utils._torch_compat import get_tracing_tensor_types

# ---------------------------------------------------------------------------
# Per-tensor detectors
# ---------------------------------------------------------------------------


def _is_meta_tensor(t: torch.Tensor) -> bool:
    """True if ``t`` lives on the meta device (no backing storage)."""
    try:
        return t.device.type == "meta"
    except Exception:
        return False


def _is_sparse_tensor(t: torch.Tensor) -> bool:
    """True for any sparse layout (COO, CSR, CSC, BSR, BSC)."""
    # ``layout`` exists on every torch.Tensor; sparse variants are not ``strided``.
    try:
        layout = t.layout
    except Exception:
        return False
    return layout is not torch.strided


def _tracing_tensor_kind(t: torch.Tensor) -> str | None:
    """Return the data-free tracing-subclass name for ``t``, if it is one.

    ``FakeTensor`` and ``FunctionalTensor`` are tensor subclasses used by Dynamo,
    AOTAutograd, ``torch.export``, and functionalization. They carry shape and
    dtype but no storage, so every TorchLens step that reads a value is either
    meaningless or fatal: ``data_ptr()`` on a FakeTensor is a torch-flagged bug,
    and torch's own fake machinery aborts the forward with a bare
    ``AssertionError`` ("Please convert all Tensors to FakeTensors first") the
    moment a real parameter meets a fake activation.

    Parameters
    ----------
    t:
        Tensor to classify.

    Returns
    -------
    str | None
        Class name of the tracing tensor, or ``None`` for an ordinary tensor.
    """
    functional_predicate = getattr(torch, "_is_functional_tensor", None)
    if callable(functional_predicate):
        try:
            if bool(functional_predicate(t)):
                return "FunctionalTensor"
        except (RuntimeError, TypeError):
            pass
    if type(t) is torch.Tensor:
        return None
    tracing_types = get_tracing_tensor_types()
    if tracing_types and isinstance(t, tracing_types):
        return type(t).__name__
    # Structural fallback for builds where the exact classes could not be probed.
    type_name = type(t).__name__
    if type_name in {"FakeTensor", "FunctionalTensor"}:
        return type_name
    return None


def _has_symbolic_shape(t: torch.Tensor) -> bool:
    """True if any dimension is a ``torch.SymInt`` / ``torch.SymFloat``.

    Concrete ``int`` dims are safe.  Symbolic dims arise under
    ``torch._dynamo.mark_dynamic`` / ``torch.export`` traces and break
    metadata collection.
    """
    SymInt = getattr(torch, "SymInt", None)
    SymFloat = getattr(torch, "SymFloat", None)
    if SymInt is None and SymFloat is None:
        return False
    try:
        shape = t.shape
    except Exception:
        return False
    for dim in shape:
        if SymInt is not None and isinstance(dim, SymInt):
            return True
        if SymFloat is not None and isinstance(dim, SymFloat):
            return True
    return False


# ---------------------------------------------------------------------------
# Model-level detectors
# ---------------------------------------------------------------------------


# Quantized module class names — string-match to avoid importing
# ``torch.ao.quantization`` modules when the user doesn't have them compiled in.
_QUANTIZED_MODULE_NAME_PREFIXES: tuple[str, ...] = (
    "torch.ao.nn.quantized",
    "torch.nn.quantized",
    "torch.ao.nn.intrinsic.quantized",
    "torch.ao.nn.qat",
    "torch.nn.qat",
)


def _is_quantized_module(module: nn.Module) -> bool:
    """True if ``module``'s class lives in a quantization namespace."""
    mod_name = type(module).__module__ or ""
    return any(mod_name.startswith(prefix) for prefix in _QUANTIZED_MODULE_NAME_PREFIXES)


def _model_has_quantized_modules(model: nn.Module) -> bool:
    """True if any submodule is a quantized ``nn`` module."""
    return any(_is_quantized_module(sub) for sub in model.modules())


# ---------------------------------------------------------------------------
# Input-tree walk
# ---------------------------------------------------------------------------


class VariantScanTruncationWarning(TorchLensWarning):
    """Emitted when the bounded entry-time tensor scan is truncated.

    The input-tree walk in :func:`_iter_tensors` is bounded (depth and total
    node count) so a pathological or adversarial container cannot stall
    capture entry. When either bound truncates the scan, tensors beyond the
    bound were NOT inspected: an unsupported variant hiding there will not
    receive the typed entry refusal and will instead fail later, mid-capture,
    with a raw error. This category discloses that honestly instead of
    silently narrowing the guarantee.
    """


_ITER_TENSORS_MAX_DEPTH = 128
"""Maximum container-nesting depth inspected by :func:`_iter_tensors`."""

_ITER_TENSORS_MAX_NODES = 4096
"""Maximum total objects inspected by one :func:`_iter_tensors` traversal."""


def _iter_tensors(
    obj: Any,
    _seen: set[int] | None = None,
) -> Iterator[torch.Tensor]:
    """Yield tensors through builtin and inspectable user containers.

    Parameters
    ----------
    obj:
        Root object to inspect.
    _seen:
        Shared object-identity set for cycle prevention.

    Yields
    ------
    torch.Tensor
        Reachable tensor values.

    Notes
    -----
    ``nn.Module`` instances are not descended into because registered state is
    handled separately. Instance ``__dict__`` is read directly, so properties and
    descriptors never execute. Traversal is iterative (an explicit worklist, so
    the depth bound is decoupled from Python's recursion limit) and capped at
    128 levels / 4096 objects; when either bound truncates the scan, a one-shot
    :class:`VariantScanTruncationWarning` disclosure is emitted because
    unsupported variants beyond the bound would fail undetected later. Opaque
    slots-only objects and tensors created later inside ``forward`` remain
    outside entry-time detection and are disclosed in the compatibility report.
    """
    if _seen is None:
        _seen = set()
    nodes = 0
    truncated_by: str | None = None
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _ITER_TENSORS_MAX_DEPTH:
            if truncated_by is None:
                truncated_by = f"depth bound ({_ITER_TENSORS_MAX_DEPTH} nesting levels)"
            continue
        if nodes >= _ITER_TENSORS_MAX_NODES:
            if truncated_by is None:
                truncated_by = f"node bound ({_ITER_TENSORS_MAX_NODES} objects)"
            # The counter never decreases, so every remaining item would be
            # skipped identically — stop instead of draining the worklist.
            break
        obj_id = id(current)
        if obj_id in _seen:
            continue
        _seen.add(obj_id)
        nodes += 1
        if isinstance(current, torch.Tensor):
            yield current
            continue
        if isinstance(current, nn.Module):
            continue
        if isinstance(current, (list, tuple, set, frozenset)):
            children = list(current)
        elif isinstance(current, dict):
            children = list(current.values())
        else:
            try:
                attributes = vars(current)
            except (TypeError, AttributeError):
                continue
            children = list(attributes.values())
        # Reverse so the stack pops children in original order (DFS preorder,
        # matching the recursive traversal this replaced).
        for child in reversed(children):
            stack.append((child, depth + 1))
    if truncated_by is not None:
        warnings.warn(
            "TorchLens entry-time tensor-variant scan was truncated at its "
            f"{truncated_by}: tensors beyond the bound were not inspected, so "
            "unsupported tensor variants hiding there will not be refused up "
            "front and may fail later during capture with a raw error.",
            VariantScanTruncationWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class UnsupportedTensorVariantError(CompatibilityError, RuntimeError):
    """Raised when ``trace`` is called on a model/input combination
    that TorchLens cannot reliably log (see module docstring for the matrix).

    The capture entry gate attaches structured context on ``fields`` so
    callers branch without parsing message text: ``code`` is always
    ``"unsupported_tensor_variant"``, ``remedy`` names the fix, and
    ``offenses`` is a tuple of ``{"name": ..., "reason": ...}`` dicts, one per
    detected variant. Mid-forward shapeless-variant refusals raised outside
    the entry gate (``backends/torch/_ops_activations.py``) do not yet carry
    these fields.
    """


def _docs_pointer(section: str | None = None) -> str:
    """Human-readable pointer to the limitations documentation.

    Parameters
    ----------
    section:
        Optional exact section heading of ``docs/reference/limitations.md``
        to cite; ``None`` points at the catalog as a whole.

    Returns
    -------
    str
        Pointer sentence naming a real documentation location.
    """
    if section is None:
        return "See docs/reference/limitations.md for supported alternatives."
    return (
        f"See the {section!r} section of docs/reference/limitations.md for supported alternatives."
    )


def check_model_and_input_variants(
    model: nn.Module,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
) -> None:
    """Pre-flight check for ``trace``.

    Raises :class:`UnsupportedTensorVariantError` when a fundamentally
    incompatible tensor variant is detected on the model or its inputs, and
    :class:`torchlens._distributed.DistributedCaptureUnsupportedError` when the
    model holds distributed/sharded state that capture would record incorrectly
    rather than fail on.
    Emits :class:`UserWarning` for variants with partial / degraded support
    (quantization) so the user knows what to treat with skepticism in the log.

    Args:
        model: The ``nn.Module`` about to be logged.
        input_args: Positional arguments that will be passed to
            ``model.forward`` (may contain nested containers of tensors).
        input_kwargs: Keyword arguments to ``model.forward``.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Distributed/sharded state is checked first: DTensor parameters otherwise
    # sail past every dense-tensor check below (a DTensor reports a real device
    # and a strided layout) and capture then silently reports zero parameters.
    check_distributed_capture(model, input_args, input_kwargs)

    # SPMD processes that first-capture with distributed already initialized
    # arm collective-boundary capture lazily here (restricted registry seeding,
    # design-merge-ranks-c v5 rule 1.3.2). Explicit torchlens.distributed.arm()
    # at process start remains the required spelling for MPMD programs.
    from .distributed._lifecycle import maybe_auto_arm

    maybe_auto_arm()

    offenses: list[tuple[str, str]] = []

    # Treat a bare tensor and a container of tensors identically — ``_iter_tensors``
    # yields tensors directly for a tensor, or recurses into list/tuple/dict.
    if input_args is None:
        args_payload: Any = []
    elif isinstance(input_args, torch.Tensor):
        args_payload = input_args
    else:
        args_payload = input_args

    # Input-side tensors.
    for t in _iter_tensors(args_payload):
        if _is_meta_tensor(t):
            offenses.append(
                (
                    "meta tensor in input",
                    "Meta tensors have no backing storage, so out saving "
                    "cannot produce usable values.",
                )
            )
        if _is_sparse_tensor(t):
            offenses.append(
                (
                    f"sparse tensor ({t.layout}) in input",
                    "TorchLens' copy/print/FLOPs paths assume dense strided layouts.",
                )
            )
        if _has_symbolic_shape(t):
            offenses.append(
                (
                    "symbolic (SymInt/SymFloat) tensor shape in input",
                    "TorchLens requires concrete integer shapes for metadata and "
                    "counter alignment.",
                )
            )
        tracing_kind = _tracing_tensor_kind(t)
        if tracing_kind is not None:
            offenses.append(
                (
                    f"{tracing_kind} in input",
                    "Tracing tensors carry shape and dtype but no data, so saved "
                    "activations would be empty and torch's own fake-tensor machinery "
                    "aborts the forward as soon as a real parameter meets a fake "
                    "activation. Capture the eager forward on real tensors instead.",
                )
            )
    for t in _iter_tensors(dict(input_kwargs)):
        if _is_meta_tensor(t):
            offenses.append(("meta tensor in keyword input", ""))
        if _is_sparse_tensor(t):
            offenses.append((f"sparse tensor ({t.layout}) in keyword input", ""))
        if _has_symbolic_shape(t):
            offenses.append(("symbolic tensor shape in keyword input", ""))
        tracing_kind = _tracing_tensor_kind(t)
        if tracing_kind is not None:
            offenses.append((f"{tracing_kind} in keyword input", ""))

    # Model params + buffers (dedupe across both generators).
    seen_ids: set[int] = set()
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen_ids:
            continue
        seen_ids.add(id(t))
        if _is_meta_tensor(t):
            offenses.append(
                (
                    "meta tensor among model parameters/buffers",
                    "Meta-init models (e.g. HuggingFace device_map='meta') must be "
                    "materialized on a real device before logging.",
                )
            )
            break  # one message is enough — don't list every param.
        tracing_kind = _tracing_tensor_kind(t)
        if tracing_kind is not None:
            offenses.append(
                (
                    f"{tracing_kind} among model parameters/buffers",
                    "The model was constructed under a fake/functional tracing mode and "
                    "holds no real weights. Build it on a real device before logging.",
                )
            )
            break

    if offenses:
        # Dedupe while preserving order of first appearance.
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for name, why in offenses:
            if name in seen:
                continue
            seen.add(name)
            unique.append((name, why))
        bullet_list = "\n".join(f"  - {name}" + (f": {why}" if why else "") for name, why in unique)
        raise UnsupportedTensorVariantError(
            "torchlens.trace cannot run on this model/input "
            "combination. Detected unsupported tensor variant(s):\n"
            f"{bullet_list}\n"
            f"\n{_docs_pointer('Capture entry and execution contexts')}",
            code="unsupported_tensor_variant",
            remedy=(
                "materialize dense, strided tensors with concrete integer "
                "shapes on a real device before capture"
            ),
            offenses=tuple({"name": name, "reason": why} for name, why in unique),
        )

    # Warnings (non-fatal).
    if _model_has_quantized_modules(model):
        warnings.warn(
            "TorchLens detected quantized submodules. Activation capture "
            "generally works, but FLOPs counts are estimated only for common "
            "quantized Linear/Conv module outputs and out dtype handling is best-effort. "
            f"{_docs_pointer()}",
            UserWarning,
            stacklevel=3,
        )
