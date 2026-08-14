"""Wrap-state identity shims: keep torch-internal identity checks truthful.

TorchLens wrapping replaces public torch callables with wrapper functions, so
a torch-internal ``x is F.y`` check whose two operands were read at different
wrap epochs silently changes answer once wrappers are installed. The two
broken operand pairings are:

* a HELD reference (a class-def-time default argument, or a user variable
  bound before the first capture) compared against a post-wrap namespace
  read -- the ``TransformerEncoderLayer`` ctor fastpath flag;
* the ORIGINAL function object passed to ``__torch_function__`` by torch's
  C-level protocol compared against a call-time namespace read -- the
  ``CausalBias`` sdpa dispatch and the expanded-weights per-sample-grads
  machinery (which additionally keys handler TABLES at import time, so its
  basis depends on whether the module was first imported before or after the
  wrap).

Census (2026-08-14, installed-source grep over the supported eager range,
identity/equality forms against wrappable callables, runtime paths only)
found exactly these sites; compiler/export/testing namespaces are out of
capture scope by contract. A fourth normalization rides along:
``torch.overrides.resolve_name`` keys its cached index by the pre-warm
originals, so a wrapper argument resolved to ``None`` -- the shim retries a
miss with the ledger original. The standing installed-tree grep gate lives
in ``tests/test_wrap_state_compat.py``; the ``nested/_internal`` NJT
identity reads it surfaces are a documented unshimmed residual (nested
jagged tensors are not supported capture inputs).

Strategy: NEVER re-implement torch's decision logic. Each shim normalizes
the identity operand to the basis the immediately-following torch comparison
uses (the call-time namespace read, or the import-time table key), then
delegates to the original torch code, so the decision itself always runs
upstream logic. Shims install with ``wrap_torch()`` and are removed by
``unwrap_torch()``; with wrappers absent every normalization is an identity
no-op. Site availability is feature-detected in
``torchlens.utils._torch_compat`` (``HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG``,
``HAS_ATTENTION_CAUSAL_BIAS``, ``HAS_EXPANDED_WEIGHTS_CONV_PICKER``) and is
visible through the doctor/compat capability snapshot.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch

from ... import _state
from ...utils import _torch_compat

__all__ = [
    "identity_shims_installed",
    "install_identity_shims",
    "remove_identity_shims",
]

_SHIM_MARKER = "_torchlens_identity_shim"
_MISSING = object()

# (holder, attribute name, original attribute value) for every installed shim.
_installed: list[tuple[Any, str, Any]] = []


def _resolve(fn: Any) -> Any:
    """Follow the wrapper ledger from a torchlens wrapper to its original.

    Parameters
    ----------
    fn:
        Any object; non-wrappers resolve to themselves.

    Returns
    -------
    Any
        The original torch callable for a torchlens wrapper, else ``fn``.
    """

    seen: set[int] = set()
    while id(fn) in _state._decorated_to_orig and id(fn) not in seen:
        seen.add(id(fn))
        fn = _state._decorated_to_orig[id(fn)]
    return fn


def _is_shimmed(value: Any) -> bool:
    """Return whether ``value`` (function or classmethod) is one of our shims."""

    fn = getattr(value, "__func__", value)
    return bool(getattr(fn, _SHIM_MARKER, False))


def identity_shims_installed() -> bool:
    """Return whether the identity shims are currently installed."""

    return bool(_installed)


def install_identity_shims() -> None:
    """Install every census-listed identity shim; idempotent.

    Caller holds the wrapper install lock. A failure mid-install restores the
    already-patched sites and re-raises: a partially shimmed process would be
    a silent-protection lie, and an unexpected error here is a TorchLens bug
    that must surface loudly.
    """

    if _installed:
        return
    records: list[tuple[Any, str, Any]] = []
    try:
        _install_transformer_ctor_shims(records)
        _install_causal_bias_shim(records)
        _install_expanded_weights_shims(records)
        _install_resolve_name_shim(records)
    except Exception:
        _restore(records)
        raise
    _installed.extend(records)


def remove_identity_shims() -> None:
    """Remove all installed identity shims; idempotent."""

    _restore(_installed)
    _installed.clear()


def _restore(records: list[tuple[Any, str, Any]]) -> None:
    """Restore original attributes for ``records``, tolerating drift.

    A site whose current value is no longer our shim (user monkeypatching
    layered on top) is left untouched rather than clobbered, mirroring the
    namespace-drift tolerance of wrapper teardown.
    """

    for holder, name, original in reversed(records):
        current = vars(holder).get(name)
        if current is None or not _is_shimmed(current):
            continue
        try:
            setattr(holder, name, original)
        except (AttributeError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Site 1: TransformerEncoderLayer / TransformerDecoderLayer constructors
# ---------------------------------------------------------------------------


def _install_transformer_ctor_shims(records: list[tuple[Any, str, Any]]) -> None:
    """Shim the transformer layer ctors' activation identity check.

    ``TransformerEncoderLayer.__init__`` decides ``activation_relu_or_gelu``
    (the fused fastpath + nested-tensor gate) with ``activation is F.relu``
    -- the DEFAULT argument is the pre-wrap original bound at class-def time,
    so every post-wrap default construction silently got flag 0 and a
    different forward kernel path. ``TransformerDecoderLayer`` has no flag
    but stores namespace-read activations (the string spelling), so it gets
    the same shim for stored-state wrap invariance.
    """

    if not _torch_compat.HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG:
        return
    for cls_name in ("TransformerEncoderLayer", "TransformerDecoderLayer"):
        cls = getattr(torch.nn, cls_name, None)
        if cls is None:
            continue
        orig_init = vars(cls).get("__init__")
        if orig_init is None or _is_shimmed(orig_init):
            continue
        try:
            sig = inspect.signature(orig_init)
        except (TypeError, ValueError):
            continue
        if "activation" not in sig.parameters:
            continue
        cls.__init__ = _make_ctor_shim(orig_init, sig)
        records.append((cls, "__init__", orig_init))
        # ``__setstate__`` injects the CURRENT ``F.relu`` -- the live wrapper
        # while wrapped -- when unpickling legacy state that lacks
        # ``activation`` (encoder writes the attribute after delegating,
        # decoder patches the state dict before). Normalize the stored object
        # afterwards so legacy unpickles are wrap-invariant too.
        orig_setstate = vars(cls).get("__setstate__")
        if orig_setstate is not None and not _is_shimmed(orig_setstate):
            cls.__setstate__ = _make_setstate_shim(orig_setstate)
            records.append((cls, "__setstate__", orig_setstate))


def _make_ctor_shim(
    orig_init: Callable[..., None], sig: inspect.Signature
) -> Callable[..., None]:
    """Build the ctor shim for one transformer layer class."""

    activation_default = sig.parameters["activation"].default

    @functools.wraps(orig_init)
    def ctor_shim(self: Any, *args: Any, **kwargs: Any) -> None:
        import torch.nn.functional as F

        try:
            bound = sig.bind(self, *args, **kwargs)
        except TypeError:
            # Let torch's own signature error surface unchanged.
            orig_init(self, *args, **kwargs)
            return
        activation = bound.arguments.get("activation", _MISSING)
        effective = activation_default if activation is _MISSING else activation
        namespace_form = None
        if callable(effective):
            resolved = _resolve(effective)
            for name in ("relu", "gelu"):
                current = getattr(F, name, None)
                if current is not None and resolved is _resolve(current):
                    namespace_form = current
                    break
        if namespace_form is not None and effective is not namespace_form:
            # Pass the object the interior `activation is F.relu/F.gelu`
            # check reads, so torch's own logic decides the flag correctly.
            bound.arguments["activation"] = namespace_form
        orig_init(*bound.args, **bound.kwargs)
        stored = getattr(self, "activation", None)
        if callable(stored):
            resolved_stored = _resolve(stored)
            if resolved_stored is not stored:
                # Store what an unwrapped construction stores: the original
                # torch function, never a torchlens wrapper (keeps module
                # state byte-identical across wrap states and pickle-clean).
                self.activation = resolved_stored

    setattr(ctor_shim, _SHIM_MARKER, True)
    return ctor_shim


def _make_setstate_shim(orig_setstate: Callable[..., None]) -> Callable[..., None]:
    """Build the ``__setstate__`` shim for one transformer layer class."""

    @functools.wraps(orig_setstate)
    def setstate_shim(self: Any, state: Any) -> None:
        orig_setstate(self, state)
        stored = getattr(self, "activation", None)
        if callable(stored):
            resolved_stored = _resolve(stored)
            if resolved_stored is not stored:
                # Store what an unwrapped unpickle stores: the original torch
                # function, never a torchlens wrapper.
                self.activation = resolved_stored

    setattr(setstate_shim, _SHIM_MARKER, True)
    return setstate_shim


# ---------------------------------------------------------------------------
# Site 2: torch.nn.attention.bias.CausalBias.__torch_function__
# ---------------------------------------------------------------------------


def _install_causal_bias_shim(records: list[tuple[Any, str, Any]]) -> None:
    """Shim CausalBias's sdpa identity dispatch.

    The C-level protocol passes the ORIGINAL sdpa as ``func``; torch compares
    it against the call-time (wrapped) namespace read. The miss silently fell
    through to the default Tensor path, DROPPING the causal mask and
    returning a broken ``CausalBias``-typed result.
    """

    if not _torch_compat.HAS_ATTENTION_CAUSAL_BIAS:
        return
    causal_bias = _torch_compat._import_module_attr_or_none(
        "torch.nn.attention.bias", "CausalBias"
    )
    if causal_bias is None:
        return
    orig_classmethod = vars(causal_bias).get("__torch_function__")
    if orig_classmethod is None or _is_shimmed(orig_classmethod):
        return
    orig_tf = orig_classmethod.__func__

    @functools.wraps(orig_tf)
    def causal_bias_shim(
        cls: type,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        import torch.nn.functional as F

        target = getattr(F, "scaled_dot_product_attention", None)
        if target is not None and func is not target and _resolve(func) is _resolve(target):
            func = target
        return orig_tf(cls, func, types, args, kwargs)

    setattr(causal_bias_shim, _SHIM_MARKER, True)
    causal_bias.__torch_function__ = classmethod(causal_bias_shim)
    records.append((causal_bias, "__torch_function__", orig_classmethod))


# ---------------------------------------------------------------------------
# Site 3: torch.nn.utils._expanded_weights (per-sample-grads machinery)
# ---------------------------------------------------------------------------


def _install_expanded_weights_shims(records: list[tuple[Any, str, Any]]) -> None:
    """Shim the expanded-weights dispatch bases.

    ``ExpandedWeight.__torch_function__`` mixes two comparison bases: handler
    TABLES keyed at module import time (original keys when imported pre-wrap,
    wrapper keys when imported post-wrap) and one call-time NAMESPACE read
    (``func is torch._cudnn_rnn_flatten_weight``). ``conv_picker`` then
    re-reads ``F.conv1d/2d/3d`` from the namespace. Each shim maps ``func``
    to the alias its next comparison actually uses.
    """

    if not _torch_compat.HAS_EXPANDED_WEIGHTS_CONV_PICKER:
        return
    import importlib

    try:
        conv_utils = importlib.import_module(
            "torch.nn.utils._expanded_weights.conv_utils"
        )
        conv_expanded = importlib.import_module(
            "torch.nn.utils._expanded_weights.conv_expanded_weights"
        )
        impl = importlib.import_module(
            "torch.nn.utils._expanded_weights.expanded_weights_impl"
        )
    except ImportError:
        return

    # conv_picker is imported BY VALUE into conv_expanded_weights at torch
    # import time, so both module attributes need the shim.
    for module in (conv_utils, conv_expanded):
        orig_picker = getattr(module, "conv_picker", None)
        if orig_picker is None or _is_shimmed(orig_picker):
            continue
        setattr(module, "conv_picker", _make_conv_picker_shim(orig_picker))  # noqa: B010
        records.append((module, "conv_picker", orig_picker))

    expanded_weight = getattr(impl, "ExpandedWeight", None)
    if expanded_weight is None:
        return
    orig_classmethod = vars(expanded_weight).get("__torch_function__")
    if orig_classmethod is None or _is_shimmed(orig_classmethod):
        return
    orig_tf = orig_classmethod.__func__

    @functools.wraps(orig_tf)
    def expanded_weight_shim(
        cls: type,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        flatten = getattr(torch, "_cudnn_rnn_flatten_weight", None)
        if flatten is not None and func is not flatten and _resolve(func) is _resolve(flatten):
            # The special case reads the namespace at call time; hand it the
            # namespace object (it is in no handler table under either alias).
            func = flatten
        else:
            rnn_decomps = getattr(impl, "expanded_weights_rnn_decomps", {})
            handled = getattr(cls, "handled_functions", {})
            if func not in rnn_decomps and func not in handled:
                alias = _state._orig_to_decorated.get(id(func))
                if alias is None:
                    alias = _state._decorated_to_orig.get(id(func))
                if alias is not None and (alias in rnn_decomps or alias in handled):
                    func = alias
        return orig_tf(cls, func, types, args, kwargs)

    setattr(expanded_weight_shim, _SHIM_MARKER, True)
    expanded_weight.__torch_function__ = classmethod(expanded_weight_shim)
    records.append((expanded_weight, "__torch_function__", orig_classmethod))


# ---------------------------------------------------------------------------
# Site 4: torch.overrides.resolve_name
# ---------------------------------------------------------------------------


def _install_resolve_name_shim(records: list[tuple[Any, str, Any]]) -> None:
    """Shim ``torch.overrides.resolve_name`` to the table's original-key basis.

    ``resolve_name`` looks the callable up in the cached overridable-functions
    index, which is keyed by the objects the namespaces held when the cache
    first materialized (the pre-wrap ORIGINALS once ``decorate_all_once``
    pre-warms both tables). A user or third-party tool passing the CURRENT
    namespace read -- the torchlens wrapper -- silently got ``None`` instead
    of the name. The shim retries a ``None`` miss with the ledger-resolved
    original, so the answer matches unwrapped eager torch under either alias.
    """

    overrides_module = getattr(torch, "overrides", None)
    if overrides_module is None:
        return
    orig_resolve = vars(overrides_module).get("resolve_name")
    if orig_resolve is None or _is_shimmed(orig_resolve):
        return

    @functools.wraps(orig_resolve)
    def resolve_name_shim(f: Any) -> Any:
        result = orig_resolve(f)
        if result is None:
            original = _resolve(f)
            if original is not f:
                result = orig_resolve(original)
        return result

    setattr(resolve_name_shim, _SHIM_MARKER, True)
    overrides_module.resolve_name = resolve_name_shim
    records.append((overrides_module, "resolve_name", orig_resolve))


def _make_conv_picker_shim(orig_picker: Callable[..., Any]) -> Callable[..., Any]:
    """Build a conv_picker shim normalizing ``func`` to the namespace basis."""

    @functools.wraps(orig_picker)
    def conv_picker_shim(func: Any, conv1d_opt: Any, conv2d_opt: Any, conv3d_opt: Any) -> Any:
        import torch.nn.functional as F

        for name in ("conv1d", "conv2d", "conv3d"):
            current = getattr(F, name, None)
            if current is not None and func is not current and _resolve(func) is _resolve(current):
                func = current
                break
        return orig_picker(func, conv1d_opt, conv2d_opt, conv3d_opt)

    setattr(conv_picker_shim, _SHIM_MARKER, True)
    return conv_picker_shim
