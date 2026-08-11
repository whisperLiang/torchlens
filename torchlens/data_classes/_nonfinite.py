"""Shared non-finite activation scan behind a revalidated per-log memo.

``print(trace)`` (``interface._str_after_pass``), ``Trace._repr_html_``, and
``report.explain`` all answer "is any saved activation non-finite?", and each
answer used to cost a full ``torch.isfinite`` pass over every saved activation
in the capture -- the whole forward's payload, re-read from scratch on every
repr (67M elements for a resnet18 at batch 8), with four separate copies of the
same loop. The scan itself is unchanged here: same sequence, same order, same
skip rules, same ``detach()``/``isfinite`` kernel, same propagated exceptions.
It is only memoized per log, and the memo is revalidated against the object
identity plus autograd version counter of every tensor the recorded scan
examined, so a mutated, replaced, added, or removed activation falls back to a
real scan rather than serving a stale verdict.

Revalidation deliberately re-reads the same ``out`` attributes in the same
order as the scan it replaces, so lazy materialization, reference-mutation
tripwires, and ``ValueError`` on unsaved payloads all still fire at the same
layer they always did.

The memo is exactly as sharp as ``Tensor._version``, which is already the
mutation oracle TorchLens itself trusts (``MutatedReferenceError`` is raised off
the same counter). A write that deliberately bypasses the autograd version
counter -- ``layer.out.data[0] = float("nan")``, a write through a retained
``numpy()``/``untyped_storage()`` view -- therefore does not invalidate a memo
recorded before it, and a repeated question can still report the pre-write
verdict. This is the same host-write-through-a-detached-handle class the
runnable contract documents as out of scope (``docs/reference/
runnable_tlspec_contract.md`` section 11); an ordinary in-place op, a payload
replacement, and a first question asked after the write are all seen normally.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Iterator, NamedTuple

import torch


class _ScanMemo(NamedTuple):
    """One recorded scan: what it examined, what it found, how far it got."""

    keys: tuple[tuple[weakref.ref, int | None], ...]
    hits: tuple[int, ...]
    complete: bool


# Keyed by log object so a memo never keeps a Trace alive, and holding only
# weakrefs to the examined tensors so it never keeps an activation alive either.
_MEMOS: "weakref.WeakKeyDictionary[Any, dict[str, _ScanMemo]]" = weakref.WeakKeyDictionary()


def _trace_out(layer: Any) -> Any:
    """Read a layer's out payload directly, letting unsaved reads raise."""

    return getattr(layer, "out", None)


def _saved_out(layer: Any) -> Any:
    """Return a layer's saved output payload, or ``None`` when unavailable.

    The report anomaly scans only reason about *saved* activation payloads. On a
    selective-save trace most layers retain no payload, and reading ``.out`` on
    such an op raises ``ValueError`` (``"... was not saved; no saved payload is
    available"``) -- a per-pass ``ValueError`` that a plain ``getattr(layer,
    "out", None)`` cannot swallow, so both the JSON and prose reports previously
    crashed on the ordinary predicate-save trace shape instead of honoring their
    documented ``unknown``/scoped-clean contract. Gate on the saved-payload flag
    first, then read the property inside the known-unavailable boundary so an
    unsaved op is honestly skipped rather than aborting the whole report.

    Parameters
    ----------
    layer:
        A per-pass operation/layer entry from ``log.layer_list``.

    Returns
    -------
    Any
        The saved output tensor when a payload was retained, else ``None``.
    """

    if not bool(getattr(layer, "has_saved_activation", False)):
        return None
    try:
        return getattr(layer, "out", None)
    except ValueError:
        return None


def _layer_list(log: Any) -> Any:
    """Return the finalized layer sequence of a log."""

    return getattr(log, "layer_list", []) or []


def _raw_layers(log: Any) -> Any:
    """Return the raw pre-postprocessing layer sequence of a partial capture."""

    return getattr(log, "raw_layers", ()) or ()


# Scan kind -> (sequence getter, out-payload gate). ``"trace"`` is the
# ``Trace.first_nonfinite`` contract (unsaved reads raise), ``"saved"`` the
# ``report.explain`` contract (unsaved layers are skipped), ``"raw"`` the
# partial-capture contract over raw layer records.
_KINDS: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "trace": (_layer_list, _trace_out),
    "saved": (_layer_list, _saved_out),
    "raw": (_raw_layers, _trace_out),
}


def _examined(log: Any, kind: str) -> Iterator[tuple[Any, torch.Tensor]]:
    """Yield each (layer, tensor) pair a scan of this kind would inspect."""

    sequence, gate = _KINDS[kind]
    for layer in sequence(log):
        out = gate(layer)
        if not isinstance(out, torch.Tensor) or out.numel() == 0:
            continue
        yield layer, out


def _has_nonfinite(out: torch.Tensor) -> bool:
    """Return whether a tensor holds any NaN or Inf, ``False`` if uncheckable."""

    try:
        return bool((~torch.isfinite(out.detach())).any().item())
    except (RuntimeError, TypeError):
        return False


def _ref(tensor: torch.Tensor) -> weakref.ref | None:
    """Return a weak reference to a tensor, or ``None`` if it forbids one."""

    try:
        return weakref.ref(tensor)
    except TypeError:
        return None


def _scan(log: Any, kind: str, stop_at_first: bool) -> tuple[list[Any], _ScanMemo]:
    """Run a real scan, returning the examined layers and the memo to record."""

    layers: list[Any] = []
    keys: list[tuple[weakref.ref | None, int | None]] = []
    hits: list[int] = []
    complete = True
    for layer, out in _examined(log, kind):
        layers.append(layer)
        keys.append((_ref(out), getattr(out, "_version", None)))
        if _has_nonfinite(out):
            hits.append(len(layers) - 1)
            if stop_at_first:
                complete = False
                break
    return layers, _ScanMemo(tuple(keys), tuple(hits), complete)  # type: ignore[arg-type]


def _revalidate(log: Any, kind: str, memo: _ScanMemo) -> list[Any] | None:
    """Return the examined layers when every recorded tensor is unchanged.

    Parameters
    ----------
    log:
        Log the memo was recorded against.
    kind:
        Scan kind whose sequence and gate to replay.
    memo:
        Previously recorded scan.

    Returns
    -------
    list[Any] | None
        Currently examined layers, positionally matching ``memo.hits``, or
        ``None`` when anything the recorded scan looked at has changed.
    """

    keys = memo.keys
    layers: list[Any] = []
    for layer, out in _examined(log, kind):
        if len(layers) >= len(keys):
            # A complete scan saw every payload; a new one means new evidence.
            return None
        ref, version = keys[len(layers)]
        if ref() is not out or getattr(out, "_version", None) != version:
            return None
        layers.append(layer)
        if not memo.complete and len(layers) == len(keys):
            # The recorded scan stopped here, so later payloads never mattered.
            return layers
    return layers if len(layers) == len(keys) else None


def _store(log: Any, kind: str, memo: _ScanMemo) -> None:
    """Record a memo for this log, skipping logs or tensors that forbid it."""

    if any(ref is None for ref, _ in memo.keys):
        return
    try:
        memos = _MEMOS.setdefault(log, {})
    except TypeError:
        return
    memos[kind] = memo


def _resolve(log: Any, kind: str, stop_at_first: bool) -> list[Any]:
    """Return the non-finite layers of a scan, from the memo when it still holds."""

    try:
        memos = _MEMOS.get(log)
    except TypeError:
        memos = None
    memo = None if memos is None else memos.get(kind)
    if memo is not None and (memo.complete or stop_at_first):
        cached_layers = _revalidate(log, kind, memo)
        if cached_layers is not None:
            return [cached_layers[index] for index in memo.hits]
    layers, memo = _scan(log, kind, stop_at_first)
    _store(log, kind, memo)
    return [layers[index] for index in memo.hits]


def first_nonfinite_layer(log: Any, *, kind: str = "trace") -> Any | None:
    """Return the first layer whose out payload holds a NaN or Inf.

    Parameters
    ----------
    log:
        Trace-like object to scan.
    kind:
        Scan contract: ``"trace"``, ``"saved"``, or ``"raw"``.

    Returns
    -------
    Any | None
        First non-finite layer record, or ``None`` when every examined payload
        is finite.
    """

    hits = _resolve(log, kind, stop_at_first=True)
    return hits[0] if hits else None


def unexamined_payload_count(log: Any, *, kind: str = "saved") -> int:
    """Return how many ops a scan of this kind cannot look at.

    A selective-save capture retains payloads for a chosen subset of ops, so a
    non-finite scan genuinely cannot speak for the rest. Callers report this count
    rather than letting a scoped clean answer read as a whole-capture one.

    Parameters
    ----------
    log:
        Trace-like object to inspect.
    kind:
        Scan contract whose sequence and gate to use.

    Returns
    -------
    int
        Number of ops in the scan sequence holding no readable out payload.

    Notes
    -----
    Counts payload availability only -- it never reads tensor values, so it adds no
    scan cost and cannot invalidate the scan memo.
    """

    sequence, gate = _KINDS[kind]
    unexamined = 0
    for layer in sequence(log):
        try:
            out = gate(layer)
        except ValueError:
            unexamined += 1
            continue
        if out is None:
            unexamined += 1
    return unexamined


def nonfinite_layers(log: Any, *, kind: str = "saved") -> list[Any]:
    """Return every layer whose out payload holds a NaN or Inf.

    Parameters
    ----------
    log:
        Trace-like object to scan.
    kind:
        Scan contract: ``"trace"``, ``"saved"``, or ``"raw"``.

    Returns
    -------
    list[Any]
        Non-finite layer records in scan order.
    """

    return _resolve(log, kind, stop_at_first=False)
