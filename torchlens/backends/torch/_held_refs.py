"""Held torch-function reference normalization for ``release_model``.

A plain module attribute holding a torch function (``self.act = F.relu``)
captures whichever object -- pristine original or wrap-epoch wrapper -- the
name resolved to at construction time. Pickle serializes functions by
reference with a strict identity check against the CURRENT module attribute,
so a held ref from the other wrap state fails ``pickle``/``torch.save`` with
``PicklingError`` in both directions. ``release_model`` normalizes each held
ref to the value currently live at its public name so the released model is
serializable NOW (split out of ``model_prep.py`` under the R43 file-size
ratchet -- this is the release-time serializability seam, not preparation).
"""

from typing import Any

from torch import nn

from ... import _state

__all__ = ["normalize_held_torch_function_refs"]


def _live_counterpart(value: Any) -> Any | None:
    """Resolve the value currently LIVE at a held torch function's public name.

    ``value`` is a plain instance attribute captured from a model tree. When
    it is a ledgered torch-function wrapper OR a ledgered original whose
    public torch name currently resolves to its exact wrap-epoch counterpart,
    return that live counterpart; otherwise return ``None`` (leave the
    attribute alone). The swap is ledger-fenced BOTH ways: a foreign patch or
    an unstamped (class-namespace) wrapper never matches and is never
    touched.
    """

    decorated_to_orig, orig_to_decorated = _state.wrap_epoch_ledgers()
    wrapper: Any = None
    if callable(value):
        if id(value) in decorated_to_orig:
            wrapper = value
        elif id(value) in orig_to_decorated:
            wrapper = orig_to_decorated[id(value)]
    if wrapper is None:
        return None
    namespace_name = getattr(wrapper, "__module__", None)
    func_name = getattr(wrapper, "__qualname__", None)
    if not namespace_name or not func_name or "." in func_name:
        return None
    from .wrappers import get_optional_torch_namespace

    namespace_obj = get_optional_torch_namespace(namespace_name)
    if namespace_obj is None or isinstance(namespace_obj, type):
        return None
    current = getattr(namespace_obj, func_name, None)
    if current is None or current is value:
        return None
    # Only the exact wrap-epoch counterpart qualifies: the held wrapper's own
    # original, or the wrapper installed over the held original.
    if (
        decorated_to_orig.get(id(current)) is decorated_to_orig.get(id(value), value)
        or decorated_to_orig.get(id(value)) is current
    ):
        return current
    return None


def normalize_held_torch_function_refs(module: nn.Module) -> None:
    """Re-point epoch-mismatched torch-function attrs at their live values.

    A plain module attribute holding a torch function (``self.act = F.relu``)
    captures whichever object -- pristine original or wrap-epoch wrapper --
    the name resolved to at construction time. Pickle serializes functions
    by reference with a strict identity check against the CURRENT module
    attribute, so a held ref from the other wrap state fails
    ``pickle``/``torch.save`` with ``PicklingError`` in both directions.
    Normalizing each held ref to the value currently live at its public name
    makes the released model serializable NOW (and a fresh-process load
    resolves the public torch name either way). The sweep is bounded: direct
    ``__dict__`` values plus one level of builtin list/tuple/dict/set
    containers; refs inside closures, partials, or custom objects are a
    disclosed residual, as are bare references held outside the model.
    """

    for attr_name, attr_value in tuple(module.__dict__.items()):
        replacement = _live_counterpart(attr_value)
        if replacement is not None:
            module.__dict__[attr_name] = replacement
        elif isinstance(attr_value, (list, dict)):
            _swap_live_refs_inplace(attr_value)
        elif isinstance(attr_value, tuple):
            swapped = _live_swapped_tuple(attr_value)
            if swapped is not None:
                module.__dict__[attr_name] = swapped
        elif isinstance(attr_value, set):
            _swap_live_refs_in_set(attr_value)


def _swap_live_refs_inplace(container: list[Any] | dict[Any, Any]) -> None:
    """Re-point epoch-mismatched refs inside one list/dict, preserving identity."""

    items = enumerate(container) if isinstance(container, list) else container.items()
    for key, item in tuple(items):
        live = _live_counterpart(item)
        if live is not None:
            container[key] = live


def _live_swapped_tuple(values: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Return a live-ref rebuild of one tuple, or ``None`` when nothing changed."""

    swapped = tuple(_live_counterpart(item) or item for item in values)
    if any(new is not old for new, old in zip(swapped, values)):
        return swapped
    return None


def _swap_live_refs_in_set(container: set[Any]) -> None:
    """Re-point epoch-mismatched refs inside one set, preserving identity."""

    for item in tuple(container):
        live = _live_counterpart(item)
        if live is not None:
            container.discard(item)
            container.add(live)
