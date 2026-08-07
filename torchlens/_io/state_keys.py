"""Narrow class-method shadow filter for portable ``__setstate__`` loads.

A portable data class rehydrates from a pickled ``state`` dict via
``self.__dict__.update(state)``. Round 54 ``sec_3`` showed that an attacker who
plants a ``state`` key shadowing a class-owned method (``_internal_set``) turns
the instance attribute into an execution boundary: ``rehydrate`` later
read-then-calls it. The primary close resolves that setter off the CLASS
(``rehydrate._assign_rehydrated_field``); this module is the defense-in-depth
enabler filter -- it refuses, at ``__setstate__`` time, any incoming state key
that shadows a class-owned *plain method / non-descriptor callable*.

The filter is deliberately NARROW. The portable dataclasses expose their real
serialized fields as ``@property`` / slot descriptors, so a *broad* "shadows any
class attribute" filter would reject legitimate state by construction (round-51
over-catch: 336 real ``@property`` collisions). Refusing only callable *methods*
had zero legitimate collisions across all portable classes (probe C2) while still
catching ``_internal_set``/``save``/``run``/``draw`` impersonation.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Mapping
from typing import Any

from ..errors import TorchLensError

# The class-owned callable *method* surface: a state key resolving to one of these
# on the class MRO can never be a legitimate serialized field value, so it is an
# attacker shadow. Descriptors (``property``/``member_descriptor``/getset) are
# EXCLUDED -- real fields are exposed through them.
_METHOD_SHADOW_TYPES: tuple[type, ...] = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    staticmethod,
    classmethod,
)


class PortableStateKeyError(TorchLensError, ValueError):
    """A portable ``__setstate__`` received a key shadowing a class-owned method.

    Raised as a load-integrity tripwire (round 54 ``sec_3``): the state dict of an
    attacker ``.tlspec`` carried a key whose name resolves to a plain method on the
    portable class, which rehydrate would otherwise be able to read-then-call.
    """

    def __init__(self, cls: type, shadowed: list[str]) -> None:
        super().__init__(
            f"Portable {cls.__name__} state carries key(s) {sorted(shadowed)!r} that shadow "
            "class-owned method(s); refusing the load (attacker-planted callable shadow)."
        )
        self.cls = cls
        self.shadowed = tuple(sorted(shadowed))


_MISSING = object()
# Cached "this name does not exist on the class" answer. Distinct from _MISSING so a
# cached absence is a HIT, not an indistinguishable miss that re-resolves forever.
_ABSENT = object()

# ``inspect.getattr_static`` is the execution-free resolver every load-integrity
# static lookup routes through, and it is expensive: it re-walks the class and
# metaclass MROs and re-checks ``__dict__`` shadowing on every call. One portable
# load asks the same (class, name) question hundreds of thousands of times, so the
# answer is memoized per class.
#
# Memoization is only sound while the class DEFINITION is unchanged, so an entry is
# validated against a definition fingerprint -- the entry count of every ``__dict__``
# the resolver consults -- at every cache-validation boundary (see
# :func:`invalidate_static_class_attr_cache`). An attribute added to, or removed
# from, any class in either MRO invalidates that class's entry. What memoization can
# never weaken is the property these lookups exist for: the answer is still derived
# from the CLASS, never from an attacker-controllable instance ``__dict__``.
_STATIC_ATTR_MEMO: dict[type, list[Any]] = {}

# Same discipline one level up: the shadow VERDICT per (class, key), so each key of
# an incoming state dict costs a plain dict lookup.
_SHADOW_VERDICT_MEMO: dict[type, list[Any]] = {}

# Bumped at every load boundary; a memo entry re-validates its fingerprint the first
# time it is used in a new generation, so the per-lookup cost is one int comparison
# instead of a fresh fingerprint.
_CACHE_GENERATION = 0


def invalidate_static_class_attr_cache() -> None:
    """Require every memoized static lookup to re-validate its class fingerprint.

    Called at the load boundaries (:func:`torchlens._io.bundle.load` and
    :func:`torchlens._io.rehydrate.rehydrate_trace`) so a class redefined between
    loads can never be answered from a stale entry.
    """

    global _CACHE_GENERATION
    _CACHE_GENERATION += 1


def _class_definition_fingerprint(cls: type) -> tuple[int, ...] | None:
    """Return a fingerprint of every ``__dict__`` a static lookup consults.

    Returns ``None`` for an exotic class whose MRO cannot be read, which disables
    memoization for that class rather than trusting a stale answer.
    """

    try:
        return tuple(len(klass.__dict__) for klass in inspect.getmro(cls)) + tuple(
            len(klass.__dict__) for klass in inspect.getmro(type(cls))
        )
    except (AttributeError, TypeError):  # pragma: no cover - exotic metaclass
        return None


def _validated_memo_entry(cls: type, memo: dict[type, list[Any]]) -> dict[str, Any] | None:
    """Return ``cls``'s memo dict, resetting it when the class definition changed.

    Entries are ``[generation, fingerprint, answers]``. The fingerprint is rebuilt
    only on the first use of an entry in a new cache generation. Returns ``None``
    when ``cls`` has no readable fingerprint, which disables memoization for it.
    """

    entry = memo.get(cls)
    if entry is not None and entry[0] == _CACHE_GENERATION:
        return entry[2]
    fingerprint = _class_definition_fingerprint(cls)
    if fingerprint is None:  # pragma: no cover - exotic metaclass
        return None
    if entry is None or entry[1] != fingerprint:
        entry = [_CACHE_GENERATION, fingerprint, {}]
        memo[cls] = entry
    else:
        entry[0] = _CACHE_GENERATION
    return entry[2]


def static_class_attr(cls: type, name: str, default: Any = _MISSING) -> Any:
    """Memoized :func:`inspect.getattr_static` for a class-owned attribute.

    Semantically identical to ``inspect.getattr_static(cls, name[, default])``:
    the class MRO is walked WITHOUT triggering any descriptor ``__get__``, so a
    planted instance attribute can never substitute for a class-owned one. The
    only difference is that the answer is cached per class behind a definition
    fingerprint (see :data:`_STATIC_ATTR_MEMO`).
    """

    answers = _validated_memo_entry(cls, _STATIC_ATTR_MEMO)
    if answers is None:  # pragma: no cover - exotic metaclass, never memoized
        resolved = _resolve_static_class_attr(cls, name)
    else:
        resolved = answers.get(name, _MISSING)
        if resolved is _MISSING:
            resolved = _resolve_static_class_attr(cls, name)
            answers[name] = resolved
    if resolved is _ABSENT:
        if default is _MISSING:
            raise AttributeError(name)
        return default
    return resolved


def _resolve_static_class_attr(cls: type, name: str) -> Any:
    """Resolve one static class attribute, returning ``_ABSENT`` when absent."""

    try:
        return inspect.getattr_static(cls, name)
    except AttributeError:
        return _ABSENT


def _is_descriptor(attr: Any) -> bool:
    """Return whether ``attr`` implements the descriptor protocol on its type."""

    attr_type = type(attr)
    return (
        hasattr(attr_type, "__get__")
        or hasattr(attr_type, "__set__")
        or hasattr(attr_type, "__delete__")
    )


def _key_shadows_class_method(cls: type, key: str) -> bool:
    """Return whether ``key`` resolves to a class-owned plain method on ``cls``.

    Uses ``inspect.getattr_static`` so the class MRO is walked WITHOUT triggering
    any descriptor ``__get__`` (execution-free). A key absent from the class, or
    resolving to a ``property`` / data descriptor / plain value, is a legitimate
    field and returns ``False``.
    """

    try:
        attr = static_class_attr(cls, key)
    except AttributeError:
        return False
    if isinstance(attr, _METHOD_SHADOW_TYPES):
        return True
    # A plain callable class attribute that is NOT itself a type and NOT a
    # descriptor (e.g. a bare callable object) is also a method-like shadow; a
    # ``property`` is a descriptor and is therefore NOT caught here.
    return callable(attr) and not isinstance(attr, type) and not _is_descriptor(attr)


def refuse_callable_shadowing_state_keys(cls: type, state: Mapping[str, Any]) -> None:
    """Refuse a portable ``state`` dict that shadows any class-owned method of ``cls``.

    Call immediately before ``self.__dict__.update(state)`` in every portable
    ``__setstate__``. Narrow by design (probe C2: 0 legitimate collisions), it
    closes the read-then-call enabler class for every non-slotted portable class at
    once without touching the ``@property``-backed real state keys.
    """

    verdicts = _validated_memo_entry(cls, _SHADOW_VERDICT_MEMO)
    if verdicts is None:  # pragma: no cover - exotic metaclass, resolve every key
        shadowed = [
            key for key in state if isinstance(key, str) and _key_shadows_class_method(cls, key)
        ]
    else:
        shadowed = []
        for key in state:
            if not isinstance(key, str):
                continue
            verdict = verdicts.get(key)
            if verdict is None:
                verdict = _key_shadows_class_method(cls, key)
                verdicts[key] = verdict
            if verdict:
                shadowed.append(key)
    if shadowed:
        raise PortableStateKeyError(cls, shadowed)
