"""Shared helpers for additive public-API deprecations."""

from __future__ import annotations

import warnings
from typing import Final


class MissingType:
    """Sentinel type used to detect explicitly supplied public kwargs.

    Notes
    -----
    Public APIs in this sprint must distinguish ``caller omitted this kwarg``
    from ``caller explicitly passed the public default``. A dedicated sentinel
    keeps those cases separate without relying on value comparisons.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a stable debugging representation."""

        return "MISSING"


class TorchLensDeprecationWarning(DeprecationWarning):
    """Warning category for TorchLens's own public-API deprecations.

    A ``DeprecationWarning`` subclass, so every user-side filter keyed on
    ``DeprecationWarning`` keeps working unchanged. The dedicated subclass
    exists so tooling can separate "TorchLens deprecated one of its own
    spellings" from "some other library, or torch, emitted a deprecation" --
    which a message-shape filter cannot do.

    This matters for the test gate specifically. Warning filters can select on
    the module a warning is *attributed* to, but correct attribution now points
    at the CALLER (see :func:`warn_deprecated_alias`), so a module-keyed rule
    can no longer recognize a TorchLens deprecation at all. Keying on this
    category instead makes the gate independent of call depth. Mirrors the
    ``torchlens._io.ArtifactSchemaAgeWarning`` precedent.
    """


MISSING: Final[MissingType] = MissingType()

#: The single removal window TorchLens advertises for its 2.x shims. Every
#: deprecation route quotes THIS text, so the package cannot promise two
#: different windows for two spellings of the same debt. Deliberately prose:
#: no concrete version has been chosen, and choosing one is a maintainer
#: decision (see ``tests/test_deprecation_inventory.py``), not something a
#: warning message may invent.
REMOVED_IN: Final[str] = "a future 2.x release"

#: Ledger of alias deprecations this process has emitted, as
#: ``"<old>-><new>"`` keys. A record, NOT a suppression gate: suppression is
#: left to Python's own per-call-site warning registry, which honours
#: ``-W always`` and ``simplefilter`` the way a hand-rolled set cannot.
#: Read by the deprecation inventory to check that every spelling the runtime
#: actually emits is a registered one.
_WARNED_DEPRECATIONS: set[str] = set()


def warn_deprecated_alias(old: str, new: str) -> None:
    """Emit a deprecation warning for an old public name, blaming the caller.

    A deprecation warning is only actionable if it points at the user's own
    line, and a hardcoded ``stacklevel`` cannot do that from here: this helper
    is reached from call depths that differ by several frames (a top-level shim
    warns two frames below the user, flat-kwarg resolution warns from deep
    inside capture, and the module ``__getattr__`` route adds a custom
    ``__getattribute__`` frame). The former fixed ``stacklevel=6`` resolved to
    ``sys:1`` for the shallow routes -- which Python's default
    ``__main__``-keyed filter hides entirely -- and to a TorchLens-internal
    frame for the deep ones. ``user_stacklevel`` resolves it per call instead.

    Parameters
    ----------
    old:
        Deprecated public name.
    new:
        Canonical replacement name.
    """

    from .utils.display import user_stacklevel

    _WARNED_DEPRECATIONS.add(f"{old}->{new}")
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead. "
        f"The old name continues to work but will be removed in {REMOVED_IN}.",
        TorchLensDeprecationWarning,
        # Resolved from THIS frame outward: `user_stacklevel` counts from its
        # own caller, which is exactly what `warnings.warn` here wants.
        stacklevel=user_stacklevel(),
    )
