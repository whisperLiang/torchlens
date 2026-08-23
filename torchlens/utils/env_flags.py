"""Closed-vocabulary parsing for torchlens-owned boolean environment knobs.

THE one boolean env parser (round-7 b7 R47): every torchlens-owned on/off
knob must parse against the same closed vocabulary, because the historical
exact-``"1"`` / raw-truthiness spellings turned a typo into a silently
different configuration -- ``TORCHLENS_COLLAPSE_STRICT=true`` left a
verification tripwire DISARMED while the exporter believed it was armed, and
``TORCHLENS_DEBUG_FORK_COPY=0`` ENABLED the debug behavior it names off.
The vocabulary and refuse-on-unrecognized behavior mirror the postprocess
audit knob parser (``torchlens/postprocess/__init__.py``), the reviewed
precedent.
"""

from __future__ import annotations

import os

from .._errors import InvalidArgumentError

__all__ = ["closed_bool_env"]

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def closed_bool_env(name: str, *, default: bool = False) -> bool:
    """Parse a torchlens-owned boolean env var against a closed vocabulary.

    Parameters
    ----------
    name:
        Environment variable name (``TORCHLENS_*``).
    default:
        Value when the variable is unset or empty (the only implicit spelling).

    Returns
    -------
    bool
        ``True`` for ``1/true/yes/on``, ``False`` for ``0/false/no/off``
        (case-insensitive, surrounding whitespace ignored).

    Raises
    ------
    InvalidArgumentError
        When the value is set but unrecognized (``env_flag_invalid``). A knob
        whose typo silently selects one of its two states is a disarmed
        tripwire; refusing is the only honest reading.
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return default
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise InvalidArgumentError(
        f"{name}={raw!r} is not a recognized value",
        code="env_flag_invalid",
        remedy=(
            "use '1'/'true'/'yes'/'on' to enable, '0'/'false'/'no'/'off' to "
            "disable explicitly, or unset the variable"
        ),
        argument=name,
    )
