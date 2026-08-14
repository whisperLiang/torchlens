"""Red-capable gates for security and runnable wire-format single sources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

pytestmark = pytest.mark.smoke


def authority_drift(values: Sequence[Any], *, require_identity: bool) -> tuple[int, ...]:
    """Return indexes whose value diverges from the first authority.

    Parameters
    ----------
    values:
        Ordered authority values with the canonical value first.
    require_identity:
        Whether aliases must be the same object as well as equal.

    Returns
    -------
    tuple[int, ...]
        Indexes of divergent values.
    """

    if not values:
        return ()
    canonical = values[0]
    return tuple(
        index
        for index, value in enumerate(values[1:], start=1)
        if value != canonical or (require_identity and value is not canonical)
    )


def test_security_policy_uses_one_callable_safety_authority() -> None:
    """The unpickler aliases every shared security policy from callable safety."""

    from torchlens._io import _safe_unpickle
    from torchlens.utils import _callable_safety

    authority_pairs = (
        (_callable_safety._DENIED_MODULES, _safe_unpickle._DENIED_FOREIGN_MODULES),
        (_callable_safety._ALLOWED_STDLIB_ROOTS, _safe_unpickle._ALLOWED_STDLIB_ROOTS),
        (
            _callable_safety._STDLIB_AND_BUILTIN_TOP_LEVEL,
            _safe_unpickle._STDLIB_AND_BUILTIN_TOP_LEVEL,
        ),
        (
            _callable_safety.is_denied_stdlib_or_builtin_module,
            _safe_unpickle._stdlib_or_builtin_denied,
        ),
        (_callable_safety._APPLIANCE_MODULES, _safe_unpickle._TORCHLENS_APPLIANCE_MODULES),
    )
    for values in authority_pairs:
        assert not authority_drift(values, require_identity=True)


def test_fenced_appliance_pin_matches_the_canonical_authority() -> None:
    """The resolver's fenced appliance copy stays equal to the canonical set."""

    from torchlens.intervention import resolver
    from torchlens.utils import _callable_safety

    assert not authority_drift(
        (
            _callable_safety._APPLIANCE_MODULES,
            resolver._TORCHLENS_APPLIANCE_MODULES,
        ),
        require_identity=False,
    )


class TestSecurityDriftGateIsRedCapable:
    """Plant policy divergence and prove each comparison can fail."""

    def test_value_drift_is_detected(self) -> None:
        """A one-sided security-set addition is reported."""

        canonical = frozenset({"os", "pickle"})
        planted = canonical | {"subprocess"}
        assert authority_drift((canonical, planted), require_identity=False) == (1,)

    def test_copy_drift_is_detected_when_identity_is_required(self) -> None:
        """An equal-but-independent replacement fails a single-source gate."""

        canonical = frozenset({"os", "pickle"})
        copied = frozenset(value for value in canonical)
        assert copied == canonical
        assert copied is not canonical
        assert authority_drift((canonical, copied), require_identity=True) == (1,)

    def test_callable_drift_is_detected(self) -> None:
        """A reimplemented predicate fails the identity gate."""

        def canonical(value: str) -> bool:
            """Return whether the planted value is denied."""

            return value == "os"

        def planted(value: str) -> bool:
            """Return a deliberately divergent planted decision."""

            return value in {"os", "sys"}

        assert authority_drift((canonical, planted), require_identity=True) == (1,)
