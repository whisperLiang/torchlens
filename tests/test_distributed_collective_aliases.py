"""Collective aliases are identified by callable identity, not attribute names."""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from types import ModuleType
from typing import Any

import pytest
import torch

from torchlens.backends.torch import collectives
from torchlens.distributed import _lifecycle as lifecycle

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not torch.distributed.is_available(), reason="torch.distributed unavailable"
    ),
]


def _unrelated_reduce(value: Any) -> Any:
    """Return a value without performing any distributed operation."""

    return value


@pytest.mark.parametrize("collision", [reduce, _unrelated_reduce], ids=["builtin", "python"])
@pytest.mark.parametrize("reverse_order", [False, True], ids=["canonical-first", "alias-first"])
def test_only_identity_proven_collective_aliases_are_wrapped(
    monkeypatch: pytest.MonkeyPatch, collision: Callable[..., Any], reverse_order: bool
) -> None:
    """Wrap both canonical APIs and genuine aliases, preserving unrelated names."""

    def public_reduce(tensor: Any, dst: int = 0) -> Any:
        """Represent an independently patched public collective entry."""

        return tensor

    def implementation_reduce(tensor: Any, dst: int = 0) -> Any:
        """Represent the canonical c10d collective implementation."""

        return tensor

    dist = torch.distributed
    c10d = dist.distributed_c10d
    monkeypatch.setattr(dist, "reduce", public_reduce)
    monkeypatch.setattr(c10d, "reduce", implementation_reduce)
    site = next(site for site in collectives.COLLECTIVE_SITES if site.attr == "reduce")
    monkeypatch.setattr(collectives, "COLLECTIVE_SITES", (site,))
    public_alias = ModuleType("public_collective_alias")
    public_alias.reduce = public_reduce
    private_alias = ModuleType("private_collective_alias")
    private_alias.reduce = implementation_reduce
    unrelated = ModuleType("unrelated_name_collision")
    unrelated.reduce = collision
    modules = [dist, c10d, public_alias, private_alias, unrelated]
    if reverse_order:
        modules.reverse()
    monkeypatch.setattr(lifecycle, "_patch_modules", lambda: modules)
    expected = {
        (dist, "reduce"): public_reduce,
        (c10d, "reduce"): implementation_reduce,
        (public_alias, "reduce"): public_reduce,
        (private_alias, "reduce"): implementation_reduce,
    }
    originals: dict[tuple[Any, str], Any] = {}
    try:
        collectives.install_collective_wraps(originals)
        assert originals == expected
        installed = {module: module.reduce for module in modules}
        for (module, attr), original in expected.items():
            assert getattr(module, attr).__wrapped__ is original
            assert getattr(module, attr).__tl_distributed_wrap__
        assert unrelated.reduce is collision
        collectives.install_collective_wraps(originals)
        assert all(module.reduce is installed[module] for module in modules)
    finally:
        collectives.remove_collective_wraps(originals)
    assert not originals
    assert all(getattr(module, attr) is original for (module, attr), original in expected.items())
    assert unrelated.reduce is collision
