"""Functional-collective resolver spellings preserve group identity or refuse."""

from __future__ import annotations

from typing import Any

import pytest

from torchlens.backends.torch.funcol import _resolve_funcol_process_group
from torchlens.utils import _torch_compat as tc

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def isolate_resolver_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore process-wide capability caches after each synthetic probe."""

    for attr in (
        "HAS_FUNCOL_GROUP_RESOLUTION",
        "_FUNCOL_GROUP_RESOLVERS",
        "_FUNCOL_GROUP_RESOLUTION_PROBED",
    ):
        monkeypatch.setattr(tc, attr, getattr(tc, attr))
    monkeypatch.setattr(tc, "_warned_missing_capabilities", set())
    monkeypatch.delenv(tc._CAPABILITY_WARNING_ENV, raising=False)


@pytest.mark.parametrize("resolver_spelling", ["_resolve_group", "_resolve_group_name"])
def test_supported_resolver_spellings_preserve_group_identity(
    monkeypatch: pytest.MonkeyPatch, resolver_spelling: str
) -> None:
    """Both reviewed resolver APIs yield the exact live process-group object."""

    live_group = object()
    calls: list[tuple[Any, ...]] = []

    def resolve_group(group: Any, tag: str) -> Any:
        """Record arguments and expose the spelling's native return form."""

        calls.append((group, tag))
        return live_group if resolver_spelling == "_resolve_group" else "group-name"

    def resolve_name(name: str) -> Any:
        """Resolve only the exact known group name."""

        assert name == "group-name"
        return live_group

    attrs = {
        ("torch.distributed._functional_collectives", resolver_spelling): resolve_group,
        ("torch.distributed.distributed_c10d", "_resolve_process_group"): resolve_name,
    }
    monkeypatch.setattr(tc, "_import_module_attr_or_none", lambda mod, attr: attrs.get((mod, attr)))
    assert tc.get_funcol_group_resolvers(force_probe=True) == (resolve_group, resolve_name)
    assert tc.HAS_FUNCOL_GROUP_RESOLUTION
    assert _resolve_funcol_process_group("input-group", "tag") is live_group
    assert calls == [("input-group", "tag")]


@pytest.mark.parametrize("missing", ["group", "name", "both"])
def test_missing_group_resolution_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Neither incomplete resolver pairs nor absent APIs grant the capability."""

    def resolve(value: Any, *args: Any) -> Any:
        """Stand in for the one resolver that may still exist."""

        return value

    attrs = {
        ("torch.distributed._functional_collectives", "_resolve_group_name"): resolve,
        ("torch.distributed.distributed_c10d", "_resolve_process_group"): resolve,
    }
    if missing in {"group", "both"}:
        del attrs[("torch.distributed._functional_collectives", "_resolve_group_name")]
    if missing in {"name", "both"}:
        del attrs[("torch.distributed.distributed_c10d", "_resolve_process_group")]
    monkeypatch.setattr(tc, "_import_module_attr_or_none", lambda mod, attr: attrs.get((mod, attr)))
    with pytest.warns(tc.TorchCapabilityWarning, match="HAS_FUNCOL_GROUP_RESOLUTION"):
        assert tc.get_funcol_group_resolvers(force_probe=True) is None
    assert not tc.HAS_FUNCOL_GROUP_RESOLUTION
    assert _resolve_funcol_process_group("unknown-group", "") is None
