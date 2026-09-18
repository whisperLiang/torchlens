"""Namespace scan fast leaves must preserve mutable-slot and subclass cleanup."""

from __future__ import annotations

import types
from collections import deque
from typing import Any

import pytest
import torch

from torchlens.backends.torch import model_prep


@pytest.mark.parametrize("leaf", [None, 1, "text", len, lambda: None, object, types])
def test_leaf_replacement_is_inspected_on_every_namespace_walk(leaf: Any) -> None:
    """Replacing an exact fast leaf at unchanged size must expose its new tensor."""

    namespace = types.ModuleType("_tl_test_mutable_namespace")
    namespace.payload = leaf
    visited: list[torch.Tensor] = []
    model_prep._clear_session_tensor_metadata(namespace, set(), visit=visited.append)
    assert visited == []
    count = len(vars(namespace))
    tensor = torch.ones(2)
    namespace.payload = {"nested": deque([(tensor,)])}
    assert len(vars(namespace)) == count
    model_prep._clear_session_tensor_metadata(namespace, set(), visit=visited.append)
    assert len(visited) == 1
    assert visited[0] is tensor


def test_namespace_filter_keeps_tensor_and_container_subclasses() -> None:
    """Only exact leaves may bypass the original subclass-aware type checks."""

    class TensorSubclass(torch.Tensor):
        """Tensor subclass reached as a direct module attribute."""

    class ListSubclass(list[torch.Tensor]):
        """Container subclass reached as a direct module attribute."""

    direct = torch.ones(2).as_subclass(TensorSubclass)
    nested = torch.zeros(2)
    namespace = types.ModuleType("_tl_test_subclass_namespace")
    namespace.direct = direct
    namespace.nested = ListSubclass([nested])
    namespace.parameter = torch.nn.Parameter(torch.ones(2))
    visited: list[torch.Tensor] = []
    model_prep._clear_session_tensor_metadata(namespace, set(), visit=visited.append)
    assert {id(value) for value in visited} == {id(direct), id(nested)}


def test_known_namespace_leaves_skip_hostile_shim_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fast path saves per-entry checks rather than caching stale slot names."""

    namespace = types.ModuleType("_tl_test_leaf_namespace")
    namespace.function = lambda: None
    namespace.builtin = len
    namespace.klass = object
    namespace.number = 1
    tensor = torch.ones(2)
    namespace.tensor = tensor
    checked: list[Any] = []
    original = model_prep._is_isinstance_hostile_deprecation_shim

    def observe(value: Any) -> bool:
        """Record which values enter the original guarded classification path."""

        checked.append(value)
        return original(value)

    monkeypatch.setattr(model_prep, "_is_isinstance_hostile_deprecation_shim", observe)
    visited: list[torch.Tensor] = []
    model_prep._clear_session_tensor_metadata(namespace, set(), visit=visited.append)
    assert {id(value) for value in checked} == {id(namespace), id(tensor)}
    assert len(visited) == 1 and visited[0] is tensor
