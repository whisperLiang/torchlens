"""Capability-aware roster entries preserve wrapper epoch and liveness contracts."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
import torch

from torchlens import _state
from torchlens._constants import (
    _get_torch_submodule_alias_funcs,
    get_orig_torch_funcs,
)
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch


def test_mandatory_roster_does_not_hide_disappearing_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capability filtering does not weaken the mandatory-site liveness census."""
    monkeypatch.delattr(torch.nn.functional, "relu")
    assert ("torch.nn.functional", "relu") in get_orig_torch_funcs(include_torchvision=False)


def test_loaded_alias_discovery_never_imports_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unloaded exporter cannot be imported merely to discover wrapper sites."""
    monkeypatch.delitem(sys.modules, "torch.onnx.operators", raising=False)

    def forbid_import(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("alias discovery must not import modules")

    monkeypatch.setattr(importlib, "import_module", forbid_import)
    assert not any(
        module == "torch.onnx.operators" for module, _ in _get_torch_submodule_alias_funcs()
    )


@pytest.mark.parametrize(
    ("alias_name", "torch_name"),
    [
        ("shape_as_tensor", "_shape_as_tensor"),
        ("reshape_from_tensor_shape", "_reshape_from_tensor"),
    ],
)
def test_onnx_alias_imported_while_wrapped_tracks_every_epoch(
    monkeypatch: pytest.MonkeyPatch, alias_name: str, torch_name: str
) -> None:
    """Late ONNX aliases restore originals and reinstall the exact same wrapper."""
    wrap_torch()
    operators = importlib.import_module("torch.onnx.operators")
    wrapped = getattr(torch, torch_name)
    original = _state._decorated_to_orig[id(wrapped)]
    # Deterministically reproduce the module's `alias = torch.<name>` body,
    # including when another test already imported ONNX before decoration.
    monkeypatch.setattr(operators, alias_name, wrapped)
    try:
        unwrap_torch()
        assert getattr(operators, alias_name) is original
        wrap_torch()
        assert getattr(operators, alias_name) is wrapped
        # An already-wrapped process also handles an alias loaded from an
        # original held before wrapping (without redecorating the callable).
        monkeypatch.setattr(operators, alias_name, original)
        wrap_torch()
        assert getattr(operators, alias_name) is wrapped
    finally:
        wrap_torch()
