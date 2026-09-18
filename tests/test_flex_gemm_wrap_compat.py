"""Torch 2.14 callable membership tables stay truthful across wrap epochs."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType

import pytest
import torch

from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

pytestmark = pytest.mark.smoke


@pytest.fixture
def flex_module() -> Iterator[ModuleType]:
    """Load the optional native frontend before installing TorchLens wrappers."""

    name = "torch._higher_order_ops.flex_gemm"
    if importlib.util.find_spec(name) is None:
        pytest.skip("This torch build does not expose FlexGEMM")
    unwrap_torch()
    try:
        yield importlib.import_module(name)
    finally:
        unwrap_torch()
        wrap_torch()


@pytest.mark.parametrize("name", ["mm", "addmm", "bmm", "baddbmm"])
def test_held_flex_frontend_accepts_both_callable_epochs(
    flex_module: ModuleType, name: str
) -> None:
    """A held frontend and either GEMM alias produce the native eager result."""

    frontend = flex_module.flex_gemm
    held_op = getattr(torch, name)
    shape = (2, 2, 2) if "b" in name else (2, 2)
    args = (torch.ones(shape), torch.full(shape, 2.0))
    if "add" in name:
        args = (torch.ones(shape), *args)
    expected = frontend(held_op, args, torch.relu)
    original_call = flex_module.FlexGemm.__call__
    wrap_torch()
    for op in (held_op, getattr(torch, name)):
        torch.testing.assert_close(frontend(op, args, torch.relu), expected)
    with pytest.raises(RuntimeError, match="unsupported GEMM op"):
        frontend(torch.sin, args, torch.relu)
    unwrap_torch()
    assert flex_module.FlexGemm.__call__ is original_call
    torch.testing.assert_close(frontend(held_op, args, torch.relu), expected)


def test_readonly_dlpack_protocol_preserves_export_and_rejection() -> None:
    """Wrapped Python tensor methods still match the native DLPack allowlist."""

    module = importlib.import_module("torch.utils.dlpack")
    cls = getattr(module, "ReadOnlyTensorWrapper", None)
    if cls is None:
        pytest.skip("This torch build does not expose read-only DLPack wrappers")
    unwrap_torch()
    try:
        value = cls(torch.ones(2))
        device = value.__dlpack_device__()
        wrap_torch()
        assert value.__dlpack_device__() == device
        assert type(value.__dlpack__()).__name__ == "PyCapsule"
        with pytest.raises(RuntimeError, match="only supports DLPack export"):
            torch.add(value, 1)
    finally:
        unwrap_torch()
        wrap_torch()


def test_flex_module_loaded_while_wrapped_survives_teardown(flex_module: ModuleType) -> None:
    """Import-hook installation normalizes aliases for the following unwrapped epoch."""

    x = torch.ones(2, 2)
    expected = torch.full((2, 2), 2.0)
    wrap_torch()
    importlib.reload(flex_module)
    frontend = flex_module.flex_gemm
    torch.testing.assert_close(frontend(torch.mm, (x, x), torch.relu), expected)
    unwrap_torch()
    assert torch.mm in flex_module.FLEX_GEMM_OP_ALIASES
    torch.testing.assert_close(frontend(torch.mm, (x, x), torch.relu), expected)
    wrap_torch()
    torch.testing.assert_close(frontend(torch.mm, (x, x), torch.relu), expected)
