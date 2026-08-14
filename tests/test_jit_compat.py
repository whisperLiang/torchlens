"""Regression tests for TorchScript compatibility and model patch idempotence."""

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens.backends.torch.wrappers as torch_wrappers
from torchlens import _state


class _AttentionStyleModule(nn.Module):
    """Small attention-like module that exercises decorated JIT builtin ops."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run a compact scaled dot-product attention block."""

        scale = float(q.size(-1)) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        return torch.matmul(weights, v)


class _StoredFuncModel(nn.Module):
    """Module with a stale torch function stored on the model instance."""

    def __init__(self, stale_relu: Any) -> None:
        """Initialize the module with a detached function reference."""

        super().__init__()
        self.act = stale_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stored callable."""

        return self.act(x)


@pytest.fixture(autouse=True)
def _ensure_wrapped() -> Iterator[None]:
    """Ensure wrappers are installed and restore their incoming state."""

    was_wrapped = _state._is_decorated
    torch_wrappers.wrap_torch()
    try:
        yield
    finally:
        if not was_wrapped:
            torch_wrappers.unwrap_torch()


def _require_torch_jit() -> None:
    """Skip when TorchScript APIs are unavailable."""

    if not hasattr(torch, "jit") or not hasattr(torch.jit, "script"):
        pytest.skip("torch.jit.script is unavailable")


def test_attention_style_module_scripts_with_decorated_builtins() -> None:
    """TorchScript should parse an attention-style module after wrapping."""

    _require_torch_jit()

    model = _AttentionStyleModule()
    scripted = torch.jit.script(model)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)

    expected = model(q, k, v)
    observed = scripted(q, k, v)

    assert torch.allclose(observed, expected)


def test_jit_builtin_registration_sanitizes_dtype_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JIT builtin registration should survive wrapper annotation sanitizing."""

    _require_torch_jit()
    import torch.jit._builtins as _jit_builtins

    original_cos = _state._decorated_to_orig[id(torch.cos)]
    builtin_name = _jit_builtins._builtin_table[id(original_cos)]
    monkeypatch.setattr(torch.cos, "__annotations__", {"dtype": "DType", "return": Any})

    torch_wrappers._register_jit_builtin_wrappers()

    assert _jit_builtins._builtin_table[id(torch.cos)] == builtin_name
    assert torch.cos.__annotations__["dtype"] is int
    assert not hasattr(torch.cos, "__wrapped__")


def _calls_softsign(x: torch.Tensor) -> torch.Tensor:
    """Call a wrapped pure-Python torch.nn.functional op (not an ATen builtin)."""

    return torch.nn.functional.softsign(x)


def test_jit_script_wrapped_functional_python_op() -> None:
    """TorchScript must compile a function calling a wrapped non-builtin functional op.

    Regression: ``F.softsign`` is a pure-Python functional op absent from torch's jit
    ``_builtin_table``, so torchlens's wrapper is not registered as a builtin. jit then
    pulls softsign's original source but resolves names against the wrapper module's
    globals, which previously lacked the ``torch.overrides`` boilerplate, failing with
    ``undefined value has_torch_function_unary`` (surfaced by spikingjelly
    ``@torch.jit.script`` surrogates in the model menagerie).
    """

    _require_torch_jit()

    scripted = torch.jit.script(_calls_softsign)
    x = torch.randn(8)
    assert torch.allclose(scripted(x), torch.nn.functional.softsign(x))


class _MaxPoolModule(nn.Module):
    """Module calling a boolean-dispatched functional (the F.max_pool* family)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a 2x2 max pool."""

        return torch.nn.functional.max_pool2d(x, kernel_size=2)


class _InterpolateModule(nn.Module):
    """Module calling a pure-Python functional whose source needs extra globals."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2x with nearest-neighbor interpolation."""

        return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")


def test_jit_script_boolean_dispatched_functional_while_wrapped() -> None:
    """TorchScript must compile the boolean-dispatched ``F.max_pool*`` family
    while wrappers are installed.

    Regression: ``torch._jit_internal.boolean_dispatched`` is keyed by the
    ORIGINAL function objects, so the namespace wrapper missed the table and
    jit tried to compile the wrapper's varargs source (``NotSupportedError``).
    """

    _require_torch_jit()

    scripted = torch.jit.script(_MaxPoolModule())
    x = torch.randn(1, 1, 4, 4)
    assert torch.allclose(scripted(x), torch.nn.functional.max_pool2d(x, kernel_size=2))


def test_jit_script_functional_needing_original_globals_while_wrapped() -> None:
    """TorchScript must compile a wrapped pure-Python functional whose source
    resolves names beyond the imported torch.overrides boilerplate.

    Regression: jit pulled ``F.interpolate``'s original source but resolved
    globals against the WRAPPER module (``undefined value math``); the
    ``__prepare_scriptable__`` hook now hands jit the original function with
    its own self-consistent globals.
    """

    _require_torch_jit()

    scripted = torch.jit.script(_InterpolateModule())
    x = torch.randn(1, 1, 4, 4)
    expected = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    assert torch.allclose(scripted(x), expected)
