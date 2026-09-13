"""Split-only tinygrad capture must not leak model-state changes."""

from __future__ import annotations

import types
from functools import partial
from typing import Any

import pytest

from torchlens.split._tinygrad_state import tinygrad_capture_state

_STATE_MODULE: Any = None
_STATE_CACHE: Any = None


def _global_state_forward(x: Any) -> Any:
    """Reach a module through a global binding rather than a closure."""

    return _STATE_MODULE(x)


def _global_cache_forward(x: Any) -> Any:
    """Rebind one referenced global without replacing its entire namespace."""

    global _STATE_CACHE
    _STATE_CACHE = x
    return x


@pytest.mark.parametrize("callable_form", ["object", "bound", "closure", "global", "partial"])
@pytest.mark.parametrize("fails", [False, True])
def test_tinygrad_capture_restores_reachable_model_state(callable_form: str, fails: bool) -> None:
    """Native writes and added cache attrs remain private, including on failure."""

    tinygrad = pytest.importorskip("tinygrad")
    Tensor = tinygrad.Tensor

    class StatefulModule:
        """Hold aliased native storage and mutable plain Python state."""

        def __init__(self) -> None:
            """Initialize persistent values before the capture scope."""

            self.buffer = Tensor([1.0, 2.0], device="CPU").realize()
            self.alias = self.buffer.reshape(1, 2)
            self.calls = 0
            self.history: list[Any] = []
            self.options = {"items": ["original"]}

        def __call__(self, x: Any) -> Any:
            """Mutate state using a raw STORE without reassigning the buffer UOp."""

            self.calls += 1
            self.history.append(x)
            self.options["items"].append("changed")
            self.cache = Tensor.zeros(x.shape[0], 2, device="CPU")
            value = Tensor(self.buffer.uop.after(self.buffer.uop.store((self.buffer + 10).uop)))
            return value.realize()

    module = StatefulModule()
    original_buffer = module.buffer
    original_uop = module.buffer.uop
    original_alias_uop = module.alias.uop
    original_options = module.options
    original_items = module.options["items"]

    def closure(x: Any) -> Any:
        """Reach the module through a lexical closure."""

        return module(x)

    model = {
        "object": module,
        "bound": module.__call__,
        "closure": closure,
        "global": types.FunctionType(_global_state_forward.__code__, {"_STATE_MODULE": module}),
        "partial": partial(module),
    }[callable_form]
    try:
        with tinygrad_capture_state(model):
            output = model(Tensor.ones(1, 2, device="CPU"))
            assert output.tolist() == [11.0, 12.0]
            assert module.alias.tolist() == [[11.0, 12.0]]
            if fails:
                raise RuntimeError("native forward failed")
    except RuntimeError as exc:
        assert fails and str(exc) == "native forward failed"

    assert module.buffer is original_buffer
    assert module.buffer.uop is original_uop
    assert module.alias.uop is original_alias_uop
    assert module.buffer.tolist() == [1.0, 2.0]
    assert module.alias.tolist() == [[1.0, 2.0]]
    assert not hasattr(module, "cache")
    assert module.calls == 0
    assert module.history == []
    assert module.options is original_options
    assert module.options["items"] is original_items
    assert module.options == {"items": ["original"]}


def test_tinygrad_capture_retains_existing_batch_cache() -> None:
    """State isolation never silently resets pre-existing batch-specific state."""

    tinygrad = pytest.importorskip("tinygrad")
    Tensor = tinygrad.Tensor

    class CachedModule:
        """Model a lazily created, batch-shaped native cache."""

        def __init__(self) -> None:
            """Start with a cache created by an earlier user forward."""

            self.cache = Tensor([3.0, 4.0], device="CPU").realize()

        def __call__(self, x: Any) -> Any:
            """Read the cache without replacing or reinitializing it."""

            return x + self.cache

    model = CachedModule()
    original_cache = model.cache
    original_uop = original_cache.uop
    with tinygrad_capture_state(model):
        assert model.cache is original_cache
        assert model(Tensor.ones(2, device="CPU")).tolist() == [4.0, 5.0]
        model.cache.assign(model.cache + 1).realize()

    assert model.cache is original_cache
    assert model.cache.uop is original_uop
    assert model.cache.tolist() == [3.0, 4.0]


@pytest.mark.parametrize("binding_kind", ["closure", "global"])
@pytest.mark.parametrize("initial_binding", ["missing", "none", "tensor"])
@pytest.mark.parametrize("fails", [False, True])
def test_tinygrad_capture_restores_rebound_function_state(
    binding_kind: str, initial_binding: str, fails: bool
) -> None:
    """Restore existing and initially absent slots on success and exceptions.

    Parameters
    ----------
    binding_kind:
        Reach model state through a closure cell or a function global.
    initial_binding:
        Start from an absent slot, an uninitialized cache, or an existing tensor.
    fails:
        Raise from the internal forward before the scope restores its state.
    """

    Tensor = pytest.importorskip("tinygrad").Tensor
    original = Tensor([3.0, 4.0], device="CPU").realize() if initial_binding == "tensor" else None
    cache = original

    def closure(x: Any) -> Any:
        """Initialize or replace a directly bound batch-shaped cache."""

        nonlocal cache
        cache = x
        return x

    assert closure.__closure__ is not None
    cell = closure.__closure__[0]
    namespace = {"_STATE_CACHE": original, "unrelated": "original"}
    if initial_binding == "missing":
        del cell.cell_contents
        del namespace["_STATE_CACHE"]
    model = (
        closure
        if binding_kind == "closure"
        else types.FunctionType(_global_cache_forward.__code__, namespace)
    )
    try:
        with tinygrad_capture_state(model):
            replacement = Tensor.ones(1, 2, device="CPU")
            assert model(replacement) is replacement
            # Unreferenced globals are not part of the transaction.
            namespace["unrelated"] = "changed"
            namespace["new_unrelated"] = "added"
            if fails:
                raise RuntimeError("native forward failed")
    except RuntimeError as exc:
        assert fails and str(exc) == "native forward failed"

    assert namespace["unrelated"] == "changed"
    assert namespace["new_unrelated"] == "added"
    if binding_kind == "closure":
        if initial_binding == "missing":
            with pytest.raises(ValueError, match="empty"):
                _ = cell.cell_contents
        else:
            assert cell.cell_contents is original
    elif initial_binding == "missing":
        assert "_STATE_CACHE" not in namespace
    else:
        assert namespace["_STATE_CACHE"] is original
    if original is not None:
        assert original.tolist() == [3.0, 4.0]


@pytest.mark.parametrize("fails", [False, True])
def test_tinygrad_capture_preserves_distinct_uops_with_one_native_buffer(fails: bool) -> None:
    """Keep native aliases and live tensor handles through temporary storage.

    Parameters
    ----------
    fails:
        Abort the internal forward after writing through an aliased buffer.
    """

    Tensor = pytest.importorskip("tinygrad").Tensor
    from tinygrad.uop.ops import UOp

    first = Tensor([1.0, 2.0], device="CPU").realize()
    second = Tensor(UOp.from_buffer(first.uop.buffer))
    grad = Tensor([0.5, 0.5], device="CPU").realize()
    first.grad = grad
    first.is_param_(False)
    original_first_uop, original_second_uop, original_grad_uop = first.uop, second.uop, grad.uop
    model = types.SimpleNamespace(first=first, second=second)
    assert first.uop is not second.uop
    assert first.uop.buffer is second.uop.buffer

    try:
        with tinygrad_capture_state(model):
            assert model.first is first
            assert model.second is second
            assert first.uop.buffer is second.uop.buffer
            assert first.uop.buffer is not original_first_uop.buffer
            output = Tensor(first.uop.after(first.uop.store((first + 10).uop))).realize()
            assert output.tolist() == [11.0, 12.0]
            assert second.tolist() == [11.0, 12.0]
            first.grad = None
            first.is_param_(True)
            if fails:
                raise RuntimeError("native forward failed")
    except RuntimeError as exc:
        assert fails and str(exc) == "native forward failed"

    assert model.first is first
    assert model.second is second
    assert first.uop is original_first_uop
    assert second.uop is original_second_uop
    assert first.grad is grad
    assert grad.uop is original_grad_uop
    assert first.is_param is False
    assert first.tolist() == second.tolist() == [1.0, 2.0]
