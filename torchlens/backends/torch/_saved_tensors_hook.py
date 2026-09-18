"""Transparent callable wrappers for autograd's mutable saved-hook metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class SavedTensorsHookWrapper:
    """Keep native hook attributes live across TorchLens wrapper layers.

    PyTorch checkpoint hooks attach temporary metadata on entry and delete it
    on exit. Copying the wrapped function's dictionary loses later updates and
    can retain graph references after the native cleanup. Attribute forwarding
    preserves that lifetime even when checkpoint token wrappers are replaced.
    """

    __slots__ = ("__dict__", "__wrapped__", "_call")
    __tl_saved_tensors_hook_scoped__ = True

    def __init__(self, call: Callable[[Any], Any], wrapped: Callable[[Any], Any]) -> None:
        """Store the invocation adapter and the native metadata owner."""
        object.__setattr__(self, "_call", call)
        object.__setattr__(self, "__wrapped__", wrapped)

    def __call__(self, value: Any) -> Any:
        """Invoke the adapter with one packed or unpacked value."""
        return self._call(value)

    def __getattr__(self, name: str) -> Any:
        """Read native metadata from the wrapped hook without copying it."""
        return getattr(object.__getattribute__(self, "__wrapped__"), name)

    def __copy__(self) -> SavedTensorsHookWrapper:
        """Preserve the identity of this closure just like a function wrapper."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> SavedTensorsHookWrapper:
        """Keep copied model contexts bound to the closure's original hook."""
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep TorchLens markers local and forward native metadata writes."""
        if name.startswith("__tl_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__wrapped__, name, value)

    def __delattr__(self, name: str) -> None:
        """Apply native cleanup to the same owner that received the metadata."""
        if name.startswith("__tl_"):
            object.__delattr__(self, name)
        else:
            delattr(self.__wrapped__, name)
