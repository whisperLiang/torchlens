"""Shared ownership marker and pause bracket for TorchLens dispatch modes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch.utils._python_dispatch import TorchDispatchMode

from ...utils._torch_compat import get_current_dispatch_mode_stack


class _TorchLensDispatchMode(TorchDispatchMode):
    """Marker base for Python dispatch modes owned by TorchLens."""


def _exit_own_dispatch_modes() -> tuple[_TorchLensDispatchMode, ...] | None:
    """Exit top-contiguous TorchLens dispatch modes from the active stack.

    Returns
    -------
    tuple[_TorchLensDispatchMode, ...] | None
        Exited modes in inner-to-outer order, or ``None`` when the stack is
        unreadable or no TorchLens-owned mode is at its top.
    """

    stack = get_current_dispatch_mode_stack()
    if not stack:
        return None
    exited: list[_TorchLensDispatchMode] = []
    while stack and isinstance(stack[-1], _TorchLensDispatchMode):
        mode = stack.pop()
        mode.__exit__(None, None, None)
        exited.append(mode)
    return tuple(exited) or None


def _reenter_own_dispatch_modes(exited: tuple[_TorchLensDispatchMode, ...]) -> None:
    """Re-enter TorchLens modes exited by :func:`_exit_own_dispatch_modes`.

    Parameters
    ----------
    exited:
        Modes in inner-to-outer exit order.
    """

    for mode in reversed(exited):
        mode.__enter__()


@contextmanager
def pause_own_dispatch_modes() -> Iterator[tuple[_TorchLensDispatchMode, ...]]:
    """Pause removable TorchLens modes and restore the exact stack on exit.

    Yields
    ------
    tuple[_TorchLensDispatchMode, ...]
        Exact mode instances paused in inner-to-outer order. An empty tuple
        means stack introspection was unavailable or a foreign mode blocked
        access to every owned mode.
    """

    stack_before = get_current_dispatch_mode_stack()
    exited = _exit_own_dispatch_modes() or ()
    try:
        yield exited
    finally:
        _reenter_own_dispatch_modes(exited)
        stack_after = get_current_dispatch_mode_stack()
        if stack_before is not None and stack_after is not None:
            if len(stack_after) != len(stack_before):
                raise RuntimeError("TorchLens dispatch-mode pause changed the active stack depth")
            if any(
                after is not before for before, after in zip(stack_before, stack_after, strict=True)
            ):
                raise RuntimeError(
                    "TorchLens dispatch-mode pause changed active stack identity or order"
                )
