"""Transactional state snapshots for function-based native split probes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ._state_snapshot import PythonStateSnapshot


def _restore_tensor(value: Any, snapshot: Any) -> None:
    """Restore a mutable framework tensor when its backend exposes a mutator."""

    assign = getattr(value, "assign", None)
    if callable(assign):
        assign(snapshot)
        return
    set_value = getattr(value, "set_value", None)
    if callable(set_value):
        set_value(snapshot)
        return
    data = getattr(value, "data", None)
    copy_ = getattr(data, "copy_", None)
    if callable(copy_):
        copy_(snapshot)


def _tensor_snapshot(value: Any, adapter: Any) -> Any | None:
    """Copy a mutable backend tensor, or return ``None`` for immutable values."""

    if not adapter.is_tensor(value):
        return None
    if not any(
        callable(getattr(value, name, None)) for name in ("assign", "set_value")
    ) and not callable(getattr(getattr(value, "data", None), "copy_", None)):
        return None
    try:
        return adapter.clone(adapter.detach(value))
    except Exception:
        return None


@contextmanager
def callable_capture_state(model: Any, adapter: Any) -> Iterator[None]:
    """Restore mutable state reachable from a Python function after a probe.

    Parameters
    ----------
    model:
        Function, bound method, or callable state reachable from the function.
    adapter:
        Native split adapter used to identify mutable framework variables.

    Yields
    ------
    None
        A transactional scope for an internal native forward.

    Notes
    -----
    ``deepcopy(function)`` returns the original function, so its globals and
    closure cells remain shared. This scope snapshots those reachable values
    without copying imported modules or executing arbitrary user code.
    """

    tensors: list[tuple[Any, Any]] = []

    def visit_tensor(value: Any, visit: Callable[[Any], None]) -> bool:
        """Snapshot backend-mutable tensors before descending into Python state."""

        snapshot = _tensor_snapshot(value, adapter)
        if snapshot is None:
            return False
        tensors.append((value, snapshot))
        return True

    state = PythonStateSnapshot(visit_tensor)
    state.visit(model)
    try:
        yield
    finally:
        for value, snapshot in tensors:
            _restore_tensor(value, snapshot)
        state.restore()


__all__ = ["callable_capture_state"]
