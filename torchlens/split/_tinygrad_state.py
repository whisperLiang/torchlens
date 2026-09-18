"""Isolate model-owned tinygrad state during split's internal forwards."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ._state_snapshot import PythonStateSnapshot


@contextmanager
def tinygrad_capture_state(model: Any) -> Iterator[None]:
    """Restore reachable model attributes and isolate mutable tensor storage.

    Parameters
    ----------
    model
        Callable object, bound method, or Python function whose referenced
        globals, closure cells, and defaults may own tinygrad model state.

    Yields
    ------
    None
        An internal capture scope with private, alias-preserving state buffers.

    Notes
    -----
    Copying a Python function returns the same function and does not isolate
    its model. In particular, a canonical B=1 capture must not leave a newly
    initialized batch-shaped cache for the subsequent B=2 witness. Existing
    cache attributes are retained, never guessed or reset by name.
    """

    from tinygrad import Tensor

    tensors: list[tuple[Any, Any, Any, Any]] = []

    def visit_tensor(value: Any, visit: Callable[[Any], None]) -> bool:
        """Retain native handles and discover tensor-owned gradient state."""

        if not isinstance(value, Tensor):
            return False
        tensors.append((value, value.uop, value.grad, value.is_param))
        visit(value.grad)
        return True

    state = PythonStateSnapshot(visit_tensor, preserve_tl=True)
    state.visit(model)
    try:
        # Raw STORE can mutate storage without replacing the Tensor's UOp.
        # Captures keep private buffers; original handles are restored below.
        _isolate_buffers(tensors)
        yield
    finally:
        for tensor, uop, grad, is_param in tensors:
            tensor.uop, tensor.grad, tensor.is_param = uop, grad, is_param
        state.restore()


def _isolate_buffers(tensors: list[tuple[Any, Any, Any, Any]]) -> None:
    """Give reachable tinygrad tensors private, alias-preserving native buffers."""

    from tinygrad import Tensor
    from tinygrad.uop.ops import Ops, UOp

    replacements: dict[Any, Any] = {}
    private_buffers: dict[int, Any] = {}
    for _tensor, uop, _grad, _is_param in tensors:
        for node in uop.toposort():
            if node.op is not Ops.BUFFER or node in replacements:
                continue
            buffer = node.buffer
            if id(buffer) not in private_buffers:
                if buffer.is_allocated():
                    private_buffers[id(buffer)] = Tensor(node).clone().realize().uop
                else:
                    private_buffers[id(buffer)] = UOp.new_buffer(node.device, node.size, node.dtype)
            replacements[node] = private_buffers[id(buffer)]
    for tensor, uop, _grad, _is_param in tensors:
        if replacements:
            tensor.uop = uop.substitute(replacements)
