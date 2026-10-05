"""Scoped CUDA context ownership for Paddle tensors in mixed-framework processes."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cache
from typing import Any

from ..registry import BackendUnsupportedError


@cache
def _cuda_driver() -> Any:
    """Load the driver lazily and bind only the context and pointer inspection APIs."""

    try:
        if sys.platform == "win32":
            driver = ctypes.WinDLL("nvcuda.dll")
        else:
            driver = ctypes.CDLL("libcuda.so.1")
        driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        for name in ("cuPointerGetAttribute", "cuCtxGetCurrent", "cuCtxSetCurrent"):
            getattr(driver, name).restype = ctypes.c_int
    except (OSError, AttributeError) as exc:
        raise BackendUnsupportedError(
            "Paddle GPU execution requires CUDA driver context inspection."
        ) from exc
    return driver


def _check_cuda(status: int, operation: str) -> None:
    """Refuse an unavailable CUDA context operation before executing Paddle kernels."""

    if status:
        raise BackendUnsupportedError(
            f"Paddle CUDA context operation {operation} failed with driver status {status}."
        )


def _tensor_leaves(values: Any) -> Iterator[Any]:
    """Yield native Paddle tensor leaves without importing or initializing another framework."""

    import paddle

    if isinstance(values, paddle.Tensor):
        yield values
    elif isinstance(values, Mapping):
        for value in values.values():
            yield from _tensor_leaves(value)
    elif isinstance(values, (tuple, list)):
        for value in values:
            yield from _tensor_leaves(value)


@contextmanager
def paddle_cuda_scope(*values: Any) -> Iterator[None]:
    """Execute on the CUDA context owning native tensors, then restore the caller's context.

    Parameters
    ----------
    *values
        Tensor/container arguments and native model state. CPU-only calls and
        non-CUDA builds leave the NVIDIA driver unloaded. Different devices may
        own different contexts; tensors on one device must share an owning context.

    Yields
    ------
    None
        The first CUDA tensor's owning context, or native execution for non-CUDA builds.

    Notes
    -----
    CUDA driver backends such as tinygrad can leave a private context current.
    Paddle's cached cuBLAS handles still belong to the tensors' original context.
    CU_POINTER_ATTRIBUTE_CONTEXT (1) identifies that context from live storage;
    device indices alone cannot distinguish two contexts on the same GPU.
    No context is created or destroyed, and no synchronization is introduced.
    ROCm builds use the same GPUPlace type but retain their native execution path.
    """

    import paddle

    if paddle.is_compiled_with_rocm() or not paddle.is_compiled_with_cuda():
        yield
        return

    driver: Any = None
    owners: dict[int, int] = {}
    seen: set[int] = set()
    for value in _tensor_leaves(values):
        if id(value) in seen or not value._is_initialized() or not value.place.is_gpu_place():
            continue
        seen.add(id(value))
        pointer = int(value.data_ptr())
        if not pointer:
            continue
        driver = _cuda_driver()
        owner = ctypes.c_void_p()
        _check_cuda(
            driver.cuPointerGetAttribute(ctypes.byref(owner), 1, pointer), "cuPointerGetAttribute"
        )
        if owner.value is None:
            raise BackendUnsupportedError("Paddle GPU tensor storage has no owning CUDA context.")
        device = int(value.place.gpu_device_id())
        previous = owners.setdefault(device, owner.value)
        if previous != owner.value:
            raise BackendUnsupportedError(
                f"Paddle tensors on GPU {device} belong to different CUDA contexts. "
                "Create inputs and model state in the same context before capturing or replaying."
            )
    if not owners:
        yield
        return
    current = ctypes.c_void_p()
    _check_cuda(driver.cuCtxGetCurrent(ctypes.byref(current)), "cuCtxGetCurrent")
    owner_context = next(iter(owners.values()))
    changed = current.value != owner_context
    if changed:
        _check_cuda(driver.cuCtxSetCurrent(owner_context), "cuCtxSetCurrent")
    try:
        yield
    finally:
        # A callback may itself switch contexts, even when entry needed no switch.
        _check_cuda(driver.cuCtxSetCurrent(current), "cuCtxSetCurrent restore")
