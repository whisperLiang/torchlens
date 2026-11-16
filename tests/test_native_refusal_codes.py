"""Concrete native provenance failures keep their structured refusal causes."""

from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.backends._errors import BackendUnsupportedError
from torchlens.backends.paddle import _cuda
from torchlens.backends.tinygrad import _module_discovery


def test_missing_cuda_driver_has_a_native_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driver load failures remain typed before any Paddle execution."""

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise OSError("driver missing")

    loader = "WinDLL" if sys.platform == "win32" else "CDLL"
    monkeypatch.setattr(ctypes, loader, unavailable)
    with pytest.raises(BackendUnsupportedError) as raised:
        _cuda._cuda_driver.__wrapped__()
    assert raised.value.code == "cuda_driver_unavailable"


def test_failed_cuda_driver_query_keeps_its_cause() -> None:
    """An unsuccessful context query is not reported as a tensor capture gap."""

    with pytest.raises(BackendUnsupportedError, match="driver status 7") as raised:
        _cuda._check_cuda(7, "cuCtxGetCurrent")
    assert raised.value.code == "cuda_context_inspection_failed"


@pytest.mark.parametrize(
    "owners,code",
    [([None], "cuda_context_unavailable"), ([11, 22], "cuda_context_mismatch")],
)
def test_bad_cuda_storage_ownership_refuses_before_execution(
    monkeypatch: pytest.MonkeyPatch, owners: list[int | None], code: str
) -> None:
    """Absent and inconsistent owners cannot reach the native execution body."""

    class Tensor:
        def __init__(self, index: int) -> None:
            self.index = index
            self.place = SimpleNamespace(is_gpu_place=lambda: True, gpu_device_id=lambda: 0)

        def _is_initialized(self) -> bool:
            return True

        def data_ptr(self) -> int:
            return self.index + 1

    class Driver:
        def cuPointerGetAttribute(self, output: Any, attribute: int, pointer: int) -> int:
            assert attribute == 1
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = owners[pointer - 1]
            return 0

    monkeypatch.setitem(
        sys.modules,
        "paddle",
        SimpleNamespace(
            Tensor=Tensor, is_compiled_with_rocm=lambda: False, is_compiled_with_cuda=lambda: True
        ),
    )
    monkeypatch.setattr(_cuda, "_cuda_driver", lambda: Driver())
    with (
        pytest.raises(BackendUnsupportedError) as raised,
        _cuda.paddle_cuda_scope([Tensor(i) for i in range(len(owners))]),
    ):
        pytest.fail("Unowned or conflicting storage reached native execution")
    assert raised.value.code == code


def test_unordered_module_address_collision_is_a_typed_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indistinguishable unordered modules cannot receive capture-dependent IDs."""

    class Tensor:
        pass

    class Block:
        def __init__(self) -> None:
            self.weight = Tensor()

        def __call__(self, value: Any) -> Any:
            return value

    monkeypatch.setattr(
        _module_discovery, "_is_tinygrad_tensor", lambda value: isinstance(value, Tensor)
    )
    model = SimpleNamespace(blocks={Block(), Block()})
    with pytest.raises(BackendUnsupportedError) as raised:
        _module_discovery._iter_tinygrad_model_leaves(model, "self")
    assert raised.value.code == "module_identity_ambiguous"
