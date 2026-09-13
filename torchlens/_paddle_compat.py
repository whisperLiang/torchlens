"""Opt-in, Linux-local isolation of Paddle's LLVM symbols from TensorFlow.

This module deliberately imports neither TorchLens nor any tensor framework. It
can also run as the standalone copy installed by ``scripts/paddle_import_compat.py``.
Installing the finder does not load Paddle: only a TensorFlow-first import of the
real Paddle extension with a bundled CINN library receives local deep binding.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from importlib.machinery import ExtensionFileLoader, ModuleSpec, PathFinder
from types import ModuleType
from typing import Any

_GUARD_ID = "torchlens.paddle_llvm.v1"
_EXTENSION = "paddle.base.libpaddle"


class _PaddleExtensionLoader:
    """Delegate extension loading after locally binding its native dependencies.

    Parameters
    ----------
    original:
        The extension loader selected by Python's ordinary path finder.
    """

    def __init__(self, original: ExtensionFileLoader) -> None:
        self.original = original
        self._handle: Any = None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        """Bind the complete Paddle DSO before Python runs its native initializer.

        Parameters
        ----------
        spec:
            The original module spec, with only its loader replaced by this proxy.

        Returns
        -------
        ModuleType
            The module created by the original extension loader.

        Notes
        -----
        Loading only ``libcinnapi.so`` cannot work: its PIR symbols depend on
        ``libpaddle.so``. Keep the complete DSO's handle alive and let failures
        propagate; falling back to an unprotected load can terminate the process.
        This never changes the interpreter-wide ``sys.setdlopenflags`` setting.
        """

        import ctypes

        self._handle = ctypes.CDLL(
            self.original.path,
            mode=os.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_DEEPBIND,
        )
        return self.original.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        """Execute the original extension loader without replacing its behavior.

        Parameters
        ----------
        module:
            Native module returned by ``create_module``.
        """

        self.original.exec_module(module)

    def __getattr__(self, name: str) -> Any:
        """Forward loader inspection and optional methods to the original loader.

        Parameters
        ----------
        name:
            Attribute requested by Python or another import consumer.
        """

        return getattr(self.original, name)


class _PaddleImportGuard:
    """Intercept only the standard Paddle extension in the TF-first case."""

    # The installed standalone copy and the repository copy share one finder.
    guard_id = _GUARD_ID

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Keep the ordinary import spec, wrapping only the affected native loader.

        Parameters
        ----------
        fullname:
            Absolute name requested by Python's import machinery.
        path:
            The parent package's search path.
        target:
            Existing module supplied during reload, if any.

        Returns
        -------
        ModuleSpec or None
            The guarded spec, or no intervention for an unrelated import/build.
        """

        if fullname != _EXTENSION or "tensorflow" not in sys.modules:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if (
            spec is None
            or not isinstance(spec.loader, ExtensionFileLoader)
            or not isinstance(spec.origin, str)
        ):
            return None
        # Derive the library from Python's resolved extension, never a global
        # library search or a user-controlled override. CPU/non-CINN builds need
        # no compatibility intervention.
        package_dir = os.path.dirname(os.path.dirname(spec.origin))
        if not os.path.isfile(os.path.join(package_dir, "libs", "libcinnapi.so")):
            return None
        spec.loader = _PaddleExtensionLoader(spec.loader)  # type: ignore[assignment]
        return spec


def install_import_guard() -> bool:
    """Install the lazy guard once, without importing any optional runtime.

    Returns
    -------
    bool
        Whether the guard is installed. Unsupported platforms and import systems
        without the standard path finder are left unchanged.
    """

    if sys.platform != "linux" or not hasattr(os, "RTLD_DEEPBIND"):
        return False
    if any(getattr(finder, "guard_id", None) == _GUARD_ID for finder in sys.meta_path):
        return True
    # Respect all custom meta-path finders ahead of the standard path finder.
    # The guard only proxies the extension that PathFinder itself would load.
    for index, finder in enumerate(sys.meta_path):
        if finder is PathFinder:
            sys.meta_path.insert(index, _PaddleImportGuard())
            return True
    return False


def remove_import_guard() -> None:
    """Remove this guard, without attempting to unload an already loaded DSO.

    Notes
    -----
    Removing a startup hook takes effect in new interpreters. An existing
    process retains any native libraries it already loaded.
    """

    sys.meta_path[:] = [
        finder for finder in sys.meta_path if getattr(finder, "guard_id", None) != _GUARD_ID
    ]
